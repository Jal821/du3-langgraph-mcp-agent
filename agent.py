"""The agent: a ReAct loop whose every tool arrives over MCP.

`create_agent` from langchain 1.x is a ReAct agent - it compiles to a LangGraph
with a model node, a tool node, and a conditional edge that sends control back to
the model for as long as it keeps asking for tools. `graph.png` is that graph.

What is worth looking at here is where the tools come from. Not one @tool
decorator in this project. Instead:

    catalogue   a local MCP server, started as a subprocess, spoken to over stdio
    web         Tavily's hosted MCP server, over streamable HTTP, if a key is set

Both are the same kind of thing to the agent, which is the argument for MCP over
framework-specific tools: the catalogue server is a standalone program that
Claude Desktop or n8n can use unchanged, and Tavily's tools required writing no
integration code at all - only a URL.

The local session is opened ONCE and held for the whole run. `client.get_tools()`
is the shorter route, but it starts a fresh MCP session per tool call, which for
a stdio server means re-spawning the subprocess and re-opening SQLite on every
single call. Over a five-call conversation that is most of the wall clock.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

import catalogue
from model import get_model

HERE = Path(__file__).parent

# The ReAct loop is bounded. Left open, a model that keeps re-searching instead
# of answering will happily spend a hundred calls doing it. Ten covers
# describe -> research -> two or three searches -> quote -> recover from a quote
# error -> answer, with a call or two spare. It was eight until a real run spent
# all eight walking a prerequisite chain one level at a time; the chain is now
# reported in full on the first error, and the ceiling has room anyway.
MAX_MODEL_CALLS = 10

# LangGraph's own recursion limit counts SUPERSTEPS, not model calls, and every
# middleware adds nodes to the loop: one iteration here is before_model, model,
# after_model and tools. So a recursion limit of 25 - which looks generous next
# to eight calls - actually cuts the run off at six, and does it by raising
# GraphRecursionError instead of by ending the agent politely.
#
# Derived from MAX_MODEL_CALLS rather than written down, so the two cannot drift
# apart. The intent is that ModelCallLimitMiddleware is always the binding
# limit and this one never fires.
RECURSION_LIMIT = MAX_MODEL_CALLS * 4 + 8

SYSTEM_PROMPT = """\
You put together AI training offers for small companies.

The catalogue of modules and the rate card are not in your memory. They are in
tools. Use them.

## How to work

1. Call describe_catalogue ONCE at the start, before anything else. It gives you
   the real track names, levels and rate card.
2. If the client's line of business matters and you do not already know what
   such a business does all day, look it up - wikipedia_lookup for what a trade
   involves, web search for anything current. Do not research for its own sake;
   one lookup is usually enough, and skip it entirely for a plainly described
   need.
3. Find modules with find_modules, using the client's own words for the query.
   Search more than once with different wording if the first result set is thin.
   find_modules already returns the COMPLETE record for every module it found,
   prerequisites included, so do not call get_module on those again - you have
   the data. get_module is only for an id you have not seen yet.
4. Price the package with quote. Pass every module id you intend to include,
   together with their prerequisites: a module's prerequisites are listed in the
   record you already have, so include them in the FIRST quote call rather than
   discovering them from an error.

## Rules

- Never do arithmetic. Not hours, not totals, not discounts, not VAT. Every
  number you state must have come out of quote. If you have not called quote,
  you do not have a price and must not name one.
- Never invent a module, an id, a price or an hourly rate. If the catalogue does
  not cover something the client asked for, say so plainly.
- A tool error is not the end of the turn. quote in particular tells you exactly
  what to change - a missing prerequisite to add, an online-only module to drop.
  Fix it and call quote again. Only report a failure if you cannot fix it.
- Respect the budget if one is given. If the package you would recommend costs
  more, say so and offer a shorter package that fits, priced properly.
- This applies to alternatives too. If you offer the client a second option -
  "or the whole thing online", "or without module X" - either call quote for
  that option as well, or describe it with NO figures and say it can be priced
  on request. An unpriced alternative is fine. A guessed one is not.
- Answer in the language the user wrote in.

## The answer

Write the offer itself, addressed to the client. Your first line is already part
of the offer, so it must not be about you: no "Now I have the quotes", no "Let me
present", no "Actually, let me reconsider". If a sentence describes what you are
doing rather than what the client gets, delete it before you answer.

Keep it short and concrete:

- one or two sentences on what you understood the need to be
- the modules, as a list: id, name, hours
- the price: total including VAT, and the lines that make it up
- anything the client has to decide, as a short list

No sales language. If you looked something up, name the source.
"""


def build_connections() -> dict:
    """The MCP servers this agent talks to.

    The web server is conditional: without a Tavily key the agent still runs on
    the catalogue and Wikipedia alone. Degrading to fewer tools is much better
    than refusing to start, and it means a marker with no API key can still see
    the thing work.
    """
    connections: dict = {
        "catalogue": {
            "transport": "stdio",
            # sys.executable, not "uv run": this is already the project's
            # interpreter, and it saves a uv startup on every launch.
            "command": sys.executable,
            "args": [str(HERE / "mcp_server.py")],
            "cwd": str(HERE),
        }
    }

    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        connections["web"] = {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_key}",
        }

    return connections


@asynccontextmanager
async def build_agent(verbose: bool = True):
    """Yield a ready ReAct agent and the tools it loaded.

    A context manager because the stdio session owns a subprocess. Leaving the
    block shuts the MCP server down; without that the process would outlive the
    script on Windows.
    """
    # Checked here, on the client side, rather than only in the server. The
    # server does report a missing database on stderr and exit - but a stdio MCP
    # server that exits during startup reaches the client as "McpError:
    # Connection closed" wrapped in a nested TaskGroup traceback, and the actual
    # reason never surfaces. Forgetting to seed is the single most likely thing
    # to go wrong on a fresh clone, so it gets a real message.
    if not catalogue.DB_PATH.exists():
        sys.exit(
            f"The catalogue database does not exist yet at {catalogue.DB_PATH.name}.\n"
            f"Build it first:  uv run seed_db.py"
        )

    # Built BEFORE the session opens, for the same reason. get_model() exits if
    # OPENAI_API_KEY is missing, and a SystemExit raised inside the session is
    # caught by anyio's task group and re-emitted as a traceback with the actual
    # message buried at the bottom. Out here it prints as the one line it is.
    model = get_model()

    connections = build_connections()
    client = MultiServerMCPClient(connections)

    async with client.session("catalogue") as session:
        tools = await load_mcp_tools(session)
        if verbose:
            print(f"MCP catalogue (stdio):  {', '.join(t.name for t in tools)}")

        if "web" in connections:
            try:
                web_tools = await client.get_tools(server_name="web")
                tools = list(tools) + list(web_tools)
                if verbose:
                    print(f"MCP web (http):         {', '.join(t.name for t in web_tools)}")
            except Exception as problem:
                # A dead remote server must not take the local tools with it.
                if verbose:
                    print(f"MCP web (http):         unavailable, continuing without it ({problem})")
        elif verbose:
            print("MCP web (http):         not configured (no TAVILY_API_KEY)")

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[
                # exit_behavior="end" makes the agent answer with what it has
                # when it hits the ceiling. "error" would raise instead, which
                # loses the work it already did.
                ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
            ],
        )

        yield agent, tools
