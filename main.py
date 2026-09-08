"""Homework 3 - a ReAct agent on LangGraph whose tools all arrive over MCP.

    uv run main.py                       interactive, keeps the conversation
    uv run main.py "your question"       answer once and exit
    uv run main.py --tools               list the tools loaded from MCP and exit
    uv run main.py --graph               write graph.png and print the graph

The trace is printed as it happens, so what the model decided to call and what
came back is visible rather than inferred. --quiet turns it off.
"""

import argparse
import asyncio
import json
import re
import sys

from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

import model as model_module
from agent import MAX_MODEL_CALLS, RECURSION_LIMIT, build_agent
from visualizer import mermaid, visualize

load_dotenv()

# The Windows console runs in cp1252 and would die with a UnicodeEncodeError on
# the first Slovak answer, so switch stdout to UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Printing the trace
# ---------------------------------------------------------------------------


def shorten(value, limit: int = 220) -> str:
    """Tool results are long. Show the shape and the head of the content."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


def report_calls(step: int, message) -> None:
    """What the model asked for on this step."""
    for call in message.tool_calls:
        arguments = {k: v for k, v in (call.get("args") or {}).items() if v not in ("", 0, [], None)}
        print(f"[step {step}] TOOL CALL: {call['name']}")
        print(f"         arguments: {shorten(arguments, 300)}")


def report_result(message) -> None:
    """What came back. A returned error is normal here, not a crash.

    The tool name is repeated because the model calls tools in parallel: one
    step can issue three calls, and the results then arrive as a batch. Without
    the name there is no way to pair a result with the call that asked for it.
    """
    name = getattr(message, "name", "?")
    marker = "ERROR" if '"error"' in str(message.content) else "result"
    print(f"         {marker} <- {name}: {shorten(message.content)}")


# ---------------------------------------------------------------------------
# Checking the prices the model states
# ---------------------------------------------------------------------------

# A number written next to a currency marker: "1 029,51 €", "EUR 540.00", "540€".
MONEY = re.compile(
    r"(?:€|EUR)\s*([0-9][0-9\s .,]*)|([0-9][0-9\s .,]*)\s*(?:€|EUR)",
    re.IGNORECASE,
)
NUMBER = re.compile(r"[0-9][0-9\s .,]*[0-9]|[0-9]")


def as_amount(raw: str) -> float | None:
    """Read a number written in any of the conventions that turn up here.

    The model answers in the user's language, so the same amount arrives as
    "1029.51", "1 029,51" or "1.029,51" depending on the sentence it is in.
    """
    text = raw.strip().replace(" ", "").replace(" ", "")
    if not text:
        return None
    # Whichever separator comes last is the decimal one.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # A comma with exactly three digits after it is a thousands separator.
        head, _, tail = text.rpartition(",")
        text = f"{head}{tail}" if len(tail) == 3 and head else text.replace(",", ".")
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def numbers_in(text: str) -> set[float]:
    """Every number in a piece of text, however it is punctuated."""
    found = set()
    for match in NUMBER.finditer(text or ""):
        value = as_amount(match.group(0))
        if value is not None:
            found.add(value)
    return found


def unsupported_prices(final: str, known: set[float]) -> list[float]:
    """Money in the answer that no tool result and no question ever mentioned.

    This is the "the model proposes, code decides" check. The system prompt
    forbids inventing a price, the read-only database stops the agent changing
    the catalogue - but nothing structural stops the model from simply TYPING a
    plausible total, and on the first Slovak run of this project it did exactly
    that in a closing "or you could also..." paragraph.

    So every figure in the answer that carries a currency marker is looked up
    against the numbers the tools actually returned. Anything that is not there
    was made up. Only currency-marked figures are checked, because hours,
    percentages and distances are legitimately restated and recombined.
    """
    stated = set()
    for match in MONEY.finditer(final or ""):
        value = as_amount(match.group(1) or match.group(2) or "")
        if value is not None:
            stated.add(value)
    # Rounding is allowed: a model saying 1029 EUR for 1029.51 is presenting,
    # not inventing.
    return sorted(
        value
        for value in stated
        if not any(abs(value - reference) <= 1.0 for reference in known)
    )


async def answer(agent, history: list, quiet: bool) -> str:
    """Run one turn through the agent, printing the trace as it streams.

    The full history goes in every time. create_agent keeps no memory between
    invocations - each call is a fresh state - so a follow-up question like "and
    onsite?" only works because the previous turns are passed back in.
    """
    step = 0
    final = ""

    # Everything the tools returned, plus everything the user themselves said.
    # A price in the answer has to trace back to one of these.
    known: set[float] = set()
    for message in history:
        content = message if isinstance(message, str) else (
            message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        )
        known |= numbers_in(str(content))

    try:
        async for chunk in agent.astream(
            {"messages": history},
            stream_mode="updates",
            config={"recursion_limit": RECURSION_LIMIT},
        ):
            for node, update in chunk.items():
                for message in update.get("messages", []) if isinstance(update, dict) else []:
                    kind = message.__class__.__name__

                    if kind == "AIMessage" and getattr(message, "tool_calls", None):
                        step += 1
                        if not quiet:
                            report_calls(step, message)
                    elif kind == "ToolMessage":
                        known |= numbers_in(str(message.content))
                        if not quiet:
                            report_result(message)
                    elif kind == "AIMessage" and message.content:
                        final = message.content
                        history.append(message)
    except GraphRecursionError:
        # Should be unreachable: RECURSION_LIMIT is set above what the call-limit
        # middleware allows. If it does fire, say so in one sentence rather than
        # printing a nested TaskGroup traceback from inside the MCP transport.
        return (
            f"The agent hit LangGraph's recursion limit of {RECURSION_LIMIT} steps "
            f"before the {MAX_MODEL_CALLS}-call limit could stop it, which means it "
            f"was looping. Try a narrower question."
        )

    if not final:
        return (
            f"The agent stopped without an answer. It may have hit the "
            f"{MAX_MODEL_CALLS}-call limit; try a narrower question."
        )

    invented = unsupported_prices(final, known)
    if invented:
        # Appended to the answer rather than raised. The offer is still mostly
        # sound and a human is reading it; what they need is to be told which
        # figure not to trust.
        amounts = ", ".join(f"{value:,.2f}" for value in invented)
        final += (
            f"\n\n---\nPRICE CHECK: {amounts} did not come from the quote tool. "
            f"Treat those figures as unverified - ask for them to be quoted properly."
        )

    return final


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


async def run_once(question: str, quiet: bool) -> None:
    async with build_agent() as (agent, _tools):
        print(f"\nQUESTION:\n  {question}\n")
        history = [{"role": "user", "content": question}]
        reply = await answer(agent, history, quiet)
        print("ANSWER:")
        print(reply)


async def run_interactive(quiet: bool) -> None:
    async with build_agent() as (agent, _tools):
        print(
            "\nI build AI training offers. Tell me about the client: what they do,\n"
            "how many people, what they struggle with, and any budget.\n"
            "Follow-up questions keep the context. Ctrl+C or 'quit' to leave.\n"
        )
        history: list = []
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q", "koniec"):
                print("Goodbye.")
                return

            history.append({"role": "user", "content": question})
            print()
            try:
                reply = await answer(agent, history, quiet)
            except Exception as problem:
                # One failed question ends that question, not the session.
                print(f"That question failed: {problem}\n")
                continue
            print(f"Agent: {reply}\n")


async def show_tools() -> None:
    """Every tool the agent can reach, and which MCP server it came from."""
    async with build_agent() as (_agent, tools):
        print(f"\n{len(tools)} tools loaded over MCP:\n")
        for tool in tools:
            first_line = (tool.description or "").strip().split("\n")[0]
            print(f"  {tool.name}")
            print(f"      {first_line}")
        print()


async def show_graph() -> None:
    """Write graph.png and print the mermaid source it was drawn from."""
    async with build_agent() as (agent, _tools):
        wrote = visualize(agent, "graph.png")
        print("\nwrote graph.png" if wrote else "\ncould not write graph.png (needs network)")
        print("\n" + mermaid(agent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A ReAct agent whose tools all come from MCP servers."
    )
    parser.add_argument("question", nargs="?", help="ask once and exit")
    parser.add_argument("--tools", action="store_true", help="list the MCP tools and exit")
    parser.add_argument("--graph", action="store_true", help="write graph.png and exit")
    parser.add_argument("--quiet", action="store_true", help="hide the tool-call trace")
    args = parser.parse_args()

    if args.tools:
        asyncio.run(show_tools())
        return 0
    if args.graph:
        asyncio.run(show_graph())
        return 0

    print(model_module.describe())

    if args.question:
        asyncio.run(run_once(args.question, args.quiet))
    else:
        asyncio.run(run_interactive(args.quiet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
