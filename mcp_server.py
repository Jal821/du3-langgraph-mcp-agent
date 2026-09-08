"""The MCP server that carries this agent's own tools.

Run directly it speaks MCP over stdio and does nothing else:

    uv run mcp_server.py

It is not imported by the agent. `agent.py` launches it as a subprocess and
talks to it over the protocol, which is the whole point of the exercise: the
tools are a separate program with a published interface, not functions closed
over the agent's variables. The same server can be pointed at Claude Desktop,
at n8n, or at a colleague's agent, with no change here.

Two conventions from the course that this file takes seriously:

  * A tool docstring is not documentation, it is PROMPT. It is the only thing
    the model reads when deciding whether to call the tool, so each one says
    what the tool is FOR and when to reach for it, not merely what it does.
  * A tool that raises kills the agent run. A tool that returns its failure as
    data lets the model try something else. Every failure below is returned.
"""

import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

import catalogue
import pricing

# log_level keeps FastMCP from narrating every ListToolsRequest onto stderr,
# which otherwise interleaves with the agent trace and makes it unreadable.
mcp = FastMCP("training-catalogue", log_level="WARNING")

RATES = catalogue.load_rates()

# Opened once, read-only, and kept for the life of the process. Read-only is the
# guard rail: see the module docstring in catalogue.py.
if not catalogue.DB_PATH.exists():
    print(
        f"Catalogue database missing at {catalogue.DB_PATH}.\n"
        f"Build it first:  uv run seed_db.py",
        file=sys.stderr,
    )
    raise SystemExit(2)

DB = catalogue.connect_readonly()

WIKIPEDIA_USER_AGENT = "du3-langgraph-mcp-agent/0.1 (course homework)"


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


@mcp.tool()
def describe_catalogue() -> dict:
    """Find out what this catalogue contains before searching or pricing it.

    Call this FIRST, once, at the start of any request about training, courses,
    modules or prices. It returns the real track names, level names, delivery
    options, module counts and the full rate card, so nothing has to be guessed.

    Takes no arguments. Cheap. If you are about to assume what a track is called
    or what an hour costs, call this instead.
    """
    return catalogue.describe(DB, RATES)


@mcp.tool()
def find_modules(
    query: str = "",
    track: str = "",
    level: str = "",
    max_hours: float = 0.0,
    limit: int = 8,
) -> dict:
    """Search the training catalogue by topic and pick out matching modules.

    This is full-text search over module names, keywords and summaries, so pass
    the client's actual problem in their own words - "invoices and receipts",
    "our team cannot find files", "replying to customers" - rather than a single
    keyword. Terms are OR-ed and results come back best match first.

    Args:
        query: what the client needs, in plain words. Empty returns everything
            that matches the filters, which is how to browse a whole track.
        track: "beginners" for staff learning everyday use, "owners" for setting
            AI up across a team. Empty means both.
        level: "beginner", "intermediate" or "advanced". Empty means any.
        max_hours: only modules no longer than this many hours. 0 means no limit.
        limit: how many modules to return, at most 20.

    Returns COMPLETE module records - id, hours, delivery mode, prerequisites and
    all. There is no need to call get_module afterwards on anything this already
    returned; everything about those modules is in this reply. Use the ids with
    quote(). If nothing matches, the reply says so and suggests widening the
    search rather than returning an error.
    """
    if limit < 1 or limit > 20:
        limit = 8

    if track and track not in catalogue.TRACKS:
        return {
            "error": f'Unknown track "{track}". Valid tracks are: '
            f'{", ".join(catalogue.TRACKS)}. Call describe_catalogue for the vocabulary.'
        }
    if level and level not in catalogue.LEVELS:
        return {
            "error": f'Unknown level "{level}". Valid levels are: '
            f'{", ".join(catalogue.LEVELS)}.'
        }

    results = catalogue.search(
        DB,
        query=query,
        track=track or None,
        level=level or None,
        max_hours=max_hours if max_hours and max_hours > 0 else None,
        limit=limit,
    )

    if not results:
        return {
            "matches": 0,
            "modules": [],
            "hint": (
                "Nothing matched. Try fewer filters, a shorter query, or call "
                "describe_catalogue to see what the catalogue actually covers."
            ),
        }

    return {
        "matches": len(results),
        "modules": results,
        "searched_for": catalogue.to_match_query(query) if query else "(filters only)",
    }


@mcp.tool()
def get_module(module_id: str) -> dict:
    """Look up one module by its exact id, for example "O-203" or "B-102".

    Use this when a module id is already known - most often to pull in a
    prerequisite that quote() has told you is missing from a package. For finding
    modules by topic, use find_modules instead.
    """
    module = catalogue.get(DB, module_id)
    if module is None:
        return {
            "error": f'No module with id "{module_id}". Ids look like B-101 or '
            f"O-203; use find_modules to discover them."
        }
    return module


@mcp.tool()
def quote(
    module_ids: list[str],
    people: int,
    delivery: str = "online",
    distance_km: float = 0.0,
) -> dict:
    """Price a package of modules. ALWAYS use this for any number in an offer.

    Never add up hours or prices yourself and never estimate a total - the rate
    card has a headcount surcharge, volume discount bands, travel and VAT, and
    getting one of them wrong produces an offer that cannot be honoured. This
    tool is the only source of a price.

    Args:
        module_ids: the modules to include, as ids from find_modules.
        people: how many people will attend.
        delivery: "online" or "onsite".
        distance_km: one-way distance to the client for onsite delivery; billed
            both ways. Ignored when delivery is online.

    Returns the full breakdown: a line per module, the headcount surcharge, the
    volume discount, travel, VAT and the total.

    Errors from this tool are RECOVERABLE and tell you what to change. A missing
    prerequisite names the module to add - fetch it with get_module and quote
    again. An online-only module in an onsite package can be dropped or the
    package switched to online. Fix and retry rather than reporting the error to
    the user as a dead end.
    """
    if not module_ids:
        return {"error": "No module ids given. Search the catalogue with find_modules first."}

    modules = []
    unknown = []
    for module_id in module_ids:
        module = catalogue.get(DB, module_id)
        if module is None:
            unknown.append(module_id)
        else:
            modules.append(module)

    if unknown:
        return {
            "error": f'Unknown module ids: {", ".join(unknown)}. '
            f"Use find_modules or get_module to get real ids."
        }

    # Duplicates would be charged twice and silently inflate the discount band.
    seen = set()
    deduplicated = []
    for module in modules:
        if module["id"] not in seen:
            seen.add(module["id"])
            deduplicated.append(module)

    result = pricing.quote(
        modules=deduplicated,
        people=people,
        delivery=delivery,
        distance_km=distance_km,
        rates=RATES,
        # The resolver lets pricing walk the whole prerequisite chain in one go,
        # so a missing dependency costs the agent one retry rather than one per
        # level of the chain.
        lookup=lambda module_id: catalogue.get(DB, module_id),
    )
    if len(deduplicated) < len(modules) and "error" not in result:
        result["note"] = "Duplicate ids were requested and have been counted once."
    return result


# ---------------------------------------------------------------------------
# The outside world
# ---------------------------------------------------------------------------


@mcp.tool()
def wikipedia_lookup(term: str, lang: str = "en") -> dict:
    """Look up background on an industry, trade or concept on Wikipedia.

    Use this to understand a client's line of business before recommending
    modules - what a freight forwarder or a notary actually does all day, which
    tells you which of their tasks are document-heavy or correspondence-heavy.
    Also the right tool for a stable definition of a term.

    It is NOT for current events, prices, news or anything time-sensitive; use
    web_search for those. It searches Wikipedia and returns the intro of the best
    matching article, so an approximate term is fine.

    Args:
        term: what to look up, for example "freight forwarder" or "bookkeeping".
        lang: Wikipedia language edition, "en" or "sk". Defaults to English,
            which has far longer articles on business topics.
    """
    if not term.strip():
        return {"error": "Nothing to look up - term was empty."}
    if lang not in ("en", "sk", "cs", "de"):
        return {"error": f'Unsupported lang "{lang}". Use "en", "sk", "cs" or "de".'}

    parameters = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": term,
        "gsrlimit": "1",
        "prop": "extracts|info",
        "exintro": "1",
        "explaintext": "1",
        "inprop": "url",
        "redirects": "1",
    }

    try:
        response = httpx.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params=parameters,
            headers={"User-Agent": WIKIPEDIA_USER_AGENT},
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as problem:
        # Returned, not raised: a flaky network should cost the agent one tool
        # call, not the whole run.
        return {"error": f"Wikipedia request failed: {problem}"}

    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return {
            "found": False,
            "term": term,
            "hint": f'Wikipedia has no article matching "{term}". Try a broader term.',
        }

    page = pages[0]
    extract = (page.get("extract") or "").strip()
    # Long intros crowd the context window without adding much; the first few
    # hundred words are what actually informs a recommendation.
    if len(extract) > 1500:
        extract = extract[:1500].rsplit(" ", 1)[0] + " ..."

    return {
        "found": True,
        "title": page.get("title"),
        "url": page.get("fullurl"),
        "extract": extract,
        "lang": lang,
    }


if __name__ == "__main__":
    # stdio: the agent starts this file as a subprocess and speaks MCP down the
    # pipe. Nothing may be printed to stdout here - that channel is the protocol.
    mcp.run(transport="stdio")
