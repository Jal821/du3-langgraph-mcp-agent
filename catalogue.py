"""The module catalogue: a SQLite file with a full-text index over it.

SQLite rather than Postgres because homework 2 already proved the Postgres path
and this machine has no Docker. Nothing here needs a server: the catalogue is
twenty rows. What it does need is real full-text search, which is why the table
is mirrored into an FTS5 index instead of being matched with LIKE.

The important design point is the connection. The MCP server opens this file
with `mode=ro`, so the agent physically cannot write to it. That is the same
guard rail as the read-only database role in homework 2, moved down a layer:
a prompt injection that talks the model into deleting a module gets a refusal
from SQLite, not from a regex over the model's output.
"""

import json
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
DB_PATH = HERE / "catalogue.sqlite"

TRACKS = ("beginners", "owners")
LEVELS = ("beginner", "intermediate", "advanced")
DELIVERIES = ("online", "onsite", "both")

# Words that carry no signal in a catalogue this small and only dilute the bm25
# score. "AI" is in here because every single module is about AI.
STOPWORDS = {
    "a", "about", "ai", "an", "and", "any", "are", "as", "at", "be", "by", "can",
    "course", "do", "does", "for", "from", "has", "have", "how", "i", "in", "is",
    "it", "me", "my", "need", "of", "on", "or", "our", "small", "so", "some",
    "team", "that", "the", "their", "them", "they", "this", "to", "training",
    "us", "want", "we", "what", "who", "will", "with", "would", "you", "your",
}


def load_modules() -> list[dict]:
    """The catalogue as it sits in data/, untouched by SQLite."""
    return json.loads((DATA / "modules.json").read_text(encoding="utf-8"))


def load_rates() -> dict:
    """The rate card as it sits in data/."""
    return json.loads((DATA / "rates.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Building the database
# ---------------------------------------------------------------------------


SCHEMA = """
CREATE TABLE modules (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    track         TEXT NOT NULL CHECK (track IN ('beginners', 'owners')),
    level         TEXT NOT NULL CHECK (level IN ('beginner', 'intermediate', 'advanced')),
    hours         REAL NOT NULL CHECK (hours > 0),
    delivery      TEXT NOT NULL CHECK (delivery IN ('online', 'onsite', 'both')),
    prerequisites TEXT NOT NULL,
    keywords      TEXT NOT NULL,
    summary       TEXT NOT NULL
);

-- Searchable text only. The FTS table holds no authoritative data, so it can be
-- rebuilt from `modules` at any time without losing anything.
CREATE VIRTUAL TABLE modules_fts USING fts5(
    id UNINDEXED,
    name,
    keywords,
    summary
);
"""


def build(path: Path = DB_PATH) -> dict:
    """Create the catalogue database from data/modules.json.

    Destructive by design: the file is rebuilt from the JSON every time, because
    the JSON is the source of truth and a half-migrated database is worse than
    no database.
    """
    modules = load_modules()

    if path.exists():
        try:
            path.unlink()
        except PermissionError as problem:
            # Windows refuses to unlink a file another process has open, and the
            # process holding it is almost always an MCP server started by a
            # running agent. The raw WinError 32 says nothing about that, so it
            # is translated into the thing to actually do.
            raise RuntimeError(
                f"Cannot rebuild {path.name}: another process has it open. "
                f"That is usually an agent still running in another terminal - "
                f"stop it and run this again."
            ) from problem

    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            """
            INSERT INTO modules
                (id, name, track, level, hours, delivery, prerequisites, keywords, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    m["id"],
                    m["name"],
                    m["track"],
                    m["level"],
                    m["hours"],
                    m["delivery"],
                    # Stored as a comma-joined string. A junction table would be
                    # correct for a real catalogue; for twenty rows it would only
                    # add a join the model has to be told about.
                    ",".join(m["prerequisites"]),
                    m["keywords"],
                    m["summary"],
                )
                for m in modules
            ],
        )
        connection.executemany(
            "INSERT INTO modules_fts (id, name, keywords, summary) VALUES (?, ?, ?, ?)",
            [(m["id"], m["name"], m["keywords"], m["summary"]) for m in modules],
        )
        connection.commit()
    finally:
        connection.close()

    return {"modules": len(modules), "path": str(path)}


# ---------------------------------------------------------------------------
# Reading it
# ---------------------------------------------------------------------------


def connect_readonly(path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the catalogue so that writes are impossible.

    `mode=ro` is enforced by SQLite itself, below anything the model can reach.
    The URI needs forward slashes even on Windows, hence as_posix().
    """
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _row_to_module(row: sqlite3.Row) -> dict:
    """One database row in the shape the rest of the code expects."""
    return {
        "id": row["id"],
        "name": row["name"],
        "track": row["track"],
        "level": row["level"],
        "hours": row["hours"],
        "delivery": row["delivery"],
        "prerequisites": [p for p in row["prerequisites"].split(",") if p],
        "keywords": row["keywords"],
        "summary": row["summary"],
    }


def stem(term: str) -> str:
    """Cut an English word back to something a prefix search can match on.

    FTS5's default tokeniser matches whole tokens, with no stemming. That is
    fine until a client writes "we cannot find the right file" and the module
    about files is indexed under "files" - the query then matches nothing at
    all. Both of the search failures this project started with were exactly
    that: file/files, and replying/replies.

    So each term is cut back to a stem and searched as a prefix. This is not a
    real stemmer and does not try to be; it handles the plural and the participle,
    which is what a sentence about business problems actually contains. The cost
    is some over-matching, which bm25 ranking absorbs - a wrong module ranked
    fourth is invisible, a right module missing entirely is not.
    """
    if len(term) > 5 and term.endswith(("ing", "ies")):
        term = term[:-3]
    # "classes", "dishes": the base word ends in a sibilant and takes "es" as
    # the whole plural ending, so both letters go. Deliberately just these two
    # patterns - adding "ches" or "zes" starts mangling words like "caches" and
    # "sizes", where the "e" belongs to the stem.
    elif len(term) > 5 and term.endswith(("sses", "shes")):
        term = term[:-2]
    # Everywhere else the "e" is the stem's own and only the plural "s" goes.
    # Dropping both over-stemmed: "files" became "fil" and "invoices" became
    # "invoic", which are then too short to search as a prefix, so they fell
    # back to an exact match and stopped finding the singular at all.
    elif len(term) > 4 and term.endswith("es"):
        term = term[:-1]
    elif len(term) > 4 and term.endswith("ed"):
        term = term[:-2]
    elif len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        term = term[:-1]
    # reply / replies / replying all have to land on the same stem, so a
    # trailing y goes too. This is the step that makes the participle work.
    if len(term) > 4 and term.endswith("y"):
        term = term[:-1]
    return term


def to_match_query(text: str) -> str:
    """Turn a plain-language question into an FTS5 MATCH expression.

    FTS5 has its own query syntax, so passing a user sentence straight through
    is both a syntax error waiting to happen ("invoices, receipts" breaks on the
    comma) and an injection surface. Every term is therefore extracted with a
    regex, double-quoted as a literal, and joined with OR - the model gets to
    influence the ranking, never the grammar.
    """
    terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", text)
        if len(term) > 1 and term.lower() not in STOPWORDS
    ]

    expressions = []
    for term in terms:
        root = stem(term)
        # Only search as a prefix when the stem is long enough to mean
        # something. "not" as a prefix would match half the catalogue.
        expressions.append(f'"{root}"*' if len(root) >= 4 else f'"{term}"')

    # Deduplicated but order kept, so a repeated word does not skew bm25.
    return " OR ".join(dict.fromkeys(expressions))


def search(
    connection: sqlite3.Connection,
    query: str = "",
    track: str | None = None,
    level: str | None = None,
    max_hours: float | None = None,
    limit: int = 8,
) -> list[dict]:
    """Full-text search over the catalogue, with structured filters on top.

    An empty query is a valid request - it means "everything matching the
    filters" - and is answered without touching the FTS index at all.
    """
    where = []
    parameters: list = []

    if track:
        where.append("m.track = ?")
        parameters.append(track)
    if level:
        where.append("m.level = ?")
        parameters.append(level)
    if max_hours is not None:
        where.append("m.hours <= ?")
        parameters.append(max_hours)

    match = to_match_query(query) if query else ""

    if match:
        sql = """
            SELECT m.*, bm25(modules_fts) AS score
            FROM modules_fts
            JOIN modules m ON m.id = modules_fts.id
            WHERE modules_fts MATCH ?
        """
        parameters = [match] + parameters
        if where:
            sql += " AND " + " AND ".join(where)
        # bm25 returns a negative number and the strongest match is the most
        # negative, so ascending is best-first.
        sql += " ORDER BY score ASC, m.id ASC LIMIT ?"
    else:
        sql = "SELECT m.*, NULL AS score FROM modules m"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY m.id ASC LIMIT ?"

    parameters.append(limit)
    rows = connection.execute(sql, parameters).fetchall()

    results = []
    for row in rows:
        module = _row_to_module(row)
        if row["score"] is not None:
            module["match_score"] = round(-row["score"], 3)
        results.append(module)
    return results


def get(connection: sqlite3.Connection, module_id: str) -> dict | None:
    """One module by its exact id, or None."""
    row = connection.execute(
        "SELECT * FROM modules WHERE id = ?", (module_id.strip().upper(),)
    ).fetchone()
    return _row_to_module(row) if row else None


def describe(connection: sqlite3.Connection, rates: dict) -> dict:
    """What is in the catalogue and what the numbers mean.

    This tool exists because of the failure mode from lesson 5: a model asked
    about a database it cannot see will invent column names, track names and
    prices. Giving it one cheap call that returns the real vocabulary is more
    reliable than a longer system prompt telling it not to guess.
    """
    counts = connection.execute(
        "SELECT track, level, COUNT(*) AS n, SUM(hours) AS hours"
        " FROM modules GROUP BY track, level ORDER BY track, level"
    ).fetchall()
    total = connection.execute(
        "SELECT COUNT(*) AS n, SUM(hours) AS hours FROM modules"
    ).fetchone()

    return {
        "modules_total": total["n"],
        "hours_total": round(total["hours"], 2),
        "tracks": {
            "beginners": "staff learning practical everyday AI use",
            "owners": "owners and managers setting AI up for a whole team",
        },
        "levels": list(LEVELS),
        "delivery_options": ["online", "onsite"],
        "breakdown": [
            {"track": r["track"], "level": r["level"], "modules": r["n"], "hours": round(r["hours"], 2)}
            for r in counts
        ],
        "rate_card": rates,
        "id_format": "B-1xx for the beginners track, O-2xx for the owners track",
        "how_to_price": (
            "Never compute a price yourself. Call quote() with the module ids, the "
            "headcount and the delivery mode; it returns every line of the breakdown."
        ),
    }
