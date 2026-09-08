"""Build catalogue.sqlite from data/, then check the load rather than trust it.

    uv run seed_db.py

The checks matter more than they look. The agent's answers are only as good as
the catalogue, and a silently half-loaded table produces confident wrong offers
instead of an error. So this script asserts that what came out of SQLite matches
what went into it, and that the prerequisite graph is actually usable.
"""

import sys

import catalogue


def main() -> int:
    modules = catalogue.load_modules()
    print(f"data/modules.json holds {len(modules)} modules")

    try:
        result = catalogue.build()
    except RuntimeError as problem:
        # The database is locked by something else. One sentence and a non-zero
        # exit, not a traceback into pathlib.
        print(problem, file=sys.stderr)
        return 2
    print(f"built {result['path']}")

    failures = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"{'OK ' if ok else 'BAD'} {label}{('  ->  ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    connection = catalogue.connect_readonly()

    # -- the load itself ----------------------------------------------------
    rows = connection.execute("SELECT COUNT(*) AS n FROM modules").fetchone()["n"]
    check(rows == len(modules), "every module reached the modules table", f"{rows} rows")

    indexed = connection.execute("SELECT COUNT(*) AS n FROM modules_fts").fetchone()["n"]
    check(
        indexed == len(modules),
        "every module reached the full-text index",
        f"{indexed} rows",
    )

    hours_json = round(sum(m["hours"] for m in modules), 2)
    hours_db = round(connection.execute("SELECT SUM(hours) AS h FROM modules").fetchone()["h"], 2)
    check(hours_json == hours_db, "total hours survived the load", f"{hours_db} h")

    ids_json = {m["id"] for m in modules}
    ids_db = {r["id"] for r in connection.execute("SELECT id FROM modules")}
    check(ids_json == ids_db, "the same ids came back out", f"{len(ids_db)} ids")

    # -- the prerequisite graph --------------------------------------------
    # A prerequisite pointing at a module that does not exist would make quote()
    # unsatisfiable: it would demand a module nothing can supply.
    dangling = sorted(
        {p for m in modules for p in m["prerequisites"] if p not in ids_json}
    )
    check(not dangling, "no prerequisite points at a missing module", ", ".join(dangling))

    # A cycle would be worse: quote() would ask for A because of B and B because
    # of A, and the agent would loop until the call limit stopped it.
    prerequisites = {m["id"]: m["prerequisites"] for m in modules}
    state: dict[str, int] = {}
    cycles: list[str] = []

    def walk(node: str, path: list[str]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cycles.append(" -> ".join(path + [node]))
            return
        state[node] = 1
        for parent in prerequisites.get(node, []):
            walk(parent, path + [node])
        state[node] = 2

    for module_id in prerequisites:
        walk(module_id, [])
    check(not cycles, "the prerequisite graph has no cycles", "; ".join(cycles))

    # -- the guard rail is real, not asserted ------------------------------
    # The MCP server opens exactly this connection. If a write succeeds here,
    # the agent's read-only guarantee is fiction.
    try:
        connection.execute("UPDATE modules SET hours = 99 WHERE id = 'B-101'")
        connection.commit()
        check(False, "the read-only connection refuses writes", "the UPDATE SUCCEEDED")
    except Exception as problem:
        check(
            "readonly" in str(problem).lower(),
            "the read-only connection refuses writes",
            str(problem),
        )

    connection.close()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("Catalogue built and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
