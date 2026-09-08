"""Checks on the tools without calling the model.

    uv run test_tools.py

The price is deliberately NOT verified with the formula that computes it, since
that would only repeat any mistake. `pricing.quote` works package-wide: it sums
the module lines, then charges the extra headcount once against the package's
total hours. The oracle here instead writes out an invoice line by line - one
line per module, and one line per extra person per module - and adds it up.

The two are algebraically the same and structurally different, which is the
point. Charging the headcount per module instead of per package, or applying the
included-headcount allowance once per module instead of once per package, moves
one side and not the other. That is exactly the class of bug that produces a
plausible wrong number in front of a client.

Same principle as homework 1, where the mortgage payment was checked by
simulating the loan month by month rather than by rerunning the annuity formula.

Nothing here touches the network or the LLM.
"""

from decimal import Decimal, ROUND_HALF_UP

import catalogue
import pricing

RATES = catalogue.load_rates()
MODULES = {m["id"]: m for m in catalogue.load_modules()}

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((bool(ok), label))
    print(f"{'OK ' if ok else 'BAD'} {label}{('  ->  ' + detail) if detail else ''}")


def package(*ids: str) -> list[dict]:
    """The catalogue rows for these ids, straight from the JSON."""
    return [MODULES[i] for i in ids]


# ---------------------------------------------------------------------------
# The independent oracle
# ---------------------------------------------------------------------------


def invoice_total(ids: list[str], people: int, delivery: str, distance_km: float = 0.0) -> dict:
    """Price the package by writing out every line, then adding them up.

    This is the check, not the implementation. It never calls pricing.py.
    """
    lines: list[Decimal] = []

    # One line per module, at its own track's rate.
    for module_id in ids:
        module = MODULES[module_id]
        rate = Decimal(str(RATES["hourly_rate"][module["track"]]))
        lines.append(Decimal(str(module["hours"])) * rate)

    # One line per extra person PER MODULE. pricing.py charges this once against
    # the package total instead; if the two ever disagree, one of them is wrong
    # about where the allowance applies.
    extra_people = max(0, people - RATES["included_headcount"])
    per_extra = Decimal(str(RATES["per_extra_person_per_hour"]))
    for _person in range(extra_people):
        for module_id in ids:
            lines.append(Decimal(str(MODULES[module_id]["hours"])) * per_extra)

    subtotal = sum(lines, Decimal("0"))

    # The discount band, worked out by counting rather than by sorting the card.
    band = Decimal("0")
    for rule in RATES["volume_discount"]:
        if len(ids) >= rule["min_modules"]:
            band = max(band, Decimal(str(rule["rate"])))

    travel = Decimal("0")
    if delivery == "onsite":
        travel = Decimal(str(distance_km)) * 2 * Decimal(str(RATES["travel_per_km"]))

    net = subtotal - (subtotal * band) + travel
    vat = net * Decimal(str(RATES["vat_rate"]))

    cents = lambda v: float(v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return {
        "subtotal": cents(subtotal),
        "net": cents(net),
        "vat": cents(vat),
        "total_incl_vat": cents(net + vat),
    }


# ---------------------------------------------------------------------------
# 1. the price, two ways
# ---------------------------------------------------------------------------

CASES = [
    (["B-101", "B-102"], 4, "online", 0.0),
    (["B-101", "B-102", "B-103"], 6, "online", 0.0),
    (["B-101", "B-102", "B-103"], 7, "online", 0.0),
    (["B-101", "B-102", "B-103", "B-104", "B-105"], 12, "online", 0.0),
    (["B-101", "B-102", "B-103", "B-104", "B-105", "B-106", "B-107", "B-109"], 20, "online", 0.0),
    (["O-201", "O-202", "O-203"], 3, "onsite", 85.0),
    (["B-101", "O-201"], 9, "onsite", 12.5),
    (["B-101", "B-102", "B-105", "O-201", "O-206"], 25, "onsite", 200.0),
]

for ids, people, delivery, km in CASES:
    produced = pricing.quote(package(*ids), people, delivery, km, RATES)
    expected = invoice_total(ids, people, delivery, km)
    label = f"{len(ids)} modules, {people} people, {delivery}"
    if "error" in produced:
        check(False, label, f"quote refused it: {produced['error']}")
        continue
    agrees = all(produced[key] == expected[key] for key in expected)
    check(
        agrees,
        label,
        f"{produced['total_incl_vat']} EUR"
        if agrees
        else f"quote {produced['total_incl_vat']} vs invoice {expected['total_incl_vat']}",
    )

# ---------------------------------------------------------------------------
# 2. the discount bands, at their edges
# ---------------------------------------------------------------------------

# Off-by-one on a threshold is invisible in the middle of a band and wrong at
# the edge, so the edges are what get tested.
for count, expected_rate in [(1, 0.0), (4, 0.0), (5, 0.10), (7, 0.10), (8, 0.15), (9, 0.15)]:
    actual = pricing.discount_rate_for(count, RATES)
    check(actual == expected_rate, f"{count} modules -> {expected_rate:.0%} discount", f"{actual:.0%}")

# ---------------------------------------------------------------------------
# 3. a mixed package is priced per track, not blended
# ---------------------------------------------------------------------------

# Both are entry modules with no prerequisites, so the only thing under test
# here is the rate, not the package rules.
mixed = pricing.quote(package("B-101", "O-201"), 2, "online", 0.0, RATES)
# B-101 is 2 h on the beginners rate, O-201 is 2 h on the owners rate. A single
# blended rate over 4 hours would give a different, wrong, number.
by_hand = 2.0 * RATES["hourly_rate"]["beginners"] + 2.0 * RATES["hourly_rate"]["owners"]
blended = 4.0 * RATES["hourly_rate"]["beginners"]
check(
    "error" not in mixed and mixed["base_training"] == round(by_hand, 2),
    "a mixed-track package charges each module at its own rate",
    f"{mixed.get('base_training')} EUR, not the blended {round(blended, 2)}",
)

# ---------------------------------------------------------------------------
# 4. the headcount allowance applies once per package
# ---------------------------------------------------------------------------

# Six people are included, so the seventh person costs per_extra_person_per_hour
# for every hour in the package - once, not once per module.
six = pricing.quote(package("B-101", "B-102"), 6, "online", 0.0, RATES)
seven = pricing.quote(package("B-101", "B-102"), 7, "online", 0.0, RATES)
step = round(seven["subtotal"] - six["subtotal"], 2)
expected_step = round(RATES["per_extra_person_per_hour"] * (2.0 + 2.5), 2)
check(
    step == expected_step and six["headcount_surcharge"] == 0.0,
    "the 7th person costs one extra-person rate over the package hours",
    f"+{step} EUR for 4.5 h",
)

# ---------------------------------------------------------------------------
# 5. errors are returned, actionable, and never raised
# ---------------------------------------------------------------------------

# B-102 requires B-101. A package with the dependant but not the prerequisite is
# not merely incomplete, it is unteachable, so it must be refused.
missing = pricing.quote(package("B-102"), 5, "online", 0.0, RATES)
check(
    "error" in missing and "B-101" in missing["error"],
    "a missing prerequisite is refused and named",
    missing.get("error", "no error returned"),
)

# The chain matters more than the first link. B-108 needs B-103, which needs
# B-102, which needs B-101. Without the transitive walk the agent is told about
# one level per call and spends four model calls rediscovering the catalogue -
# which is exactly what happened on the first real run of this project.
lookup = MODULES.get
chain = pricing.quote(package("B-108"), 4, "online", 0.0, RATES, lookup=lookup)
check(
    "error" in chain and chain.get("add_these") == ["B-101", "B-102", "B-103"],
    "the whole prerequisite chain is reported at once, not one level per call",
    ", ".join(chain.get("add_these", [])) or chain.get("error", ""),
)

# Without a lookup it can only see one level, and must not pretend otherwise.
one_level = pricing.quote(package("B-108"), 4, "online", 0.0, RATES)
check(
    one_level.get("add_these") == ["B-103"],
    "with no lookup it reports only the immediate gap",
    ", ".join(one_level.get("add_these", [])),
)

# And once the chain is supplied, the package prices without complaint.
complete = pricing.quote(
    package("B-101", "B-102", "B-103", "B-108"), 4, "online", 0.0, RATES, lookup=lookup
)
check(
    "error" not in complete,
    "the completed chain prices cleanly",
    f"{complete.get('total_incl_vat')} EUR",
)

# B-108 is online-only. Quoting it onsite has to fail, and has to say which
# module caused it, or the model cannot fix its own request.
onsite_conflict = pricing.quote(
    package("B-101", "B-102", "B-103", "B-108"), 4, "onsite", 10.0, RATES
)
check(
    "error" in onsite_conflict and "B-108" in onsite_conflict["error"],
    "an online-only module is refused onsite and named",
    onsite_conflict.get("error", "no error returned"),
)

# The same package online is fine. Without this the check above would pass on a
# rule that simply rejects everything.
same_online = pricing.quote(package("B-101", "B-102", "B-103", "B-108"), 4, "online", 0.0, RATES)
check(
    "error" not in same_online,
    "the same package is accepted online",
    f"{same_online.get('total_incl_vat')} EUR",
)

# A negative distance used to be billed as a NEGATIVE travel line, quietly
# discounting the whole offer by 0.90 EUR per km "travelled".
negative = pricing.quote(package("O-201"), 4, "onsite", -100.0, RATES)
check(
    "error" in negative and "negative" in negative["error"].lower(),
    "a negative distance is refused, not credited against the total",
    negative.get("error", f"travel={negative.get('travel')}"),
)

too_many = pricing.quote(package("B-101"), pricing.MAX_GROUP_SIZE + 1, "online", 0.0, RATES)
check(
    "error" in too_many and "two separate groups" in too_many["error"],
    "an oversized group is refused with the alternative",
    too_many.get("error", "no error returned"),
)

for bad in (
    pricing.quote([], 5, "online", 0.0, RATES),
    pricing.quote(package("B-101"), 0, "online", 0.0, RATES),
    pricing.quote(package("B-101"), 5, "hologram", 0.0, RATES),
):
    check("error" in bad, "invalid input returns an error and does not raise", bad.get("error", "")[:60])

# ---------------------------------------------------------------------------
# 6. travel is billed both ways
# ---------------------------------------------------------------------------

near = pricing.quote(package("O-201"), 3, "onsite", 0.0, RATES)
far = pricing.quote(package("O-201"), 3, "onsite", 100.0, RATES)
check(
    round(far["travel"] - near["travel"], 2) == round(100.0 * 2 * RATES["travel_per_km"], 2),
    "100 km onsite is billed as a 200 km round trip",
    f"{far['travel']} EUR",
)

online_ignores_distance = pricing.quote(package("O-201"), 3, "online", 500.0, RATES)
check(
    online_ignores_distance["travel"] == 0.0,
    "an online package is not charged travel however far away it is",
)

# ---------------------------------------------------------------------------
# 7. VAT and the total actually add up
# ---------------------------------------------------------------------------

whole = pricing.quote(
    package("B-101", "B-102", "B-103", "B-104", "B-105"), 14, "online", 0.0, RATES
)
adds_up = (
    round(whole["subtotal"] - whole["discount"] + whole["travel"], 2) == whole["net"]
    and round(whole["net"] + whole["vat"], 2) == whole["total_incl_vat"]
)
check(adds_up, "the breakdown adds up to the total it states", f"{whole['total_incl_vat']} EUR")

# ---------------------------------------------------------------------------
# 8. full-text search finds the module a client would be asking for
# ---------------------------------------------------------------------------

connection = catalogue.connect_readonly()

SEARCHES = [
    ("we drown in invoices and receipts every month", "O-206"),
    ("nobody on the team can find the right file", "O-203"),
    ("we spend all day replying to customer emails", "B-103"),
    ("can it write our contracts summaries", "B-104"),
    ("worried about GDPR and personal data", "B-109"),
    ("we want a chatbot on our own documents", "O-208"),
]
for query, expected_id in SEARCHES:
    found = catalogue.search(connection, query=query, limit=3)
    ids = [m["id"] for m in found]
    check(expected_id in ids, f'"{query[:38]}..." finds {expected_id}', ", ".join(ids) or "nothing")

# The stemmer over-shot on "es": "files" became "fil" and "invoices" became
# "invoic", both too short to search as a prefix, so they fell back to an exact
# match and stopped finding the singular. A plural and its singular have to
# reach the same prefix.
for plural, singular in [("files", "file"), ("invoices", "invoice"), ("notes", "note")]:
    check(
        catalogue.stem(plural) == catalogue.stem(singular) == singular,
        f'"{plural}" and "{singular}" stem alike',
        f"{catalogue.stem(plural)} / {catalogue.stem(singular)}",
    )

# And the words where the "e" really is part of the plural ending, or part of
# the stem, must not be mangled either.
for word, expected_stem in [
    ("classes", "class"),
    ("dishes", "dish"),
    ("business", "business"),
    ("process", "process"),
    ("caches", "cache"),
]:
    check(
        catalogue.stem(word) == expected_stem,
        f'"{word}" stems to "{expected_stem}"',
        catalogue.stem(word),
    )

# Searching for either spelling has to find the module, in both directions.
for query in ("file", "files", "invoice", "invoices"):
    ids = [m["id"] for m in catalogue.search(connection, query=query, limit=5)]
    check(
        bool(ids),
        f'"{query}" matches something either way round',
        ", ".join(ids) or "nothing",
    )

# A sentence full of FTS5 syntax must be treated as words, not as a query
# expression. If the sanitiser leaks, this raises instead of returning rows.
try:
    hostile = catalogue.search(
        connection, query='invoices" OR 1=1 -- AND NEAR(x, y) *: "', limit=3
    )
    check(True, "FTS5 syntax in the query is neutralised, not executed", f"{len(hostile)} rows")
except Exception as problem:
    check(False, "FTS5 syntax in the query is neutralised, not executed", str(problem))

# An empty query is browsing, not an error.
browse = catalogue.search(connection, query="", track="owners", limit=20)
check(
    len(browse) == 10 and all(m["track"] == "owners" for m in browse),
    "an empty query with a filter browses that track",
    f"{len(browse)} owners modules",
)

# ---------------------------------------------------------------------------
# 9. the price-provenance check catches an invented total
# ---------------------------------------------------------------------------

# Importing main pulls in the agent module but starts nothing; no MCP server is
# launched and no model is contacted until build_agent() is entered.
import main  # noqa: E402  (imported here so the checks above run without it)

# These are the real numbers from the run that exposed the problem: the agent
# quoted the onsite package correctly, then closed with an "or all online"
# paragraph in which 544 EUR and 125.12 EUR appeared from nowhere.
KNOWN = {540.0, 72.0, 225.0, 837.0, 192.51, 1029.51, 250.0, 0.45, 60.0, 4.0, 8.0, 9.0, 23.0}
REAL_ANSWER = (
    "Celkom vrátane DPH 1 029,51 €. Cestovné 225,00 €. "
    "Alternatíva: celý balík online by bol 1 029,51 € "
    "(training 544 € + DPH 125,12 € + 0 cestovné) za 10,5 hodiny. 23 %."
)
check(
    main.unsupported_prices(REAL_ANSWER, KNOWN) == [125.12, 544.0],
    "an invented total is caught and a quoted one is not",
    ", ".join(str(v) for v in main.unsupported_prices(REAL_ANSWER, KNOWN)),
)

check(
    main.unsupported_prices("Total 1029.51 EUR, travel 225.00 EUR, 12.5 hours", KNOWN) == [],
    "an answer built only from quoted figures raises nothing",
)

# Hours, percentages and distances are restated and recombined legitimately, so
# a bare number must not be treated as a price.
check(
    main.unsupported_prices("The package runs 12.5 hours over 4 sessions, 23% VAT", KNOWN) == [],
    "numbers without a currency marker are not treated as prices",
)

# The same amount is written three ways depending on the language of the answer.
check(
    main.as_amount("1 029,51") == main.as_amount("1,029.51") == main.as_amount("1.029,51") == 1029.51,
    "an amount reads the same in Slovak, English and German punctuation",
)

# "0,450" was read as 450: the three-digits-after-the-comma rule fired on what
# is unambiguously a decimal, because no thousands group ever starts with a
# lone zero. It mattered because it is how the travel rate gets written.
check(
    main.as_amount("0,450") == 0.45 and main.as_amount("1,450") == 1450.0,
    "a leading zero marks a decimal comma, not a thousands separator",
    f'0,450 -> {main.as_amount("0,450")}, 1,450 -> {main.as_amount("1,450")}',
)

# Tool results used to be harvested with the prose regex, which pulled the "206"
# out of the module id "B-206" and added it to the set of figures a price may
# match. Over a run that admitted every id from 101 to 210, leaving a
# hundred-euro band where an invented total passed as verified.
import json  # noqa: E402

ID_HEAVY = json.dumps(
    {"modules": [{"id": "B-206", "hours": 2.5}, {"id": "O-201", "hours": 2.0}]}
)
from_payload = main.numbers_in_payload(ID_HEAVY)
check(
    206.0 not in from_payload and 201.0 not in from_payload and 2.5 in from_payload,
    "module ids do not enter the known set as numbers",
    f"harvested {sorted(from_payload)}",
)
check(
    main.unsupported_prices("Total: 206,00 EUR.", from_payload) == [206.0],
    "a price that merely matches a module id is still flagged",
    ", ".join(str(v) for v in main.unsupported_prices("Total: 206,00 EUR.", from_payload)),
)

# MCP hands a tool reply back as [{"type": "text", "text": "<json>"}], so the
# real numbers are a level of JSON-in-string down. Miss that and every genuine
# price gets flagged instead.
ENVELOPE = json.dumps([{"type": "text", "text": json.dumps({"total_incl_vat": 885.6, "id": "B-206"})}])
through = main.numbers_in_payload(ENVELOPE)
check(
    885.6 in through and 206.0 not in through,
    "a value is found through the MCP envelope and the id in it is not",
    f"harvested {sorted(through)}",
)

# The harvester runs on whatever a tool returned, including a tool that failed,
# so it must never be the thing that raises.
for junk in ("", "not json at all", "{broken", None, 42, ["a", {"b": 3.5}]):
    try:
        main.numbers_in_payload(junk)
        ok = True
    except Exception as problem:  # noqa: BLE001
        ok = False
    check(ok, f"a payload of {type(junk).__name__} does not raise")

# A real breakdown has to pass in full, or the warning becomes noise nobody reads.
GENUINE = main.numbers_in_payload(
    json.dumps({"base_training": 750.0, "headcount_surcharge": 300.0, "discount": 105.0,
                "vat": 217.35, "total_incl_vat": 1162.35})
)
check(
    main.unsupported_prices(
        "Base 750,00 €, surcharge 300,00 €, discount -105,00 €, VAT 217,35 €, total 1 162,35 €.",
        GENUINE,
    ) == [],
    "a genuine breakdown passes without a single false flag",
)

# The check must not treat the agent's own earlier answer as evidence. `known`
# holds tool and user numbers only; folding replies back in meant an invented
# price became established fact on the following turn.
carried = {100.0}
first_turn = main.unsupported_prices("The total is 999,99 EUR.", carried)
second_turn = main.unsupported_prices("As I said, 999,99 EUR.", carried)
check(
    first_turn == [999.99] and second_turn == [999.99],
    "an invented price is not laundered into the next turn",
    f"turn 1 {first_turn}, turn 2 {second_turn}",
)

# ---------------------------------------------------------------------------
# 10. the guard rail is proved, not asserted
# ---------------------------------------------------------------------------

# This is the connection the MCP server holds. If a write gets through here, the
# agent's read-only guarantee is decoration.
try:
    connection.execute("DELETE FROM modules WHERE id = 'B-101'")
    connection.commit()
    check(False, "the catalogue connection refuses writes", "the DELETE SUCCEEDED")
except Exception as problem:
    check("readonly" in str(problem).lower(), "the catalogue connection refuses writes", str(problem))

# Rebuilding while this reader is open is exactly what happens when seed_db.py
# is run with an agent still going in another terminal. Windows refuses to
# unlink the file, and a raw WinError 32 says nothing about the cause.
try:
    catalogue.build()
    check(False, "a locked rebuild is refused with an explanation", "the rebuild SUCCEEDED")
except RuntimeError as problem:
    check(
        "another terminal" in str(problem),
        "a locked rebuild is refused with an explanation",
        str(problem),
    )
except Exception as problem:  # noqa: BLE001
    check(False, "a locked rebuild is refused with an explanation",
          f"raised {type(problem).__name__} instead: {problem}")

connection.close()

# ---------------------------------------------------------------------------

failed = [label for ok, label in results if not ok]
print()
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    for label in failed:
        print(f"  failed: {label}")
    raise SystemExit(1)
