"""Deterministic quote arithmetic for the training catalogue.

This module is the reason the agent is allowed near a price at all. The model
picks WHICH modules to quote; every number in the answer comes from here.

Two rules hold throughout:

  * No function raises into the agent loop. Invalid input comes back as a dict
    with an "error" key, so the model reads the problem as a tool result and can
    fix its own request instead of the run dying.
  * Money is computed in Decimal and rounded to cents once, at the end. Rounding
    each intermediate line would drift, and a quote that does not add up is
    worse than no quote.

The errors here are deliberately actionable. A missing prerequisite comes back
naming the module to add, because the point of a ReAct loop is that the model
can act on what a tool tells it.
"""

from decimal import Decimal, ROUND_HALF_UP

# Above this many people a single session stops working as training and starts
# working as a lecture. The catalogue is priced per group, so the honest answer
# is two groups rather than a bigger discount.
MAX_GROUP_SIZE = 25

CENTS = Decimal("0.01")


def _money(value: Decimal) -> float:
    """Round a Decimal to cents and hand back a float, for JSON."""
    return float(value.quantize(CENTS, rounding=ROUND_HALF_UP))


def discount_rate_for(module_count: int, rates: dict) -> float:
    """The volume discount that applies to this many modules.

    The rate card lists thresholds; the highest one that the count reaches wins.
    Sorting here rather than trusting the file order means the rate card can be
    edited without a silent change of meaning.
    """
    applicable = [
        band["rate"]
        for band in sorted(
            rates["volume_discount"], key=lambda b: b["min_modules"], reverse=True
        )
        if module_count >= band["min_modules"]
    ]
    return applicable[0] if applicable else 0.0


def missing_prerequisites(modules: list[dict], lookup=None) -> list[str]:
    """Every module the package depends on but does not include.

    Resolved TRANSITIVELY when a lookup is given, and that detail is the
    difference between one round trip and four. B-108 requires B-103, which
    requires B-102, which requires B-101. Reporting only the immediate gap
    means the agent adds B-103, is told about B-102, adds it, is told about
    B-101 - three extra model calls to discover something the catalogue knew
    all along, and on a short call budget it runs out before answering.

    Args:
        modules: the rows being quoted.
        lookup: id -> module row, or None to report only the immediate gap.
    """
    selected = {m["id"] for m in modules}
    missing: set[str] = set()

    queue = [p for m in modules for p in m["prerequisites"]]
    while queue:
        needed = queue.pop()
        if needed in selected or needed in missing:
            continue
        missing.add(needed)
        if lookup is not None:
            parent = lookup(needed)
            if parent is not None:
                queue.extend(parent["prerequisites"])

    return sorted(missing)


def quote(
    modules: list[dict],
    people: int,
    delivery: str,
    distance_km: float,
    rates: dict,
    lookup=None,
) -> dict:
    """Price a package of modules for a group.

    Args:
        modules: the full catalogue rows being quoted, in the order requested.
        people: headcount attending.
        delivery: "online" or "onsite".
        distance_km: one-way distance for onsite delivery; billed both ways.
        rates: the rate card, as loaded from data/rates.json.
        lookup: id -> module row, used to resolve the prerequisite chain in one
            pass. Optional so that pricing stays testable without a database.

    Returns a breakdown with every line that makes up the total, or a dict with
    an "error" key that says what to change.
    """
    if not modules:
        return {"error": "No modules to quote. Call find_modules first and pass the ids you want."}

    if people < 1:
        return {"error": "Headcount must be at least 1 person."}

    if people > MAX_GROUP_SIZE:
        return {
            "error": (
                f"{people} people is above the maximum group size of {MAX_GROUP_SIZE}. "
                f"Quote two separate groups instead and say so in the offer."
            )
        }

    if delivery not in ("online", "onsite"):
        return {"error": 'Delivery must be either "online" or "onsite".'}

    # A negative distance would be billed as a negative travel line and quietly
    # discount the whole offer. Rejected rather than clamped to zero, because a
    # negative distance means the caller got something wrong and silently
    # treating it as "next door" hides that.
    if distance_km < 0:
        return {"error": "Distance cannot be negative. Give the one-way distance in km, or 0 for online."}

    # A module marked online-only cannot be delivered in a room. Reported with
    # the ids so the model can either drop them or switch the whole package to
    # online, rather than guessing which.
    online_only = [m["id"] for m in modules if m["delivery"] == "online"]
    if delivery == "onsite" and online_only:
        return {
            "error": (
                f"These modules are online-only and cannot be delivered onsite: "
                f"{', '.join(online_only)}. Either drop them from the package or "
                f"quote the whole package as online."
            )
        }

    # Prerequisites are part of the price, not a footnote. If a package skips a
    # prerequisite the group cannot follow the module, so the quote is wrong
    # rather than merely incomplete.
    missing = missing_prerequisites(modules, lookup)
    if missing:
        return {
            "error": (
                f"Package is missing prerequisites: {', '.join(missing)}. "
                f"That is the complete chain, not just the first level - add all "
                f"of them and quote again, or drop the modules that depend on them."
            ),
            "add_these": missing,
        }

    total_hours = sum(Decimal(str(m["hours"])) for m in modules)

    # The hourly rate is per track, so a mixed package is priced module by
    # module. Summing hours first and multiplying once would quietly charge the
    # beginners rate for an owners module.
    lines = []
    base = Decimal("0")
    for module in modules:
        hours = Decimal(str(module["hours"]))
        rate = Decimal(str(rates["hourly_rate"][module["track"]]))
        line_total = hours * rate
        base += line_total
        lines.append(
            {
                "id": module["id"],
                "name": module["name"],
                "track": module["track"],
                "hours": float(hours),
                "hourly_rate": float(rate),
                "line_total": _money(line_total),
            }
        )

    # Headcount is charged on the whole package, not per module: one extra person
    # sits through every hour of training exactly once.
    extra_people = max(0, people - rates["included_headcount"])
    per_extra = Decimal(str(rates["per_extra_person_per_hour"]))
    headcount_surcharge = Decimal(extra_people) * per_extra * total_hours

    subtotal = base + headcount_surcharge

    rate = Decimal(str(discount_rate_for(len(modules), rates)))
    discount = subtotal * rate

    travel = Decimal("0")
    if delivery == "onsite":
        travel = Decimal(str(distance_km)) * Decimal("2") * Decimal(str(rates["travel_per_km"]))

    net = subtotal - discount + travel
    vat = net * Decimal(str(rates["vat_rate"]))

    return {
        "currency": rates["currency"],
        "modules": lines,
        "module_count": len(modules),
        "total_hours": float(total_hours),
        "people": people,
        "delivery": delivery,
        "base_training": _money(base),
        "headcount_surcharge": _money(headcount_surcharge),
        "extra_people_charged": extra_people,
        "included_headcount": rates["included_headcount"],
        "subtotal": _money(subtotal),
        "discount_rate": float(rate),
        "discount": _money(discount),
        "travel": _money(travel),
        "distance_km": float(distance_km) if delivery == "onsite" else 0.0,
        "net": _money(net),
        "vat_rate": rates["vat_rate"],
        "vat": _money(vat),
        "total_incl_vat": _money(net + vat),
    }
