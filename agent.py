"""Your CRM agent. The model calls the tools in TOOLS, plain Python functions it sees by name, docstring and type
hints, until it can name the deals that answer the question. Sharpen them, add your own, or tell it more in SYSTEM."""

import lab

SYSTEM = lab.CRM_SYSTEM  # today's date, and the "DEALS: D0001, ..." line the answer must end with


def search(
    query: str = "",
    # filters: dict | None = None,
    # sort_by: str | None = None,
    # descending: bool = True,
    # limit: int = 10,
) -> dict:
    """Search the CRM's deals by words. Returns how many deals it found and the best 10."""
    # filters: {field: value | [values] | {"min": x, "max": y}}; sort_by: a number or date field; limit: at most 50.
    # lab.DEALS holds 600 deals like {"deal_id": "D0001", "company": "Elm Motors", "stage": "Closed won", ...}
    deals = lab.DEALS
    # deals = [d for d in deals if all(lab.matches(d, f, w) for f, w in (filters or {}).items())]
    if query:
        deals = lab.search_deals(query, deals)
    # if sort_by:
    #     deals = sorted(deals, key=lambda d: d[sort_by], reverse=descending)
    # return dict(total=len(deals), deals=deals[:min(limit, 50)])
    return dict(total=len(deals), deals=deals[:10])


def read(deal_id: str) -> dict:
    """Read one deal in full."""
    # lab.DEAL maps a deal id to its deal; deals also have industry, country, owner, employees, close_date, notes, ...
    return lab.DEAL.get(deal_id) or {"error": f"no deal {deal_id}"}


def values(field: str) -> list[str]:
    """Every value of a category field (industry, country, stage, owner, product or lead_source)."""
    # lab.find_values("fintech") finds the values behind a word instead: [{"field": "industry", "value": ...}, ...]
    return sorted({d[field] for d in lab.DEALS})


# More in lab: matches(deal, field, want), find_values(text), find_fields(text), STAGES, CRM_TODAY, chat, tool_of.
TOOLS = [
    search,
    # read,
    # values,
]


def answer(question: str) -> set[str]:
    return lab.agent(question, TOOLS, SYSTEM)
