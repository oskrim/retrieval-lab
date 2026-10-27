"""Your CRM agent. The model calls the tools in TOOLS, plain Python functions it sees by name, docstring and type
hints, until it can name the deals that answer the question. These tools only search by words: write better ones."""

import lab

SYSTEM = lab.CRM_SYSTEM  # today's date, and the "DEALS: D0001, ..." line the answer must end with


def search(query: str) -> list[dict]:
    """Search the CRM's deals by words. Returns the 5 best, each with its company, stage and amount."""
    # lab.DEALS holds 600 deals like {"deal_id": "D0001", "company": "Elm Motors", "stage": "Closed won", ...}
    return [{f: d[f] for f in ("deal_id", "company", "stage", "amount_eur")} for d in lab.search_deals(query)[:5]]


def read(deal_id: str) -> dict:
    """Read one deal in full."""
    return lab.DEAL.get(deal_id) or {"error": f"no deal {deal_id}"}


def values(field: str) -> list[str]:
    """Every value of a category field (industry, country, stage, owner, product or lead_source)."""
    return sorted({d[field] for d in lab.DEALS})


# More in lab: matches(deal, field, want), find_values(text), find_fields(text), STAGES, CRM_TODAY, chat, tool_of.
TOOLS = [search, read, values]


def answer(question: str) -> set[str]:
    return lab.agent(question, TOOLS, SYSTEM)
