"""Your CRM agent: the agent model (started by lab serve) calls the tools in TOOLS until it can name the deals that
answer the question; answer returns their ids.

  lab eval agent                           exact sets on 10 dev questions, saved as a run
  lab eval agent --worst 3                 ... and 3 wrong answers: the deals missing and extra
  lab compare agent                        what changed since the run before (lab compare question N: one question)
  lab trace crm 3                          question 3 step by step: each tool call, what it returned, the answer
  lab query agent "your question"          ask it anything
  lab submit agent                         runs it on the leaderboard's CRM questions

A tool is a Python function: the model sees its name, its docstring and its parameters (type hints: str, int, float,
bool, list[...], dict, or X | None with a default for an optional one), and gets back what it returns (as JSON).
As it is, the model can only search the deals by words and read them: it cannot filter by country or stage, sort by
amount or count deals, so it gets few questions exactly right. Write better tools.

What you can use (lab.py):
  lab.DEALS                       the 600 deals (a list of dicts, below); lab.DEAL[deal_id]: one deal
  lab.search_deals(query, deals)  the deals (default all) that best match query by words, best first (the notebook's
                                  hybrid; n=20 of them)
  lab.matches(deal, field, want)  whether deal[field] is want: a value, a list of values (any of them) or
                                  {"min": x, "max": y} (both ends included); text ignores case
  lab.find_values(text, field)    the exact category values behind words: "fintech" -> {"field": "industry",
                                  "value": "Financial services"}; field (optional) keeps one field's
  lab.find_fields(text)           the number and date fields behind words: "biggest" -> amount_eur
  lab.STAGES                      the stages in pipeline order, each with its win probability
  lab.CRM_TODAY                   today in the CRM's world (YYYY-MM-DD)
  lab.CRM_SYSTEM                  the system prompt: today's date and the "DEALS: D0001, ..." line the answer ends with
  lab.agent(question, tools, system)  the loop: up to 5 turns of tool calls, then the answer's deal ids
  lab.chat(messages, tools)       one reply of the agent model, to write your own loop (lab.tool_of(fn) makes a
                                  function's tool schema)

A deal:
  {"deal_id": "D0001", "company": "Elm Motors", "industry": "Construction", "country": "Denmark",
   "stage": "Closed won", "amount_eur": 56400.0, "annual_revenue_eur": 631880000.0, "employees": 5525,
   "probability": 1.0, "close_date": "2026-06-12", "last_activity_date": "2026-05-30", "owner": "Lukas Weber",
   "product": "Support Plus", "lead_source": "Webinar", "notes": "Elm Motors wants to consolidate reporting ..."}
Categories: industry, country, stage, owner, product, lead_source. Numbers: amount_eur, annual_revenue_eur, employees,
probability. Dates (YYYY-MM-DD, so they compare and sort as text): close_date, last_activity_date.
"""

import lab

SYSTEM = lab.CRM_SYSTEM


def search(query: str) -> list[dict]:
    """Search the CRM's deals by words. Returns the 5 best, each with its company, stage and amount."""
    return [{f: d[f] for f in ("deal_id", "company", "stage", "amount_eur")} for d in lab.search_deals(query)[:5]]


def read(deal_id: str) -> dict:
    """Read one deal in full."""
    return lab.DEAL.get(deal_id) or {"error": f"no deal {deal_id}"}


def values(field: str) -> list[str]:
    """Every value of a category field (industry, country, stage, owner, product or lead_source)."""
    return sorted({d[field] for d in lab.DEALS})


TOOLS = [search, read, values]


def answer(question: str) -> set[str]:
    return lab.agent(question, TOOLS, SYSTEM)
