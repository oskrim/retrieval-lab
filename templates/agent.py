"""Your CRM agent: the agent model (started by lab serve) calls the tools in TOOLS until it can name the deals that
answer the question; answer returns their ids.

  lab eval agent.py                        exact sets on 10 dev questions
  lab eval agent.py --worst 3              ... and 3 wrong answers: the deals missing and extra
  lab submit agent.py -m "what changed"    runs it on the leaderboard's CRM questions

As it is, the model has plain text search only: it cannot filter by country or stage, sort by amount or count deals,
so it gets few questions exactly right. Climb:
  1. TOOLS = [lab.ATTR_SEARCH], FUNCTIONS = lab.ATTR: search with filters, sorting and counts (lab.search_crm)
  2. add lab.FIND_VALUES and lab.FIND_FIELDS: the exact values and fields behind words like "fintech" or "biggest"
  3. tell the model in SYSTEM when to use which
"""

import lab

SYSTEM = lab.CRM_SYSTEM
TOOLS = [lab.PLAIN_SEARCH, lab.READ_DEAL]
FUNCTIONS = lab.PLAIN  # tool name -> the Python function that runs it


def answer(question: str) -> set[str]:
    return lab.crm_agent(question, TOOLS, FUNCTIONS, system=SYSTEM)["deals"]
