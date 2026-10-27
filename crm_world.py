"""The notebook's bonus on tool search: the CRM of the section on filters, sorting and counts as a small world that an
agent changes with real tools, and tasks graded on the world's end state.

A World holds its own copy of the 600 deals plus the state the actions write: notes, follow-up tasks, sent emails and
calendar events. Its tools are the CRM search tools (search_deals, find_values, find_fields), CRM actions, and the
workplace catalog of lab.tool_catalog(), whose tools outside the CRM do nothing and answer {"ok": true}."""

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

import lab

TODAY = "Tuesday 2026-10-27"  # lab.CRM_TODAY with its weekday, so that "Thursday" and "next Monday" are one date each
_FIRST = ("Tom", "Laura", "Pekka", "Julia", "Daniel", "Elena", "Marco", "Nina", "Oliver", "Petra", "Ravi", "Sophie",
          "Viktor", "Hanna", "Ingrid", "Karl", "Lea", "Felix", "Greta", "Jon")
_LAST = ("Andersen", "Becker", "Costa", "Dahlberg", "Eriksen", "Fischer", "Garcia", "Hansen", "Ito", "Jensen", "Keller",
         "Lehtonen", "Moreau", "Novak", "Olsen", "Peters", "Quinn", "Ramos", "Silva", "Tanaka", "Ueda", "Varga",
         "Weiss", "Young", "Zimmer", "Berg", "Kowalski", "Murphy", "Nieminen", "Ortiz")
_ROLES = ("CFO", "Head of IT", "Procurement manager", "COO", "Head of sales", "CTO")


def _contacts() -> list[dict]:
    """One contact per deal's company (every company has one deal); the same every time."""
    rng = np.random.default_rng(1)
    out = []
    for d in lab.DEALS:
        first, last = str(rng.choice(_FIRST)), str(rng.choice(_LAST))
        domain = re.sub(r"[^a-z]", "", d["company"].lower()) + ".example"
        out.append(dict(contact_id="C" + d["deal_id"][1:], name=f"{first} {last}", role=str(rng.choice(_ROLES)),
                        email=f"{first}.{last}@{domain}".lower(), phone=f"+358 40 {rng.integers(100, 999)} {rng.integers(1000, 9999)}",
                        company=d["company"], deal_id=d["deal_id"]))
    return out


CONTACTS = _contacts()
CONTACT_OF = {c["deal_id"]: c for c in CONTACTS}
DEAL_OF = {c["email"]: c["deal_id"] for c in CONTACTS}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


class World:
    """The CRM for one task: a copy of the deals, so that writes do not leak into other tasks, and empty logs."""

    def __init__(self) -> None:
        self.deals = copy.deepcopy(lab.DEALS)
        self.row = {d["deal_id"]: i for i, d in enumerate(self.deals)}
        self.notes: list[dict] = []
        self.tasks: list[dict] = []
        self.outbox: list[dict] = []
        self.calendar: list[dict] = []

    def deal(self, deal_id) -> dict:
        i = self.row.get(str(deal_id).strip().upper())
        if i is None:
            raise ValueError(f"no deal {deal_id!r}; deal ids look like D0042")
        return self.deals[i]

    def call(self, name: str, args: dict) -> str:
        """Runs tool `name` with args; the result as text for the model, an error included."""
        if name not in TOOLS:
            return json.dumps(dict(error=f"no tool named {name}"))
        if not isinstance(args, dict):
            return json.dumps(dict(error="the arguments must be a JSON object"))
        fn = ACTIONS.get(name)
        if fn is None:
            return json.dumps(dict(ok=True))  # a workplace tool outside the CRM: it does nothing here
        try:
            out = fn(self, **args)
        except (TypeError, ValueError, KeyError) as e:
            out = dict(error=str(e))
        return json.dumps(out, ensure_ascii=False)


def _one_of(value, allowed, what: str) -> str:
    """allowed's entry equal to value, ignoring case; ValueError naming the allowed values otherwise."""
    for a in allowed:
        if str(value).strip().lower() == a.lower():
            return a
    raise ValueError(f"unknown {what} {value!r}; the {what}s are: {', '.join(allowed)}")


def _date(value, what: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value).strip()):
        raise ValueError(f"{what} must be a date as YYYY-MM-DD, not {value!r}")
    return str(value).strip()


def _time(value, what: str) -> str:
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[T ](\d{1,2}):(\d{2})(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?", str(value).strip())
    if not m:
        raise ValueError(f"{what} must be a date and time as YYYY-MM-DD HH:MM, not {value!r}")
    return f"{m[1]} {int(m[2]):02d}:{m[3]}"


def _emails(value) -> list[str]:
    found = EMAIL.findall(" ".join(value) if isinstance(value, list) else str(value))
    if not found:
        raise ValueError(f"give email addresses, not {value!r}")
    return [e.lower() for e in found]


# ---- the CRM search tools of the section on filters, sorting and counts, on this world's deals

def search_deals(w: World, query="", filters=None, sort_by=None, descending=True, fields=None, limit=10, group_by=None):
    """lab.search_crm on w's deals."""
    query, filters = query or "", filters or {}
    if not isinstance(filters, dict):
        raise ValueError('filters must be an object: {field: value | [values] | {"min": x, "max": y}}')
    descending, limit = descending is not False, 10 if limit is None else limit
    unknown = [f for f in [*filters, sort_by, group_by, *(fields or [])] if f is not None and f not in w.deals[0]]
    if unknown:
        return dict(error=f"unknown fields {unknown}; the fields are {list(w.deals[0])}")
    mask = np.array([all(lab.matches(d, f, v) for f, v in filters.items()) for d in w.deals])
    rows = list(lab.hybrid(lab.crm_index("crm"), query, n=20, mask=mask)) if query.strip() else list(np.flatnonzero(mask))
    if sort_by:
        rows.sort(key=lambda i: lab._key(sort_by, w.deals[i][sort_by]), reverse=descending)
    keep = ["deal_id", *(fields or ["company", "industry", "country", "stage", "amount_eur", "owner"])]
    out = dict(total=int(mask.sum()), deals=[{f: w.deals[i][f] for f in keep} for i in rows[:min(int(limit), 50)]])
    if group_by:
        out["counts"] = dict(Counter(w.deals[i][group_by] for i in np.flatnonzero(mask)).most_common())
    return out


def find_values(w: World, text, field=None, k=10):
    return lab.find_values(text, field, k)


def find_fields(w: World, text, k=3):
    return lab.find_fields(text, k)


# ---- CRM actions

def get_deal(w: World, deal_id):
    d = w.deal(deal_id)
    return dict(d, contact=CONTACT_OF[d["deal_id"]], crm_notes=[n["text"] for n in w.notes if n["deal_id"] == d["deal_id"]],
                follow_up_tasks=[t for t in w.tasks if t["deal_id"] == d["deal_id"]])


def update_deal_stage(w: World, deal_id, stage):
    d = w.deal(deal_id)
    d["stage"] = _one_of(stage, lab.STAGES, "stage")
    d["probability"] = lab.STAGES[d["stage"]]
    return dict(ok=True, deal_id=d["deal_id"], stage=d["stage"])


def set_deal_owner(w: World, deal_id, owner):
    d = w.deal(deal_id)
    d["owner"] = _one_of(owner, lab.OWNERS, "owner")
    return dict(ok=True, deal_id=d["deal_id"], owner=d["owner"])


def add_deal_note(w: World, deal_id, text):
    d = w.deal(deal_id)
    if not str(text).strip():
        raise ValueError("the note is empty")
    w.notes.append(dict(deal_id=d["deal_id"], text=str(text)))
    return dict(ok=True, deal_id=d["deal_id"], notes=sum(n["deal_id"] == d["deal_id"] for n in w.notes))


def create_follow_up_task(w: World, deal_id, title, due_date):
    d = w.deal(deal_id)
    w.tasks.append(dict(deal_id=d["deal_id"], title=str(title), due_date=_date(due_date, "due_date")))
    return dict(ok=True, deal_id=d["deal_id"], due_date=w.tasks[-1]["due_date"])


def search_contacts(w: World, query):
    words = lab.tokenize(str(query))
    if not words:
        raise ValueError("the query is empty")
    hits = [c for c in CONTACTS if all(x in lab.tokenize(f"{c['name']} {c['email']} {c['company']}") for x in words)]
    return dict(total=len(hits), contacts=hits[:10])


def get_account(w: World, account_id):
    """account_id: the company's name."""
    name = str(account_id).strip().lower()
    deals = [d for d in w.deals if d["company"].lower() == name]
    if not deals:
        raise ValueError(f"no account {account_id!r}; the account id is the company's name, e.g. 'Zenith Group'")
    ids = {d["deal_id"] for d in deals}
    return dict(company=deals[0]["company"], deals=deals, contacts=[CONTACT_OF[i] for i in sorted(ids)],
                crm_notes=[n for n in w.notes if n["deal_id"] in ids], follow_up_tasks=[t for t in w.tasks if t["deal_id"] in ids])


def send_email(w: World, to, subject, body):
    w.outbox.append(dict(to=_emails(to), subject=str(subject), body=str(body)))
    return dict(ok=True, sent_to=w.outbox[-1]["to"])


def create_calendar_event(w: World, title, start, end, attendees):
    w.calendar.append(dict(title=str(title), start=_time(start, "start"), end=_time(end, "end"), attendees=_emails(attendees)))
    return dict(ok=True, start=w.calendar[-1]["start"], attendees=w.calendar[-1]["attendees"])


ACTIONS = dict(search_deals=search_deals, find_values=find_values, find_fields=find_fields, crm_get_deal=get_deal,
               crm_update_deal_stage=update_deal_stage, crm_set_deal_owner=set_deal_owner, crm_add_deal_note=add_deal_note,
               crm_create_follow_up_task=create_follow_up_task, crm_search_contacts=search_contacts,
               crm_get_account=get_account, send_email=send_email, create_calendar_event=create_calendar_event)
CRM_TOOLS = [  # the tools the workplace catalog lacks; it has crm_update_deal_stage, crm_search_contacts and the rest
    dict(lab.ATTR_SEARCH, function=dict(lab.ATTR_SEARCH["function"], name="search_deals")), lab.FIND_VALUES, lab.FIND_FIELDS,
    lab.tool("crm_get_deal", "One sales deal in full: its fields, its contact person, CRM notes and follow-up tasks.",
             ["deal_id"], deal_id=dict(type="string")),
    lab.tool("crm_set_deal_owner", "Hand a sales deal over to another salesperson.", ["deal_id", "owner"],
             deal_id=dict(type="string"), owner=dict(type="string", description="The salesperson's full name.")),
    lab.tool("crm_add_deal_note", "Add a note to a sales deal's history.", ["deal_id", "text"],
             deal_id=dict(type="string"), text=dict(type="string")),
    lab.tool("crm_create_follow_up_task", "Create a follow-up task on a sales deal, due on a date.",
             ["deal_id", "title", "due_date"], deal_id=dict(type="string"), title=dict(type="string"),
             due_date=dict(type="string", description="YYYY-MM-DD")),
]
CATALOG = lab.tool_catalog() + CRM_TOOLS
TOOLS = {t["function"]["name"]: t for t in CATALOG}
assert set(ACTIONS) <= set(TOOLS) and len(TOOLS) == len(CATALOG)


def tool_documents() -> tuple[list[str], list[str], list[str]]:
    """(ids, titles, bodies) of CATALOG, written as lab.documents("tools") writes the workplace catalog."""
    fns = [t["function"] for t in CATALOG]
    return ([f["name"] for f in fns], [f["name"].replace("_", " ") for f in fns],
            [f"{f['description']} Parameters: {', '.join(f['parameters']['properties'])}." for f in fns])


# ---- tasks and their grading

def effects(w: World) -> dict[tuple[str, str], list[str]]:
    """What w's tools changed, as (kind, deal id) -> values: stage, owner, note, task (its due date), email (subject and
    body) and event (its start). An email or event goes to the deal of its contact, or to the address if no contact has it."""
    out: dict[tuple[str, str], list[str]] = {}
    for d0, d in zip(lab.DEALS, w.deals):
        for kind in ("stage", "owner"):
            if d[kind] != d0[kind]:
                out.setdefault((kind, d["deal_id"]), []).append(d[kind])
    for n in w.notes:
        out.setdefault(("note", n["deal_id"]), []).append(n["text"])
    for t in w.tasks:
        out.setdefault(("task", t["deal_id"]), []).append(t["due_date"])
    for m in w.outbox:
        for to in m["to"]:
            out.setdefault(("email", DEAL_OF.get(to, to)), []).append(f"{m['subject']}\n{m['body']}")
    for e in w.calendar:
        for to in e["attendees"]:
            out.setdefault(("event", DEAL_OF.get(to, to)), []).append(e["start"])
    return out


@dataclass(frozen=True)
class Task:
    text: str
    solution: tuple  # (tool, args) calls that do the task
    words: tuple = ()  # what a note or email must mention, each a regular expression (any case)


def _ids(**search) -> list[str]:
    return [d["deal_id"] for d in search_deals(World(), **{"limit": 50, **search})["deals"]]


def _email(deal_id: str) -> str:
    return CONTACT_OF[deal_id]["email"]


def _tasks() -> list[Task]:
    nordic_fintech = _ids(filters=dict(industry="Financial services", country=lab.NORDICS, stage=lab.OPEN))
    biggest_de = _ids(filters=dict(country="Germany", stage="Proposal"), sort_by="amount_eur", limit=1)[0]
    mikko_at = _ids(filters=dict(owner="Mikko Laine", country="Austria", stage=lab.OPEN))
    quiet = _ids(filters=dict(owner="Maria Virtanen", stage=lab.OPEN), sort_by="last_activity_date", descending=False, limit=3)
    champion = [d["deal_id"] for d in lab.DEALS if d["country"] == "Germany" and d["stage"] in lab.OPEN
                and "champion left" in d["notes"]]
    tomasz_is = _ids(filters=dict(owner="Tomasz Nowak", country="Iceland", stage=lab.OPEN))
    biggest_fi = _ids(filters=dict(country="Finland", stage=lab.OPEN), sort_by="amount_eur", limit=1)[0]
    zenith, fjord, aurora = (_ids(filters=dict(company=c))[0] for c in ("Zenith Group", "Fjord Systems", "Aurora Labs"))
    big_open = _ids(filters=dict(stage=lab.OPEN, amount_eur=dict(min=150000)))
    return [
        Task("Move Zenith Group's deal to negotiation.",
             (("crm_update_deal_stage", dict(deal_id=zenith, stage="Negotiation")),)),
        Task("Log a note on deal D0250 that legal approved the terms.",
             (("crm_add_deal_note", dict(deal_id="D0250", text="Legal approved the terms.")),), ("legal", "approv")),
        Task("Every open fintech deal in the Nordics: add a follow-up task for next Monday.",
             tuple(("crm_create_follow_up_task", dict(deal_id=d, title="Follow up", due_date="2026-11-02")) for d in nordic_fintech)),
        Task("Email the contact at the biggest German deal in proposal stage and say we can meet Thursday.",
             (("send_email", dict(to=_email(biggest_de), subject="Meeting", body="We can meet on Thursday.")),), ("thursday",)),
        Task("Hand Mikko's open deals in Austria over to Sara Lindqvist.",
             tuple(("crm_set_deal_owner", dict(deal_id=d, owner="Sara Lindqvist")) for d in mikko_at)),
        Task("We lost the Fjord Systems deal to Vantage: close it as lost and note why.",
             (("crm_update_deal_stage", dict(deal_id=fjord, stage="Closed lost")),
              ("crm_add_deal_note", dict(deal_id=fjord, text="Lost to Vantage.")),), ("vantage",)),
        Task("Book a 30-minute call with our contact at Aurora Labs on Thursday at 10.",
             (("create_calendar_event", dict(title="Call", start="2026-10-29 10:00", end="2026-10-29 10:30",
                                             attendees=_email(aurora))),)),
        Task("Maria's three open deals that have gone quiet the longest: add a follow-up task for tomorrow to each.",
             tuple(("crm_create_follow_up_task", dict(deal_id=d, title="Follow up", due_date="2026-10-28")) for d in quiet)),
        Task("Open deals in Germany where our champion left: note on each that we need a new champion.",
             tuple(("crm_add_deal_note", dict(deal_id=d, text="We need a new champion.")) for d in champion), ("champion",)),
        Task("Tell the contact at deal D0182 by email that our proposal is ready, and move the deal to negotiation.",
             (("send_email", dict(to=_email("D0182"), subject="Proposal", body="Our proposal is ready.")),
              ("crm_update_deal_stage", dict(deal_id="D0182", stage="Negotiation"))), ("proposal",)),
        Task("Give every open deal Tomasz has in Iceland to Henrik.",
             tuple(("crm_set_deal_owner", dict(deal_id=d, owner="Henrik Dahl")) for d in tomasz_is)),
        Task("Set up a one-hour meeting with the contact of our largest open deal in Finland on Wednesday next week at 14.",
             (("create_calendar_event", dict(title="Meeting", start="2026-11-04 14:00", end="2026-11-04 15:00",
                                             attendees=_email(biggest_fi))),)),
        Task("Every open deal worth at least €150,000: note that the CFO must sign off.",
             tuple(("crm_add_deal_note", dict(deal_id=d, text="The CFO must sign off.")) for d in big_open), ("cfo",)),
    ]


def solve(task: Task) -> World:
    """A world after task's solution."""
    w = World()
    for name, args in task.solution:
        out = json.loads(w.call(name, args))
        assert "error" not in out, out
    return w


TASKS = _tasks()
_WANT = {t.text: effects(solve(t)) for t in TASKS}


def grade(task: Task, w: World) -> str:
    """completed: w's changes are exactly the solution's, notes and emails mentioning task.words; wrong deals: a deal or
    person outside the task was changed; nothing changed; else incomplete: something the task needs is missing or wrong."""
    want, got = _WANT[task.text], effects(w)
    if {key for _, key in got} - {key for _, key in want}:
        return "wrong deals"
    if not got:
        return "nothing changed"

    def done(kind, key, values):
        have = got.get((kind, key), [])
        if kind in ("note", "email"):
            return any(all(re.search(x, v, re.I) for x in task.words) for v in have)
        return values[-1] in have if kind in ("stage", "owner") else any(v in have for v in values)

    return "completed" if all(done(kind, key, values) for (kind, key), values in want.items()) else "incomplete"
