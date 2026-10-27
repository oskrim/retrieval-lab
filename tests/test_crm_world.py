"""pytest -p no:xdist tests"""

import json

import pytest

import crm_world as cw
import lab

VALID = {  # arguments for every tool that acts on the world; the others take any strings
    "search_deals": dict(filters={"country": "Finland"}, sort_by="amount_eur", limit=3, group_by="stage"),
    "find_values": dict(text="fintech"),
    "find_fields": dict(text="biggest"),
    "crm_get_deal": dict(deal_id="D0001"),
    "crm_update_deal_stage": dict(deal_id="D0001", stage="negotiation"),
    "crm_set_deal_owner": dict(deal_id="d0001", owner="maria virtanen"),
    "crm_add_deal_note": dict(deal_id="D0001", text="Called them."),
    "crm_create_follow_up_task": dict(deal_id="D0001", title="Call back", due_date="2026-11-02"),
    "crm_search_contacts": dict(query="Elm Motors"),
    "crm_get_account": dict(account_id="Elm Motors"),
    "send_email": dict(to="Tom <tom@example.com>", subject="Hi", body="Thursday works."),
    "create_calendar_event": dict(title="Call", start="2026-10-29T10:00", end="2026-10-29 10:30", attendees=["a@b.example"]),
}


@pytest.fixture(autouse=True)
def no_index(monkeypatch):
    """find_values and find_fields are lab's; here they answer without building the value and field indexes."""
    monkeypatch.setattr(lab, "find_values", lambda text, field=None, k=10: [dict(field="industry", value=text)])
    monkeypatch.setattr(lab, "find_fields", lambda text, k=3: [dict(field="amount_eur", about=text)])


def test_every_tool_answers_valid_arguments_without_an_error():
    w = cw.World()
    for name, schema in cw.TOOLS.items():
        args = VALID.get(name) or {p: "x" for p in schema["function"]["parameters"]["properties"]}
        out = json.loads(w.call(name, args))
        assert "error" not in out, (name, out)
    assert w.deal("D0001")["stage"] == "Negotiation" and w.deal("D0001")["probability"] == lab.STAGES["Negotiation"]
    assert w.deal("D0001")["owner"] == "Maria Virtanen"
    assert w.outbox[0]["to"] == ["tom@example.com"] and w.calendar[0]["start"] == "2026-10-29 10:00"


@pytest.mark.parametrize("name, args, says", [
    ("crm_update_deal_stage", dict(deal_id="D0001", stage="won"), "unknown stage"),
    ("crm_set_deal_owner", dict(deal_id="D9999", owner="Maria Virtanen"), "no deal"),
    ("crm_create_follow_up_task", dict(deal_id="D0001", title="x", due_date="next Monday"), "YYYY-MM-DD"),
    ("send_email", dict(to="Tom", subject="x", body="y"), "email addresses"),
    ("crm_add_deal_note", dict(deal_id="D0001"), "text"),
    ("search_deals", dict(filters={"colour": "red"}), "unknown fields"),
    ("no_such_tool", {}, "no tool"),
])
def test_bad_arguments_come_back_as_an_error_the_model_can_read(name, args, says):
    w = cw.World()
    assert says in json.loads(w.call(name, args))["error"]
    assert cw.effects(w) == {}


def test_worlds_do_not_share_writes():
    w = cw.World()
    w.call("crm_update_deal_stage", dict(deal_id="D0001", stage="Proposal"))
    assert cw.World().deal("D0001")["stage"] == lab.DEALS[0]["stage"]


@pytest.mark.parametrize("task", cw.TASKS, ids=lambda t: t.text[:40])
def test_the_solution_completes_the_task_and_wrong_ones_do_not(task):
    assert cw.grade(task, cw.solve(task)) == "completed"
    assert cw.grade(task, cw.World()) == "nothing changed"
    if len(task.solution) > 1:
        assert cw.grade(task, cw.solve(cw.Task(task.text, task.solution[:-1], task.words))) == "incomplete"
    stray = cw.Task(task.text, (*task.solution, ("crm_add_deal_note", dict(deal_id="D0600", text="x"))), task.words)
    assert cw.grade(task, cw.solve(stray)) == "wrong deals"
    for kind, key in cw._WANT[task.text]:
        if kind in ("note", "email"):
            name, args = next((n, a) for n, a in task.solution if n in ("crm_add_deal_note", "send_email"))
            wordless = cw.Task(task.text, ((name, dict(args, text="Done.") if "text" in args else dict(args, subject="x", body="Done.")),),
                               task.words)
            assert cw.grade(task, cw.solve(wordless)) != "completed"
