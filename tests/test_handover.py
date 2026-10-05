"""Tests for plan-execution handover (接棒): continue an unfinished execution
under a new plan, starting from the progress already on the books."""
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import execution
from app.main import app
from app.models import (
    ExecutionRegisterRequest,
    HandoverRequest,
    StageCommitRequest,
)
from app.execution import (
    CommitRejected as CommitStageRejected,
    HandoverRejected,
    handover_execution,
    register_execution,
)

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "rotation.db"
    monkeypatch.setenv("ROTATION_DB_PATH", str(db))
    execution.reset_store()
    yield
    execution.reset_store()


def cert(cid, subject, issuer, nvb=None, nva=None):
    return {
        "cert_id": cid,
        "subject": subject,
        "issuer": issuer,
        "not_valid_before": (nvb or NOW - timedelta(days=365)).isoformat(),
        "not_valid_after": (nva or NOW + timedelta(days=365)).isoformat(),
    }


SCENARIO = {
    "certificates": [
        cert("root-old", "CN=Old Root CA", "CN=Old Root CA"),
        cert("root-new", "CN=New Root CA", "CN=New Root CA"),
        cert("leaf-old", "CN=svc", "CN=Old Root CA",
             nva=NOW + timedelta(days=600)),
        cert("leaf-new", "CN=svc", "CN=New Root CA",
             nvb=NOW - timedelta(days=200)),
    ],
    "trust_anchors": [{"client_id": "c", "cert_id": "root-old"}],
    "endpoints": [
        {
            "endpoint_id": "svc",
            "current_chain": ["leaf-old", "root-old"],
            "target_chain": ["leaf-new", "root-new"],
        }
    ],
    "relationships": [{"client_id": "c", "endpoint_id": "svc"}],
    "actions": [
        {"action_id": "a1-add", "type": "add_trust", "client_id": "c",
         "anchor_cert_id": "root-new", "allowed_stages": [1, 2, 3]},
        {"action_id": "a2-replace", "type": "replace_cert", "endpoint_id": "svc",
         "new_chain": ["leaf-new", "root-new"], "allowed_stages": [2, 3]},
        {"action_id": "a3-remove", "type": "remove_trust", "client_id": "c",
         "anchor_cert_id": "root-old", "allowed_stages": [1, 2, 3]},
    ],
    "plan_time": NOW.isoformat(),
}

PLAN = [
    {"stage": 1, "action_ids": ["a1-add"]},
    {"stage": 2, "action_ids": ["a2-replace", "a3-remove"]},
]

STAGE1_COMMIT = {
    "expected_version": 0,
    "stage": 1,
    "action_ids": ["a1-add"],
    "anchor_observations": [
        {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
        {"client_id": "c", "anchor_cert_id": "root-old", "present": True},
    ],
    "chain_observations": [
        {"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}
    ],
}


def _continuation_scenario(expected_version=1):
    """A valid new scenario starting exactly at the post-stage-1 snapshot:

    anchors {root-old, root-new}, svc still serving [leaf-old, root-old];
    continuation rotates to a second new root, then removes both old anchors.
    """
    return {
        "certificates": [
            cert("root-old", "CN=Old Root CA", "CN=Old Root CA"),
            cert("root-new", "CN=New Root CA", "CN=New Root CA"),
            cert("leaf-old", "CN=svc", "CN=Old Root CA",
                 nva=NOW + timedelta(days=600)),
            cert("leaf-new", "CN=svc", "CN=New Root CA",
                 nvb=NOW - timedelta(days=200)),
            cert("root-new2", "CN=New Root CA 2", "CN=New Root CA 2",
                 nvb=NOW - timedelta(days=300)),
            cert("leaf-new2", "CN=svc", "CN=New Root CA 2",
                 nvb=NOW - timedelta(days=200)),
        ],
        "trust_anchors": [
            {"client_id": "c", "cert_id": "root-old"},
            {"client_id": "c", "cert_id": "root-new"},
        ],
        "endpoints": [
            {
                "endpoint_id": "svc",
                "current_chain": ["leaf-old", "root-old"],
                "target_chain": ["leaf-new2", "root-new2"],
            }
        ],
        "relationships": [{"client_id": "c", "endpoint_id": "svc"}],
        "actions": [
            {"action_id": "b1-add2", "type": "add_trust", "client_id": "c",
             "anchor_cert_id": "root-new2", "allowed_stages": [1, 2]},
            {"action_id": "b2-replace2", "type": "replace_cert",
             "endpoint_id": "svc", "new_chain": ["leaf-new2", "root-new2"],
             "allowed_stages": [2]},
            {"action_id": "b3-remove-old", "type": "remove_trust",
             "client_id": "c", "anchor_cert_id": "root-old",
             "allowed_stages": [2]},
            {"action_id": "b4-remove-new1", "type": "remove_trust",
             "client_id": "c", "anchor_cert_id": "root-new",
             "allowed_stages": [2]},
        ],
        "plan_time": NOW.isoformat(),
        "expected_version": expected_version,
        "plan": [
            {"stage": 1, "action_ids": ["b1-add2"]},
            {"stage": 2,
             "action_ids": ["b2-replace2", "b3-remove-old", "b4-remove-new1"]},
        ],
    }


def _register_parent(plan=PLAN, scenario=SCENARIO):
    r = client.post("/plan-executions", json={**scenario, "plan": plan})
    assert r.status_code == 201, r.text
    return r.json()["execution_id"]


def _commit_stage_one(eid):
    r = client.post(f"/plan-executions/{eid}/stages", json=STAGE1_COMMIT)
    assert r.status_code == 200, r.text
    return r.json()


def _handover(eid, payload, expected=201):
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == expected, r.text
    return r


def _execution_rows():
    with sqlite3.connect(execution.db_path()) as conn:
        return conn.execute(
            "SELECT execution_id, current_version, lifecycle, successor_id, "
            "predecessor_id, predecessor_version FROM plan_executions"
        ).fetchall()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_handover_registers_successor_at_version_zero_from_latest_snapshot():
    eid = _register_parent()
    _commit_stage_one(eid)

    body = _handover(eid, _continuation_scenario()).json()
    cid = body["execution_id"]
    assert cid != eid
    assert body["version"] == 0
    assert body["stage_count"] == 2
    assert body["completed_stages"] == []
    assert body["next_stage"] == 1
    assert body["finished"] is False
    assert body["lifecycle"] == "active"
    assert body["successor_execution_id"] is None
    assert body["predecessor_execution_id"] == eid
    assert body["predecessor_version"] == 1

    # The successor's version-0 snapshot is the predecessor's latest snapshot:
    # same anchors, same endpoint chain, same valid relationships.
    assert len(body["snapshots"]) == 1
    snap = body["snapshots"][0]
    assert snap["version"] == 0 and snap["stage"] == 0
    assert snap["action_ids"] == []
    assert snap["anchor_observations"] == []
    assert snap["chain_observations"] == []
    anchors = sorted(
        (a["client_id"], a["anchor_cert_id"]) for a in snap["trust_anchors"]
    )
    assert anchors == [("c", "root-new"), ("c", "root-old")]
    assert snap["endpoint_chains"] == [
        {"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}
    ]
    rel = snap["relationships"][0]
    assert rel["client_id"] == "c" and rel["endpoint_id"] == "svc"
    assert rel["valid"] is True
    assert [c["cert_id"] for c in rel["chain"]] == ["leaf-old", "root-old"]
    assert rel["trusted_anchors"] == ["root-new", "root-old"]

    old = client.get(f"/plan-executions/{eid}").json()
    assert old["lifecycle"] == "handed_over"
    assert old["successor_execution_id"] == cid
    assert old["predecessor_execution_id"] is None
    assert old["version"] == 1  # its own progress is untouched
    assert old["next_stage"] is None
    assert old["finished"] is False
    assert len(old["snapshots"]) == 2  # nothing removed or rewritten


def test_handover_at_version_zero_uses_initial_snapshot():
    eid = _register_parent()
    # No stage committed; the continuation must match the version-0 snapshot:
    # only root-old trusted, svc serving leaf-old.
    payload = _continuation_scenario(expected_version=0)
    payload["trust_anchors"] = [{"client_id": "c", "cert_id": "root-old"}]
    body = _handover(eid, payload).json()
    assert body["version"] == 0
    assert body["predecessor_version"] == 0
    anchors = [
        (a["client_id"], a["anchor_cert_id"])
        for a in body["snapshots"][0]["trust_anchors"]
    ]
    assert anchors == [("c", "root-old")]


def test_successor_stages_commit_normally_to_completion():
    eid = _register_parent()
    _commit_stage_one(eid)
    cid = _handover(eid, _continuation_scenario()).json()["execution_id"]

    r = client.post(
        f"/plan-executions/{cid}/stages",
        json={"expected_version": 0, "stage": 1, "action_ids": ["b1-add2"]},
    )
    assert r.status_code == 200, r.text

    r = client.post(
        f"/plan-executions/{cid}/stages",
        json={
            "expected_version": 1, "stage": 2,
            "action_ids": ["b2-replace2", "b3-remove-old", "b4-remove-new1"],
            "anchor_observations": [
                {"client_id": "c", "anchor_cert_id": "root-new2",
                 "present": True}
            ],
            "chain_observations": [
                {"endpoint_id": "svc", "chain": ["leaf-new2", "root-new2"]}
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["finished"] is True and body["next_stage"] is None
    final = body["snapshots"][-1]
    assert [c["cert_id"] for c in final["relationships"][0]["chain"]] == [
        "leaf-new2", "root-new2"
    ]
    assert final["relationships"][0]["trusted_anchors"] == ["root-new2"]


def test_handover_accepts_noop_follow_up_stage():
    eid = _register_parent()
    _commit_stage_one(eid)
    payload = _continuation_scenario()
    # Nothing is eligible at continuation stage 1; stage 1 is an explicit gap.
    for act in payload["actions"]:
        act["allowed_stages"] = [2]
    payload["plan"] = [
        {"stage": 1, "action_ids": []},
        {"stage": 2,
         "action_ids": ["b1-add2", "b2-replace2", "b3-remove-old",
                        "b4-remove-new1"]},
    ]
    # The single-stage batch adds root-new2 before removing the old anchors,
    # so the relationship survives the joint stage.
    cid = _handover(eid, payload).json()["execution_id"]
    r = client.post(
        f"/plan-executions/{cid}/stages",
        json={"expected_version": 0, "stage": 1, "action_ids": []},
    )
    assert r.status_code == 200, r.text
    assert r.json()["snapshots"][-1]["action_ids"] == []


def test_handover_can_chain_across_generations():
    eid = _register_parent()
    _commit_stage_one(eid)
    cid = _handover(eid, _continuation_scenario()).json()["execution_id"]
    # Commit the successor's stage 1 (add root-new2), then hand over again.
    r = client.post(
        f"/plan-executions/{cid}/stages",
        json={"expected_version": 0, "stage": 1, "action_ids": ["b1-add2"]},
    )
    assert r.status_code == 200
    grand_payload = _continuation_scenario(expected_version=1)
    # At the child's v1 snapshot anchors are {root-old, root-new, root-new2}.
    grand_payload["trust_anchors"].append(
        {"client_id": "c", "cert_id": "root-new2"}
    )
    grand_payload["actions"] = [
        {"action_id": "g1-replace", "type": "replace_cert",
         "endpoint_id": "svc", "new_chain": ["leaf-new2", "root-new2"],
         "allowed_stages": [1]},
        {"action_id": "g2-remove-old", "type": "remove_trust", "client_id": "c",
         "anchor_cert_id": "root-old", "allowed_stages": [1]},
        {"action_id": "g3-remove-new1", "type": "remove_trust", "client_id": "c",
         "anchor_cert_id": "root-new", "allowed_stages": [1]},
    ]
    grand_payload["plan"] = [
        {"stage": 1,
         "action_ids": ["g1-replace", "g2-remove-old", "g3-remove-new1"]},
    ]
    grand = _handover(cid, grand_payload).json()
    assert grand["predecessor_execution_id"] == cid
    assert grand["predecessor_version"] == 1

    middle = client.get(f"/plan-executions/{cid}").json()
    assert middle["lifecycle"] == "handed_over"
    assert middle["predecessor_execution_id"] == eid
    assert middle["successor_execution_id"] == grand["execution_id"]

    first = client.get(f"/plan-executions/{eid}").json()
    assert first["successor_execution_id"] == cid  # lineage is not rewritten


# ---------------------------------------------------------------------------
# Guards: version / lifecycle / finished / unknown
# ---------------------------------------------------------------------------


def test_handover_stale_expected_version_is_409_and_creates_no_child():
    eid = _register_parent()
    _commit_stage_one(eid)  # current version is now 1
    r = client.post(
        f"/plan-executions/{eid}/handover",
        json=_continuation_scenario(expected_version=0),
    )
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "version_conflict"
    assert body["current_version"] == 1
    assert body["expected_version"] == 1
    assert body["successor_execution_id"] is None
    rows = _execution_rows()
    assert len(rows) == 1 and rows[0][0] == eid and rows[0][2] == "active"


def test_handover_after_parent_committed_next_stage_is_version_conflict():
    eid = _register_parent()
    _commit_stage_one(eid)
    # Parent commits its final stage first: the handover based on v1 must lose.
    r = client.post(
        f"/plan-executions/{eid}/stages",
        json={
            "expected_version": 1, "stage": 2,
            "action_ids": ["a2-replace", "a3-remove"],
            "anchor_observations": [
                {"client_id": "c", "anchor_cert_id": "root-new",
                 "present": True},
                {"client_id": "c", "anchor_cert_id": "root-old",
                 "present": False},
            ],
            "chain_observations": [
                {"endpoint_id": "svc", "chain": ["leaf-new", "root-new"]}
            ],
        },
    )
    assert r.status_code == 200
    # The handover still bases itself on v1: the version advanced first, so the
    # later handover loses with version_conflict and creates no successor.
    r = client.post(
        f"/plan-executions/{eid}/handover",
        json=_continuation_scenario(expected_version=1),
    )
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "version_conflict"
    assert body["current_version"] == 2
    assert len(_execution_rows()) == 1


def test_second_handover_fails_and_leaves_no_child_record():
    eid = _register_parent()
    _commit_stage_one(eid)
    payload = _continuation_scenario()
    first = _handover(eid, payload).json()
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "already_handed_over"
    assert body["successor_execution_id"] == first["execution_id"]
    rows = _execution_rows()
    assert len(rows) == 2
    child_rows = [row for row in rows if row[0] != eid]
    assert len(child_rows) == 1
    assert child_rows[0][0] == first["execution_id"]


def test_handover_of_finished_execution_is_409():
    eid = _register_parent()
    _commit_stage_one(eid)
    r = client.post(
        f"/plan-executions/{eid}/stages",
        json={
            "expected_version": 1, "stage": 2,
            "action_ids": ["a3-remove", "a2-replace"],
            "anchor_observations": [
                {"client_id": "c", "anchor_cert_id": "root-new",
                 "present": True},
            ],
            "chain_observations": [
                {"endpoint_id": "svc", "chain": ["leaf-new", "root-new"]}
            ],
        },
    )
    assert r.status_code == 200 and r.json()["finished"] is True
    r = client.post(
        f"/plan-executions/{eid}/handover",
        json=_continuation_scenario(expected_version=2),
    )
    assert r.status_code == 409
    assert r.json()["reason"] == "execution_finished"


def test_handover_unknown_execution_is_404():
    r = client.post(
        "/plan-executions/ghost/handover",
        json=_continuation_scenario(expected_version=0),
    )
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_handover_requires_follow_up_plan():
    eid = _register_parent()
    payload = _continuation_scenario()
    payload["plan"] = []
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Concurrency: the two changes land together exactly once
# ---------------------------------------------------------------------------


def test_concurrent_handovers_one_wins_one_is_already_handed_over():
    eid = register_execution(
        ExecutionRegisterRequest.model_validate({**SCENARIO, "plan": PLAN})
    ).execution_id
    execution.commit_stage(
        eid, StageCommitRequest.model_validate(STAGE1_COMMIT)
    )
    payload = _continuation_scenario()
    results: list[str] = []

    def do_handover():
        try:
            handover_execution(eid, HandoverRequest.model_validate(payload))
            results.append("won")
        except HandoverRejected as exc:
            results.append(exc.conflict.reason)

    threads = [threading.Thread(target=do_handover) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == ["already_handed_over", "won"], results
    rows = _execution_rows()
    assert len(rows) == 2
    parent = next(row for row in rows if row[0] == eid)
    assert parent[2] == "handed_over" and parent[3] is not None
    children = [row for row in rows if row[0] != eid]
    assert len(children) == 1
    assert children[0][4] == eid and children[0][5] == 1


def test_concurrent_handover_and_stage_commit_exactly_one_takes_effect():
    eid = register_execution(
        ExecutionRegisterRequest.model_validate({**SCENARIO, "plan": PLAN})
    ).execution_id
    execution.commit_stage(
        eid, StageCommitRequest.model_validate(STAGE1_COMMIT)
    )
    outcomes: list[str] = []

    def do_handover():
        try:
            handover_execution(
                eid, HandoverRequest.model_validate(_continuation_scenario())
            )
            outcomes.append("handover")
        except HandoverRejected:
            outcomes.append("handover_lost")

    def do_commit():
        try:
            execution.commit_stage(
                eid,
                StageCommitRequest.model_validate(
                    {
                        "expected_version": 1,
                        "stage": 2,
                        "action_ids": ["a2-replace", "a3-remove"],
                        "anchor_observations": [
                            {"client_id": "c", "anchor_cert_id": "root-new",
                             "present": True},
                            {"client_id": "c", "anchor_cert_id": "root-old",
                             "present": False},
                        ],
                        "chain_observations": [
                            {"endpoint_id": "svc",
                             "chain": ["leaf-new", "root-new"]}
                        ],
                    }
                ),
            )
            outcomes.append("commit")
        except CommitStageRejected:
            outcomes.append("commit_lost")

    t1 = threading.Thread(target=do_handover)
    t2 = threading.Thread(target=do_commit)
    t1.start(); t2.start()
    t1.join(); t2.join()

    rows = {row[0]: row for row in _execution_rows()}
    parent = rows[eid]
    if "handover" in outcomes:
        # Handover won: parent frozen at v1/handed_over, commit must have lost.
        assert outcomes == ["handover", "commit_lost"] or \
            set(outcomes) == {"handover", "commit_lost"}
        assert parent[1] == 1 and parent[2] == "handed_over"
        assert len(rows) == 2
    else:
        # Commit won: v2/active, handover must have lost with version_conflict.
        assert set(outcomes) == {"handover_lost", "commit"}
        assert parent[1] == 2 and parent[2] == "active"
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Starting-point consistency (422 structured detail)
# ---------------------------------------------------------------------------


def test_handover_rejects_diverging_initial_anchors():
    eid = _register_parent()
    _commit_stage_one(eid)
    payload = _continuation_scenario()
    payload["trust_anchors"] = [
        {"client_id": "c", "cert_id": "root-old"},
        {"client_id": "c", "cert_id": "root-new2"},  # rewrites the anchor set
    ]
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "handover_snapshot_mismatch"
    assert detail["execution_id"] == eid
    assert detail["expected_version"] == 1
    assert detail["current_version"] == 1
    components = {(m["component"], m["anchor_cert_id"])
                  for m in detail["mismatches"]}
    assert ("trust_anchors", "root-new") in components
    assert ("trust_anchors", "root-new2") in components
    assert len(_execution_rows()) == 1


def test_handover_rejects_diverging_endpoint_chain():
    eid = _register_parent()
    _commit_stage_one(eid)
    payload = _continuation_scenario()
    payload["endpoints"][0]["current_chain"] = ["leaf-new", "root-new"]
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == 422
    mm = r.json()["detail"]["mismatches"]
    assert any(
        m["component"] == "endpoint_chains"
        and m["endpoint_id"] == "svc"
        and m["expected"] == ["leaf-old", "root-old"]
        and m["found"] == ["leaf-new", "root-new"]
        for m in mm
    )
    assert len(_execution_rows()) == 1


def test_handover_rejects_dropped_endpoint():
    eid = _register_parent()
    _commit_stage_one(eid)
    payload = _continuation_scenario()
    payload["endpoints"] = []
    payload["relationships"] = []
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == 422
    # Engine reference validation may answer first (relationships/endpoints),
    # but nothing may have been created.
    assert len(_execution_rows()) == 1


def test_handover_rejects_diverging_relationships():
    eid = _register_parent()
    _commit_stage_one(eid)
    payload = _continuation_scenario()
    payload["relationships"] = [
        {"client_id": "c", "endpoint_id": "svc"},
        {"client_id": "other-client", "endpoint_id": "svc"},
    ]
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(
        m["component"] == "relationships"
        and m["client_id"] == "other-client"
        for m in detail.get("mismatches", [])
    ) or detail["error"] == "handover_snapshot_mismatch"


def test_handover_rejects_rewritten_certificate_attributes():
    eid = _register_parent()
    _commit_stage_one(eid)
    payload = _continuation_scenario()
    # Same cert_id, attribute changed (but still valid at plan_time so the
    # engine itself would accept the scenario): must be rejected as a rewrite.
    reused = next(
        c for c in payload["certificates"] if c["cert_id"] == "root-old"
    )
    reused["not_valid_before"] = (NOW - timedelta(days=999)).isoformat()
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == 422
    mm = r.json()["detail"]["mismatches"]
    item = next(m for m in mm if m["component"] == "certificate")
    assert item["cert_id"] == "root-old"
    assert "not_valid_before" in item["attribute"]
    assert len(_execution_rows()) == 1


def test_handover_allows_new_certificates_in_targets_and_actions():
    eid = _register_parent()
    _commit_stage_one(eid)
    body = _handover(eid, _continuation_scenario()).json()
    # root-new2 / leaf-new2 exist only in the successor scenario.
    export = client.get(
        f"/plan-executions/{body['execution_id']}/export"
    ).json()
    ids = {c["cert_id"] for c in export["scenario"]["certificates"]}
    assert {"root-new2", "leaf-new2"} <= ids


# ---------------------------------------------------------------------------
# Continuation plan must keep communicating at every stage
# ---------------------------------------------------------------------------


def test_handover_rejects_continuation_that_breaks_communication():
    eid = _register_parent()
    _commit_stage_one(eid)
    payload = _continuation_scenario()
    # Replace to root-new2 and strip every other anchor in one stage before
    # root-new2 is trusted: the post-stage relationship has no trust path.
    payload["actions"][0]["allowed_stages"] = [1, 2]  # b1-add2
    payload["plan"] = [
        {"stage": 1,
         "action_ids": ["b2-replace2", "b3-remove-old", "b4-remove-new1"]},
        {"stage": 2, "action_ids": ["b1-add2"]},
    ]
    r = client.post(f"/plan-executions/{eid}/handover", json=payload)
    assert r.status_code == 409
    body = r.json()
    # Same Failure shape as /plans/verify.
    assert body["status"] == "invalid"
    assert body["stage"] == 1
    assert body["reason"] in {"untrusted_root", "action_not_allowed_at_stage"}
    # Atomicity: the failed validation happened before the transaction; the
    # parent stays active and no successor row exists.
    rows = _execution_rows()
    assert len(rows) == 1
    assert rows[0][2] == "active" and rows[0][3] is None


# ---------------------------------------------------------------------------
# Old execution frozen after handover; idempotent replay unchanged
# ---------------------------------------------------------------------------


def test_old_execution_rejects_unfinished_stage_after_handover():
    eid = _register_parent()
    _commit_stage_one(eid)
    _handover(eid, _continuation_scenario())

    r = client.post(
        f"/plan-executions/{eid}/stages",
        json={
            "expected_version": 1, "stage": 2,
            "action_ids": ["a2-replace", "a3-remove"],
        },
    )
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "execution_handed_over"
    status = client.get(f"/plan-executions/{eid}").json()
    assert status["version"] == 1  # nothing advanced


def test_committed_stage_idempotent_replay_still_works_after_handover():
    eid = _register_parent()
    first = _commit_stage_one(eid)
    _handover(eid, _continuation_scenario())

    # Same content, based on the post-commit current version (1): still returns
    # the identical stored snapshot per the original idempotency rule.
    replay = client.post(
        f"/plan-executions/{eid}/stages",
        json={**STAGE1_COMMIT, "expected_version": 1},
    )
    assert replay.status_code == 200, replay.text
    body = replay.json()
    assert body["version"] == 1
    assert body["snapshots"][1]["committed_at"] == \
        first["snapshots"][1]["committed_at"]

    # A stale replay is still a version_conflict, never a silent commit.
    stale = client.post(
        f"/plan-executions/{eid}/stages", json=STAGE1_COMMIT
    )
    assert stale.status_code == 409
    assert stale.json()["reason"] == "version_conflict"


# ---------------------------------------------------------------------------
# Traceability: status, export and off-site verification on both sides
# ---------------------------------------------------------------------------


def test_exports_carry_lineage_and_verify_off_site_on_both_sides():
    eid = _register_parent()
    _commit_stage_one(eid)
    cid = _handover(eid, _continuation_scenario()).json()["execution_id"]

    parent_export = client.get(f"/plan-executions/{eid}/export").json()
    assert parent_export["lifecycle"] == "handed_over"
    assert parent_export["successor_execution_id"] == cid
    assert parent_export["predecessor_execution_id"] is None
    assert parent_export["next_stage"] is None
    assert parent_export["finished"] is False

    child_export = client.get(f"/plan-executions/{cid}/export").json()
    assert child_export["lifecycle"] == "active"
    assert child_export["predecessor_execution_id"] == eid
    assert child_export["predecessor_version"] == 1
    assert child_export["version"] == 0
    assert len(child_export["commits"]) == 1

    for pkg in (parent_export, child_export):
        verdict = client.post("/execution-verifications", json=pkg).json()
        assert verdict["status"] == "ok", pkg["execution_id"]
        assert verdict["lifecycle"] == pkg["lifecycle"]
        assert verdict["successor_execution_id"] == pkg["successor_execution_id"]
        assert verdict["predecessor_execution_id"] == \
            pkg["predecessor_execution_id"]
        assert verdict["predecessor_version"] == pkg["predecessor_version"]

    parent_verdict = client.post(
        "/execution-verifications", json=parent_export
    ).json()
    # The unfinished, handed-over prefix verifies exactly as far as it got.
    assert parent_verdict["verified_version"] == 1
    assert parent_verdict["next_stage"] is None
    assert parent_verdict["finished"] is False


def test_handed_over_export_with_forged_next_stage_fails_verification():
    eid = _register_parent()
    _commit_stage_one(eid)
    _handover(eid, _continuation_scenario())
    pkg = client.get(f"/plan-executions/{eid}/export").json()
    # A tampered export claiming another stage remains due must conflict.
    pkg["next_stage"] = 2
    verdict = client.post("/execution-verifications", json=pkg).json()
    assert verdict["status"] == "conflict"
    assert verdict["mismatch"]["reason"] == "header_mismatch"


# ---------------------------------------------------------------------------
# SQLite migration for stores created by the pre-handover schema
# ---------------------------------------------------------------------------


def test_pre_handover_schema_is_migrated_transparently():
    db_path = execution.db_path()

    # Build an old-format database at the fixture path before the service
    # connects: no lifecycle / lineage columns.
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE plan_executions (
                execution_id    TEXT PRIMARY KEY,
                scenario_json   TEXT NOT NULL,
                plan_json       TEXT NOT NULL,
                stage_count     INTEGER NOT NULL,
                current_version INTEGER NOT NULL,
                created_at      TEXT NOT NULL
            );
            CREATE TABLE stage_commits (
                execution_id  TEXT NOT NULL,
                version       INTEGER NOT NULL,
                stage         INTEGER NOT NULL,
                payload_json  TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                committed_at  TEXT NOT NULL,
                PRIMARY KEY (execution_id, version)
            );
            """
        )
        conn.execute(
            "INSERT INTO plan_executions "
            "(execution_id, scenario_json, plan_json, stage_count, "
            "current_version, created_at) VALUES ('legacy-exec', '{}', "
            "'[]', 0, 0, ?)",
            (NOW.isoformat(),),
        )
        old_cols = {
            r[1] for r in conn.execute(
                "PRAGMA table_info(plan_executions)"
            ).fetchall()
        }
    assert "lifecycle" not in old_cols

    # The next service connection sees the old table and migrates it in
    # place; the legacy row defaults to lifecycle 'active'.
    execution.reset_store()
    conn = execution._connect()
    try:
        cols = {
            r["name"] for r in conn.execute(
                "PRAGMA table_info(plan_executions)"
            ).fetchall()
        }
        assert {
            "lifecycle", "successor_id", "predecessor_id",
            "predecessor_version",
        } <= cols
    finally:
        conn.close()

    status = execution.get_execution("legacy-exec")
    assert status.lifecycle == "active"
    assert status.successor_execution_id is None
    assert status.predecessor_execution_id is None
