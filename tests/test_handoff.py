"""Tests for mid-execution handoffs to a new continuation plan."""
import threading

import pytest
from fastapi.testclient import TestClient

from app import execution
from app.main import app
from app.models import (
    ExecutionHandoffRequest,
    ExecutionRegisterRequest,
    StageCommitRequest,
)

NOW = "2026-06-01T00:00:00Z"

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    db = tmp_path / "rotation.db"
    monkeypatch.setenv("ROTATION_DB_PATH", str(db))
    execution.reset_store()
    yield
    execution.reset_store()


def cert(cid, subject, issuer, nbf="2025-01-01T00:00:00Z",
         naf="2027-12-31T00:00:00Z"):
    return {
        "cert_id": cid,
        "subject": subject,
        "issuer": issuer,
        "not_valid_before": nbf,
        "not_valid_after": naf,
    }


SCENARIO = {
    "certificates": [
        cert("root-old", "CN=Old Root CA", "CN=Old Root CA",
             "2020-01-01T00:00:00Z", "2030-01-01T00:00:00Z"),
        cert("root-new", "CN=New Root CA", "CN=New Root CA",
             "2020-01-01T00:00:00Z", "2035-01-01T00:00:00Z"),
        cert("leaf-old", "CN=svc", "CN=Old Root CA"),
        cert("leaf-new", "CN=svc", "CN=New Root CA"),
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
    "plan_time": NOW,
}

PLAN = [
    {"stage": 1, "action_ids": ["a1-add"]},
    {"stage": 2, "action_ids": ["a2-replace", "a3-remove"]},
]


def register(plan=PLAN, scenario=SCENARIO):
    r = client.post("/plan-executions", json={**scenario, "plan": plan})
    assert r.status_code == 201, r.text
    return r.json()


def commit(exec_id, expected, stage, actions, anchors=(), chains=()):
    return client.post(
        f"/plan-executions/{exec_id}/stages",
        json={
            "expected_version": expected,
            "stage": stage,
            "action_ids": actions,
            "anchor_observations": anchors,
            "chain_observations": chains,
        },
    )


# New scenario after stage 1: both roots trusted, endpoint still serving the
# old chain. The continuation rotates onto a fresh leaf (leaf-alt) and only
# then removes the old root.
def new_scenario(extra_cert=True):
    certs = [
        cert("root-old", "CN=Old Root CA", "CN=Old Root CA",
             "2020-01-01T00:00:00Z", "2030-01-01T00:00:00Z"),
        cert("root-new", "CN=New Root CA", "CN=New Root CA",
             "2020-01-01T00:00:00Z", "2035-01-01T00:00:00Z"),
        cert("leaf-old", "CN=svc", "CN=Old Root CA"),
        cert("leaf-new", "CN=svc", "CN=New Root CA"),
    ]
    if extra_cert:
        certs.append(cert("leaf-alt", "CN=svc-alt", "CN=New Root CA"))
    return {
        "certificates": certs,
        "trust_anchors": [
            {"client_id": "c", "cert_id": "root-old"},
            {"client_id": "c", "cert_id": "root-new"},
        ],
        "endpoints": [
            {
                "endpoint_id": "svc",
                "current_chain": ["leaf-old", "root-old"],
                "target_chain": ["leaf-alt", "root-new"],
            }
        ],
        "relationships": [{"client_id": "c", "endpoint_id": "svc"}],
        "actions": [
            {"action_id": "b1-replace", "type": "replace_cert",
             "endpoint_id": "svc",
             "new_chain": ["leaf-alt", "root-new"],
             "allowed_stages": [1, 2]},
            {"action_id": "b2-remove", "type": "remove_trust",
             "client_id": "c", "anchor_cert_id": "root-old",
             "allowed_stages": [2]},
        ],
        "plan_time": NOW,
    }


NEW_PLAN = [
    {"stage": 1, "action_ids": ["b1-replace"]},
    {"stage": 2, "action_ids": ["b2-remove"]},
]


def handoff(exec_id, expected_version, scenario=None, plan=NEW_PLAN):
    scenario = new_scenario() if scenario is None else scenario
    return client.post(
        f"/plan-executions/{exec_id}/handoffs",
        json={**scenario, "expected_version": expected_version, "plan": plan},
    )


# ---------------------------------------------------------------------------
# Basic handoff
# ---------------------------------------------------------------------------


def test_handoff_after_one_stage_starts_child_from_recorded_snapshot():
    parent = register()
    pid = parent["execution_id"]
    r = commit(
        pid, 0, 1, ["a1-add"],
        anchors=[
            {"client_id": "c", "anchor_cert_id": "root-old", "present": True},
            {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
        ],
        chains=[{"endpoint_id": "svc", "chain": ["leaf-old", "root-old"]}],
    )
    assert r.status_code == 200, r.text

    r = handoff(pid, 1)
    assert r.status_code == 201, r.text
    body = r.json()

    p = body["parent"]
    ch = body["child"]

    assert p["execution_id"] == pid
    assert p["version"] == 1
    assert p["handed_off"] is True
    assert p["handoff"]["child_execution_id"] == ch["execution_id"]
    assert p["handoff"]["parent_version"] == 1
    assert p["origin"] is None
    assert len(p["snapshots"]) == 2  # untouched history

    assert ch["execution_id"] != pid
    assert ch["version"] == 0
    assert ch["stage_count"] == 2
    assert ch["next_stage"] == 1
    assert ch["finished"] is False
    assert ch["handed_off"] is False
    assert ch["origin"] == {
        "kind": "handoff",
        "parent_execution_id": pid,
        "parent_version": 1,
        "handed_off_at": ch["origin"]["handed_off_at"],
    }

    snap0 = ch["snapshots"][0]
    assert snap0["version"] == 0
    assert snap0["stage"] == 0
    assert snap0["action_ids"] == []
    anchors = sorted(
        (a["client_id"], a["anchor_cert_id"])
        for a in snap0["trust_anchors"]
    )
    assert anchors == [("c", "root-new"), ("c", "root-old")]
    assert snap0["endpoint_chains"][0]["chain"] == ["leaf-old", "root-old"]
    assert all(rel["valid"] for rel in snap0["relationships"])


def test_handoff_allowed_at_version_zero():
    parent = register()
    pid = parent["execution_id"]

    # A handoff before any stage commit: the new scenario equals the parent's
    # initial state and carries a different continuation plan.
    r = handoff(
        pid, 0,
        scenario={
            **SCENARIO,
            "actions": [
                {"action_id": "z-add", "type": "add_trust", "client_id": "c",
                 "anchor_cert_id": "root-new", "allowed_stages": [1]},
                {"action_id": "z-replace", "type": "replace_cert",
                 "endpoint_id": "svc",
                 "new_chain": ["leaf-new", "root-new"],
                 "allowed_stages": [2]},
                {"action_id": "z-remove", "type": "remove_trust",
                 "client_id": "c", "anchor_cert_id": "root-old",
                 "allowed_stages": [2]},
            ],
        },
        plan=[
            {"stage": 1, "action_ids": ["z-add"]},
            {"stage": 2, "action_ids": ["z-replace", "z-remove"]},
        ],
    )
    assert r.status_code == 201, r.text
    child = r.json()["child"]
    assert child["version"] == 0
    assert child["origin"]["parent_version"] == 0
    anchors = [
        (a["client_id"], a["anchor_cert_id"])
        for a in child["snapshots"][0]["trust_anchors"]
    ]
    assert anchors == [("c", "root-old")]


def test_handoff_unknown_execution_is_404():
    r = handoff("no-such-execution", 0)
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Optimistic lock and single-winner semantics
# ---------------------------------------------------------------------------


def test_handoff_stale_expected_version_is_409_and_leaves_no_child():
    parent = register()
    pid = parent["execution_id"]
    r = commit(pid, 0, 1, ["a1-add"])
    assert r.status_code == 200

    r = handoff(pid, 0)  # the execution is already at version 1
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "version_conflict"
    assert body["current_version"] == 1
    assert body["expected_version"] == 1
    # nothing was created: only the parent exists and it is not handed off
    assert client.get(f"/plan-executions/{pid}").json()["handed_off"] is False


def test_second_handoff_is_409_and_creates_no_successor():
    parent = register()
    pid = parent["execution_id"]
    commit(pid, 0, 1, ["a1-add"])
    first = handoff(pid, 1)
    assert first.status_code == 201
    first_child = first.json()["child"]["execution_id"]

    r = handoff(pid, 1)
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "handoff_already_exists"
    assert body["child_execution_id"] == first_child
    # parent still points at the first child
    status = client.get(f"/plan-executions/{pid}").json()
    assert status["handoff"]["child_execution_id"] == first_child


def _handoff_service(exec_id, expected_version):
    try:
        execution.handoff_execution(
            exec_id,
            ExecutionHandoffRequest.model_validate(
                {**new_scenario(),
                 "expected_version": expected_version,
                 "plan": NEW_PLAN}
            ),
        )
        return 201
    except execution.CommitRejected as exc:
        return exc.conflict.reason


def test_handoff_racing_with_stage_commit_one_side_wins():
    parent = register()
    pid = parent["execution_id"]
    results = []

    # Stage 1 commits from version 0 at the same time as a handoff based on
    # version 0. Exactly one side wins: either the commit lands (then the
    # handoff loses the optimistic lock and creates no child) or the handoff
    # lands (then the commit is refused because the parent is handed off).
    def do_commit():
        try:
            execution.commit_stage(
                pid,
                StageCommitRequest(
                    expected_version=0, stage=1, action_ids=["a1-add"]
                ),
            )
            results.append("commit")
        except execution.CommitRejected:
            results.append("commit-rejected")

    t1 = threading.Thread(target=do_commit)
    t2 = threading.Thread(
        target=lambda: results.append(_handoff_service(pid, 0))
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    status = client.get(f"/plan-executions/{pid}").json()
    if status["handed_off"]:
        assert results == ["commit-rejected", 201]
        child_id = status["handoff"]["child_execution_id"]
        child = client.get(f"/plan-executions/{child_id}").json()
        # The child is still at version 0 (starting from the pre-stage-1
        # snapshot): the losing commit did not leak into it.
        assert child["version"] == 0
        assert child["snapshots"][0]["trust_anchors"] == [
            {"client_id": "c", "anchor_cert_id": "root-old"}
        ]
    else:
        # the commit won: version advanced to 1, handoff lost version lock
        assert status["version"] == 1
        assert "commit" in results and "version_conflict" in results
        assert status["handoff"] is None


# ---------------------------------------------------------------------------
# Snapshot anchoring (422)
# ---------------------------------------------------------------------------


def _handoff_422(scenario):
    parent = register()
    pid = parent["execution_id"]
    r = handoff(pid, 0, scenario=scenario)
    assert r.status_code == 422, r.text
    return pid, r.json()["detail"]


def test_handoff_rejects_changed_certificate_attributes():
    scenario = new_scenario()
    # Reuse root-old's id but change its subject.
    scenario["certificates"][0]["subject"] = "CN=Old Root CA RENAMED"
    # Keep the request internally consistent enough to reach the handoff
    # checks: the renamed root is self-signed, so update issuer as chains
    # are not rebuilt before the cert check; leaf-old still chains by name
    # to the old name — but the cert attribute check runs first.
    pid, detail = _handoff_422(scenario)
    assert detail["error"] == "handoff_snapshot_mismatch"
    assert detail["parent_execution_id"] == pid
    assert detail["parent_version"] == 0
    kinds = [m["kind"] for m in detail["mismatches"]]
    assert kinds == ["certificate_attribute_mismatch"]
    m = detail["mismatches"][0]
    assert m["cert_id"] == "root-old"
    assert m["expected"] == "CN=Old Root CA"
    assert m["found"] == "CN=Old Root CA RENAMED"
    # rejected handoff creates no lineage
    assert client.get(f"/plan-executions/{pid}").json()["handed_off"] is False


def test_handoff_rejects_validity_change_on_reused_certificate():
    scenario = new_scenario()
    scenario["certificates"][0]["not_valid_after"] = "2029-01-01T00:00:00Z"
    _, detail = _handoff_422(scenario)
    assert [m["kind"] for m in detail["mismatches"]] == [
        "certificate_attribute_mismatch"
    ]
    assert detail["mismatches"][0]["message"].endswith(
        "reused certificate ids must keep identical attributes."
    )


def test_handoff_rejects_anchor_set_mismatch():
    # Snapshot v0 trusts only root-old; declare an extra anchor the snapshot
    # does not have.
    scenario = new_scenario()
    scenario["certificates"].append(
        cert("root-extra", "CN=Extra Root CA", "CN=Extra Root CA",
             "2020-01-01T00:00:00Z", "2035-01-01T00:00:00Z")
    )
    scenario["trust_anchors"].append(
        {"client_id": "c", "cert_id": "root-extra"}
    )
    _, detail = _handoff_422(scenario)
    assert [m["kind"] for m in detail["mismatches"]] == [
        "trust_anchor_mismatch"
    ]
    m = detail["mismatches"][0]
    assert (m["client_id"], m["anchor_cert_id"]) == ("c", "root-extra")


def test_handoff_rejects_endpoint_chain_mismatch():
    parent = register()
    pid = parent["execution_id"]
    commit(pid, 0, 1, ["a1-add"])

    # Claim a different current chain than the recorded one.
    scenario = new_scenario()
    scenario["endpoints"][0]["current_chain"] = ["leaf-new", "root-new"]
    scenario["trust_anchors"] = [
        {"client_id": "c", "cert_id": "root-old"},
        {"client_id": "c", "cert_id": "root-new"},
    ]
    r = handoff(pid, 1, scenario=scenario)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert [m["kind"] for m in detail["mismatches"]] == [
        "endpoint_chain_mismatch"
    ]
    m = detail["mismatches"][0]
    assert m["endpoint_id"] == "svc"
    assert m["expected"] == ["leaf-old", "root-old"]
    assert m["found"] == ["leaf-new", "root-new"]


def test_handoff_rejects_added_endpoint():
    scenario = new_scenario()
    scenario["endpoints"].append(
        {"endpoint_id": "svc2",
         "current_chain": ["leaf-alt", "root-new"],
         "target_chain": ["leaf-alt", "root-new"]}
    )
    scenario["relationships"].append(
        {"client_id": "c", "endpoint_id": "svc2"}
    )
    _, detail = _handoff_422(scenario)
    kinds = {(m["kind"], m["endpoint_id"]) for m in detail["mismatches"]}
    assert ("endpoint_chain_mismatch", "svc2") in kinds


def test_handoff_rejects_relationship_set_mismatch():
    scenario = new_scenario()
    scenario["relationships"] = []  # the required relationship disappeared
    _, detail = _handoff_422(scenario)
    assert [m["kind"] for m in detail["mismatches"]] == [
        "relationship_mismatch"
    ]
    m = detail["mismatches"][0]
    assert (m["client_id"], m["endpoint_id"]) == ("c", "svc")


def test_handoff_rejects_unsafe_continuation_plan_with_409():
    parent = register()
    pid = parent["execution_id"]
    commit(pid, 0, 1, ["a1-add"])

    # The endpoint still serves leaf-old (chaining to root-old). Removing
    # root-old at continuation stage 1 before replacing the chain breaks the
    # relationship: the full plan replay must reject the handoff with 409 and
    # create no child.
    scenario = new_scenario()
    # b2-remove allowed at stage 1, but b1-replace only at stage 2: stage 1
    # is forced to remove root-old while the endpoint still serves leaf-old.
    scenario["actions"][1]["allowed_stages"] = [1]
    scenario["actions"][0]["allowed_stages"] = [2]
    r = handoff(
        pid, 1, scenario=scenario,
        plan=[
            {"stage": 1, "action_ids": ["b2-remove"]},
            {"stage": 2, "action_ids": ["b1-replace"]},
        ],
    )
    assert r.status_code == 409, r.text
    assert r.json()["reason"] == "untrusted_root"
    assert client.get(f"/plan-executions/{pid}").json()["handed_off"] is False


def test_handoff_requires_at_least_one_plan_stage():
    parent = register()
    pid = parent["execution_id"]
    r = client.post(
        f"/plan-executions/{pid}/handoffs",
        json={**SCENARIO, "expected_version": 0, "plan": []},
    )
    assert r.status_code == 422  # request validation (min_length=1)


def test_handoff_against_finished_execution_is_409():
    # Finish the original rotation (version 2 == stage_count), then propose
    # another rotation wave anchored at the final snapshot: the recorded
    # rotation has ended, so the handoff is refused and creates no child.
    parent = register()
    pid = parent["execution_id"]
    r = commit(pid, 0, 1, ["a1-add"])
    assert r.status_code == 200
    r = commit(
        pid, 1, 2, ["a2-replace", "a3-remove"],
        anchors=[
            {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
            {"client_id": "c", "anchor_cert_id": "root-old", "present": False},
        ],
        chains=[{"endpoint_id": "svc", "chain": ["leaf-new", "root-new"]}],
    )
    assert r.status_code == 200

    finished_scenario = {
        "certificates": [
            cert("root-new", "CN=New Root CA", "CN=New Root CA",
                 "2020-01-01T00:00:00Z", "2035-01-01T00:00:00Z"),
            cert("leaf-new", "CN=svc", "CN=New Root CA"),
            cert("leaf-alt", "CN=svc-alt2", "CN=New Root CA",
                 "2026-03-01T00:00:00Z", "2028-02-28T00:00:00Z"),
        ],
        "trust_anchors": [{"client_id": "c", "cert_id": "root-new"}],
        "endpoints": [
            {"endpoint_id": "svc",
             "current_chain": ["leaf-new", "root-new"],
             "target_chain": ["leaf-alt", "root-new"]}
        ],
        "relationships": [{"client_id": "c", "endpoint_id": "svc"}],
        "actions": [
            {"action_id": "c1-replace", "type": "replace_cert",
             "endpoint_id": "svc",
             "new_chain": ["leaf-alt", "root-new"],
             "allowed_stages": [1]}
        ],
        "plan_time": NOW,
    }
    r = handoff(
        pid, 2, scenario=finished_scenario,
        plan=[{"stage": 1, "action_ids": ["c1-replace"]}],
    )
    assert r.status_code == 409
    assert r.json()["reason"] == "execution_already_finished"
    assert client.get(f"/plan-executions/{pid}").json()["handed_off"] is False


# ---------------------------------------------------------------------------
# Post-handoff behaviour of the parent execution
# ---------------------------------------------------------------------------


def _parent_with_child():
    parent = register()
    pid = parent["execution_id"]
    commit(pid, 0, 1, ["a1-add"])
    body = handoff(pid, 1).json()
    return pid, body["child"]["execution_id"]


def test_parent_cannot_commit_unfinished_stage_after_handoff():
    pid, cid = _parent_with_child()
    r = commit(pid, 1, 2, ["a2-replace", "a3-remove"])
    assert r.status_code == 409
    body = r.json()
    assert body["reason"] == "execution_already_handed_off"
    assert body["child_execution_id"] == cid
    # no partial progress on the parent
    status = client.get(f"/plan-executions/{pid}").json()
    assert status["version"] == 1
    assert len(status["snapshots"]) == 2


def test_parent_idempotent_replay_of_committed_stage_still_works():
    pid, _ = _parent_with_child()
    # identical re-submission of stage 1 based on current version 1 returns
    # the same stored snapshot — the original idempotency rule is unchanged.
    r = commit(pid, 1, 1, ["a1-add"])
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert len(body["snapshots"]) == 2


def test_parent_replay_with_different_content_still_conflicts():
    pid, _ = _parent_with_child()
    r = commit(
        pid, 1, 1, ["a1-add"],
        anchors=[{"client_id": "c", "anchor_cert_id": "root-new",
                  "present": True}],
    )
    assert r.status_code == 409
    assert r.json()["reason"] == "duplicate_payload_mismatch"


def test_parent_stale_commit_after_handoff_is_version_conflict_first():
    pid, _ = _parent_with_child()
    r = commit(pid, 0, 1, ["a1-add"])
    assert r.status_code == 409
    assert r.json()["reason"] == "version_conflict"


# ---------------------------------------------------------------------------
# The child execution is an ordinary execution
# ---------------------------------------------------------------------------


def test_child_commits_stages_normally_and_finishes():
    pid, cid = _parent_with_child()

    r = commit(
        cid, 0, 1, ["b1-replace"],
        anchors=[
            {"client_id": "c", "anchor_cert_id": "root-old", "present": True},
            {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
        ],
        chains=[{"endpoint_id": "svc", "chain": ["leaf-alt", "root-new"]}],
    )
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 1
    assert r.json()["origin"]["parent_execution_id"] == pid

    r = commit(
        cid, 1, 2, ["b2-remove"],
        anchors=[
            {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
            {"client_id": "c", "anchor_cert_id": "root-old",
             "present": False},
        ],
        chains=[{"endpoint_id": "svc", "chain": ["leaf-alt", "root-new"]}],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["finished"] is True
    assert body["next_stage"] is None


def test_child_can_itself_handoff_again():
    pid, cid = _parent_with_child()
    commit(
        cid, 0, 1, ["b1-replace"],
        anchors=[
            {"client_id": "c", "anchor_cert_id": "root-old", "present": True},
            {"client_id": "c", "anchor_cert_id": "root-new", "present": True},
        ],
        chains=[{"endpoint_id": "svc", "chain": ["leaf-alt", "root-new"]}],
    )
    grand = new_scenario()
    # after child stage 1: both roots, serving leaf-alt
    grand["certificates"] = [
        cert("root-old", "CN=Old Root CA", "CN=Old Root CA",
             "2020-01-01T00:00:00Z", "2030-01-01T00:00:00Z"),
        cert("root-new", "CN=New Root CA", "CN=New Root CA",
             "2020-01-01T00:00:00Z", "2035-01-01T00:00:00Z"),
        cert("leaf-alt", "CN=svc-alt", "CN=New Root CA"),
    ]
    grand["endpoints"] = [
        {"endpoint_id": "svc",
         "current_chain": ["leaf-alt", "root-new"],
         "target_chain": ["leaf-alt", "root-new"]}
    ]
    grand["actions"] = [
        {"action_id": "g1-remove", "type": "remove_trust", "client_id": "c",
         "anchor_cert_id": "root-old", "allowed_stages": [1]}
    ]
    r = handoff(
        cid, 1, scenario=grand,
        plan=[{"stage": 1, "action_ids": ["g1-remove"]}],
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["parent"]["execution_id"] == cid
    grandchild = body["child"]
    assert grandchild["origin"]["parent_execution_id"] == cid
    assert grandchild["origin"]["parent_version"] == 1
    # the middle execution carries lineage on both sides
    middle = client.get(f"/plan-executions/{cid}").json()
    assert middle["origin"]["parent_execution_id"] == pid
    assert middle["handoff"]["child_execution_id"] == grandchild["execution_id"]


# ---------------------------------------------------------------------------
# Exports and off-site verification carry the lineage
# ---------------------------------------------------------------------------


def test_parent_export_remains_readable_and_carries_handoff():
    pid, cid = _parent_with_child()
    r = client.get(f"/plan-executions/{pid}/export")
    assert r.status_code == 200
    pkg = r.json()
    assert pkg["execution_id"] == pid
    assert pkg["version"] == 1
    assert pkg["handoff"]["child_execution_id"] == cid
    assert pkg["handoff"]["parent_version"] == 1
    assert pkg["origin"] is None
    # full committed history is still exported
    assert [c["version"] for c in pkg["commits"]] == [0, 1]


def test_child_export_carries_origin_and_verifies_off_site():
    pid, cid = _parent_with_child()
    pkg = client.get(f"/plan-executions/{cid}/export").json()
    assert pkg["origin"] == {
        "kind": "handoff",
        "parent_execution_id": pid,
        "parent_version": 1,
        "handed_off_at": pkg["origin"]["handed_off_at"],
    }
    assert pkg["handoff"] is None
    # An unrelated verifier replays the package using only its own contents:
    # the version-0 snapshot is the forked parent snapshot and replays from
    # the child scenario's declared initial state.
    r = client.post("/execution-verifications", json=pkg)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["verified_version"] == 0
    assert body["next_stage"] == 1
    assert body["finished"] is False
    assert body["origin"]["parent_execution_id"] == pid
    assert body["handoff"] is None


def test_verification_reports_child_whose_snapshot_was_tampered():
    pid, cid = _parent_with_child()
    pkg = client.get(f"/plan-executions/{cid}/export").json()
    # Tamper the version-0 anchor set in the exported snapshot only.
    pkg["commits"][0]["snapshot"]["trust_anchors"] = [
        {"client_id": "c", "anchor_cert_id": "root-new"}
    ]
    r = client.post("/execution-verifications", json=pkg)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "conflict"
    assert body["mismatch"]["reason"] == "snapshot_mismatch"
    assert body["mismatch"]["component"] == "trust_anchors"
