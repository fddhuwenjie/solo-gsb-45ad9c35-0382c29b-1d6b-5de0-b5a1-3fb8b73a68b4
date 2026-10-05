"""Auditable, transactional stage commits for registered rotation plans.

Caller flow:

1. ``POST /plan-executions`` registers an **immutable** plan (normally the
   output of ``/plans``) together with its scenario. The service replays the
   whole plan once, stores the scenario/plan verbatim and saves the **version
   0** (initial-state) snapshot. The current version starts at 0.
2. ``POST /plan-executions/{id}/stages`` commits one stage's executed actions
   plus the field observations (client trust anchors and endpoint chains)
   attesting them. The commit is validated **inside one SQLite transaction**:
   the action set must equal exactly the plan stage, every observation must
   match the state the stage deterministically produces, and every required
   communication relationship must still have a valid trust path. Only when
   all checks pass does the version advance and the stage snapshot get saved.
   Nothing is ever left half-applied.

Optimistic concurrency: every commit carries ``expected_version``. A stale or
concurrent commit (its ``expected_version`` differs from the stored version) is
rejected with ``409 version_conflict`` without touching stored state. Re-
submitting an already-committed stage with the content-identical commit *and*
the version it was based on is idempotent: it returns the same stored snapshot
instead of failing; different content for the same stage always returns 409.

The plan and every snapshot are immutable; the only mutable fields of an
execution are its current version and the strictly appended snapshots.

Handover (接棒): when a rotation still in progress must switch plans,
``POST /plan-executions/{id}/handover`` marks the original execution
``handed_over`` **and** registers a successor at version 0 in one transaction.
The successor's version-0 snapshot is the original execution's latest snapshot,
so the progress already on the books is its starting point; both effects land
atomically. A handover whose ``expected_version`` is stale (the original
execution committed another stage first) or that arrives second (a successor
already exists) is rejected without creating any child. Afterwards the old
execution cannot commit an unfinished stage, while an already-committed stage's
identical retry stays idempotent; both executions stay readable/exportable and
carry predecessor/successor lineage.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .engine import (
    Scenario,
    ScenarioValidationError,
    TrustEdge,
    TrustState,
    build_scenario,
    check_batch,
    evaluate_path,
    first_broken_relationship,
    initial_state,
    state_for_mask_applied,
)
from .models import (
    ChainCert,
    CommitAnchorObservation,
    CommitChainObservation,
    CommitConflict,
    ExecutionRegisterRequest,
    ExecutionStage,
    ExecutionStatus,
    Failure,
    HandoverConsistencyDetail,
    HandoverConflict,
    HandoverMismatchItem,
    HandoverRequest,
    PlannedStage,
    PlanRequest,
    RelationshipRef,
    StageCommitRequest,
    StageSnapshot,
    SnapshotEndpointChain,
    SnapshotRelationship,
    SnapshotTrustAnchor,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS plan_executions (
    execution_id    TEXT PRIMARY KEY,
    scenario_json   TEXT NOT NULL,
    plan_json       TEXT NOT NULL,
    stage_count     INTEGER NOT NULL,
    current_version INTEGER NOT NULL,
    lifecycle       TEXT NOT NULL DEFAULT 'active',
    successor_id    TEXT,
    predecessor_id  TEXT,
    predecessor_version INTEGER,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_commits (
    execution_id  TEXT NOT NULL,
    version       INTEGER NOT NULL,
    stage         INTEGER NOT NULL,
    payload_json  TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    committed_at  TEXT NOT NULL,
    PRIMARY KEY (execution_id, version),
    UNIQUE (execution_id, stage),
    FOREIGN KEY (execution_id) REFERENCES plan_executions(execution_id)
);
"""

# Columns added after the first archive release; applied idempotently to SQLite
# files created by the older schema so existing stores migrate transparently.
_ADDED_COLUMNS = (
    ("lifecycle", "TEXT NOT NULL DEFAULT 'active'"),
    ("successor_id", "TEXT"),
    ("predecessor_id", "TEXT"),
    ("predecessor_version", "INTEGER"),
)


class ExecutionReplayError(Exception):
    """The registered/committed plan replay breaks a relationship (HTTP 409)."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class CommitRejected(Exception):
    """A stage commit failed its transactional validation (HTTP 409)."""

    def __init__(self, conflict: CommitConflict) -> None:
        super().__init__(conflict.message)
        self.conflict = conflict


class UnknownExecution(Exception):
    """No registered execution with that id (HTTP 404)."""

    def __init__(self, execution_id: str) -> None:
        super().__init__(f"unknown execution_id '{execution_id}'")
        self.execution_id = execution_id


class HandoverRejected(Exception):
    """A handover failed one of its transactional guards (HTTP 409).

    The rejected handover never inserts a successor row: there is no child
    record to clean up.
    """

    def __init__(self, conflict: "HandoverConflict") -> None:  # type: ignore[name-defined]
        super().__init__(conflict.message)
        self.conflict = conflict


class HandoverConsistencyError(Exception):
    """The new scenario does not start at the original snapshot (HTTP 422)."""

    def __init__(self, detail: "HandoverConsistencyDetail") -> None:  # type: ignore[name-defined]
        super().__init__(detail.message)
        self.detail = detail


# ---------------------------------------------------------------------------
# SQLite plumbing
# ---------------------------------------------------------------------------


def db_path() -> str:
    return os.environ.get("ROTATION_DB_PATH", "/data/rotation.db")


_initialized: set[str] = set()
_init_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    path = db_path()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # Writers take the write lock up front so concurrent commits serialize;
    # busy_timeout makes lock contention wait instead of erroring immediately.
    conn.execute(f"PRAGMA busy_timeout = {10_000}")
    conn.execute("PRAGMA foreign_keys = ON")
    if path not in _initialized:
        with _init_lock:
            if path not in _initialized:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(SCHEMA)
                existing = {
                    r["name"]
                    for r in conn.execute(
                        "PRAGMA table_info(plan_executions)"
                    ).fetchall()
                }
                for column, decl in _ADDED_COLUMNS:
                    if column not in existing:
                        conn.execute(
                            f"ALTER TABLE plan_executions ADD COLUMN {column} {decl}"
                        )
                _initialized.add(path)
    return conn


def reset_store() -> None:
    """Forget cached schema state (tests point ROTATION_DB_PATH at a fresh dir)."""

    _initialized.clear()


# ---------------------------------------------------------------------------
# Plan structural validation + full replay (shared registration semantics)
# ---------------------------------------------------------------------------


def _structural_batches(
    sc: Scenario, stages: list[ExecutionStage]
) -> list[list[int]]:
    """Translate plan stages to action-index batches exactly like /plans/verify."""

    index = {aid: i for i, aid in enumerate(sc.action_ids)}
    seen: set[str] = set()
    expected_stage = 1
    batches: list[list[int]] = []
    for st in stages:
        if st.stage != expected_stage:
            raise ScenarioValidationError(
                f"plan: stages must be numbered consecutively from 1; "
                f"expected stage {expected_stage}, got {st.stage}"
            )
        expected_stage += 1
        batch: list[int] = []
        for aid in st.action_ids:
            if aid not in index:
                raise ScenarioValidationError(
                    f"plan: unknown action_id '{aid}'"
                )
            if aid in seen:
                raise ScenarioValidationError(
                    f"plan: action '{aid}' appears more than once"
                )
            seen.add(aid)
            batch.append(index[aid])
        batches.append(batch)
    missing = sorted(set(sc.action_ids) - seen)
    if missing:
        raise ScenarioValidationError(
            "plan: every action must be scheduled exactly once; missing: "
            + ", ".join(missing)
        )
    return batches


def _replay_message(sc: Scenario, rf: Any) -> str:
    rel = sc.relationships[rf.relationship_index]
    who = f"relationship {rel.client_id} -> {rel.endpoint_id}"
    prefix = "initial state already invalid" if rf.stage == 0 else f"stage {rf.stage}"
    if rf.reason == "untrusted_root":
        broken = rf.chain[rf.failure_index]
        return (
            f"{prefix}: {who} loses its trust path; chain terminates at "
            f"'{broken}', which is not among the client's trust anchors "
            f"({', '.join(rf.trusted_anchors) or 'none'})"
        )
    if rf.reason == "issuer_mismatch":
        broken = rf.chain[rf.failure_index]
        nxt = rf.chain[rf.failure_index + 1]
        return (
            f"{prefix}: {who} chain breaks at '{broken}' -> '{nxt}': "
            f"issuer/subject mismatch"
        )
    if rf.reason == "action_not_allowed_at_stage":
        return (
            f"{prefix}: action is scheduled outside its allowed_stages; "
            f"{who} cannot be verified"
        )
    if rf.reason == "contradictory_actions_same_stage":
        return (
            f"{prefix}: an add_trust and a remove_trust on the same anchor "
            f"are scheduled together; {who} cannot be verified"
        )
    return f"{prefix}: {who} invalid ({rf.reason})"


def _replay_failure_dto(sc: Scenario, rf: Any, *, status: str) -> Failure:
    rel = sc.relationships[rf.relationship_index]
    return Failure(
        status=status,  # type: ignore[arg-type]
        stage=rf.stage,
        relationship=RelationshipRef(
            client_id=rel.client_id, endpoint_id=rel.endpoint_id
        ),
        blocking_action=(
            sc.action_ids[rf.blocking_action_index]
            if rf.blocking_action_index is not None
            else None
        ),
        reason=rf.reason,
        chain=[
            ChainCert(
                cert_id=cid,
                subject=sc.certs[cid].subject,
                issuer=sc.certs[cid].issuer,
            )
            for cid in rf.chain
        ],
        trusted_anchors=rf.trusted_anchors,
        message=_replay_message(sc, rf),
    )


def replay_plan(sc: Scenario, batches: list[list[int]]) -> None:
    """Replay the whole plan; raise ExecutionReplayError at the first break."""

    state = initial_state(sc)
    broken = first_broken_relationship(sc, state)
    if broken is not None:
        from .engine import ReplayFailure

        ridx, pf = broken
        rf = ReplayFailure(
            stage=0,
            relationship_index=ridx,
            blocking_action_index=None,
            reason=pf.reason,
            chain=pf.chain,
            failure_index=pf.failure_index,
            trusted_anchors=pf.trusted_anchors,
        )
        raise ExecutionReplayError(_replay_failure_dto(sc, rf, status="invalid"))

    for num, batch in enumerate(batches, start=1):
        bits = 0
        for i in batch:
            bits |= 1 << i
        rf = check_batch(sc, state, bits, num)
        if rf is not None:
            raise ExecutionReplayError(
                _replay_failure_dto(sc, rf, status="invalid")
            )
        state = state_for_mask_applied(sc, state, bits)


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------


def _build_snapshot(
    sc: Scenario,
    state: TrustState,
    *,
    version: int,
    stage: int,
    action_ids: list[str],
    anchor_observations: list[CommitAnchorObservation],
    chain_observations: list[CommitChainObservation],
    committed_at: datetime,
) -> StageSnapshot:
    anchor_snap = sorted(
        (
            SnapshotTrustAnchor(client_id=edge.client_id, anchor_cert_id=edge.cert_id)
            for edge in state.anchors
        ),
        key=lambda a: (a.client_id, a.anchor_cert_id),
    )
    chain_snap = [
        SnapshotEndpointChain(endpoint_id=ep, chain=list(state.chains[ep]))
        for ep in sorted(state.chains)
    ]
    rel_snaps: list[SnapshotRelationship] = []
    for rel in sc.relationships:
        failure = evaluate_path(sc, state, rel.client_id, rel.endpoint_id)
        chain_ids = list(state.chains[rel.endpoint_id])
        rel_snaps.append(
            SnapshotRelationship(
                client_id=rel.client_id,
                endpoint_id=rel.endpoint_id,
                valid=failure is None,
                chain=[
                    ChainCert(
                        cert_id=cid,
                        subject=sc.certs[cid].subject,
                        issuer=sc.certs[cid].issuer,
                    )
                    for cid in chain_ids
                ],
                trusted_anchors=sorted(
                    edge.cert_id
                    for edge in state.anchors
                    if edge.client_id == rel.client_id
                ),
            )
        )
    return StageSnapshot(
        version=version,
        stage=stage,
        action_ids=list(action_ids),
        anchor_observations=sorted(
            anchor_observations,
            key=lambda o: (o.client_id, o.anchor_cert_id),
        ),
        chain_observations=sorted(
            chain_observations, key=lambda o: o.endpoint_id
        ),
        trust_anchors=anchor_snap,
        endpoint_chains=chain_snap,
        relationships=rel_snaps,
        committed_at=committed_at,
    )


# ---------------------------------------------------------------------------
# Status assembly
# ---------------------------------------------------------------------------


def _status_from_row(conn: sqlite3.Connection, row: sqlite3.Row) -> ExecutionStatus:
    cur = conn.execute(
        "SELECT snapshot_json FROM stage_commits WHERE execution_id = ? "
        "ORDER BY version ASC",
        (row["execution_id"],),
    )
    snapshots = [
        StageSnapshot.model_validate_json(blob) for (blob,) in cur.fetchall()
    ]
    version = row["current_version"]
    stage_count = row["stage_count"]
    completed = [s.stage for s in snapshots if s.stage >= 1]
    lifecycle = row["lifecycle"]
    handed_over = lifecycle == "handed_over"
    finished = version >= stage_count
    # A handed-over execution cannot commit further stages even when its own
    # plan was not finished: next_stage is null for both terminal states.
    next_stage = None if (finished or handed_over) else version + 1
    return ExecutionStatus(
        execution_id=row["execution_id"],
        lifecycle=lifecycle,  # type: ignore[arg-type]
        version=version,
        stage_count=stage_count,
        completed_stages=completed,
        next_stage=next_stage,
        finished=finished,
        successor_execution_id=row["successor_id"],
        predecessor_execution_id=row["predecessor_id"],
        predecessor_version=row["predecessor_version"],
        snapshots=snapshots,
    )


def get_execution(execution_id: str) -> ExecutionStatus:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM plan_executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None:
            raise UnknownExecution(execution_id)
        return _status_from_row(conn, row)
    finally:
        conn.close()


def export_execution(execution_id: str) -> "ExecutionExport":
    """Return one self-contained, single-version export of an execution.

    The scenario row and every committed version row (0..current_version) are
    read inside **one read transaction**, so a stage that lands while the
    package is being served cannot split it across versions: the package always
    reflects exactly one consistent ``current_version``. The package carries
    everything a second instance needs to replay the committed prefix without
    access to this SQLite file.
    """

    from .models import (
        ExecutionExport,
        ExportedStageCommit,
        PlannedStage,
        PlanRequest,
        StageSnapshot,
    )

    conn = _connect()
    try:
        # A deferred transaction is a read transaction here: in WAL mode its
        # snapshot is fixed at the first read and stays stable for both SELECTs.
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT * FROM plan_executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            raise UnknownExecution(execution_id)
        commit_rows = conn.execute(
            "SELECT version, stage, snapshot_json, committed_at "
            "FROM stage_commits WHERE execution_id = ? ORDER BY version ASC",
            (execution_id,),
        ).fetchall()
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()

    version = row["current_version"]
    stage_count = row["stage_count"]

    # Internal integrity: the write path guarantees this, but an export must
    # never hand out a torn package, so fail loudly rather than fabricate one.
    expected_versions = list(range(version + 1))
    found_versions = [r["version"] for r in commit_rows]
    if found_versions != expected_versions or any(
        r["stage"] != r["version"] for r in commit_rows
    ):
        raise RuntimeError(
            f"execution '{execution_id}' store is inconsistent: versions "
            f"{found_versions} do not form the required prefix 0..{version}"
        )

    scenario = PlanRequest.model_validate(json.loads(row["scenario_json"]))
    plan = [PlannedStage.model_validate(d) for d in json.loads(row["plan_json"])]

    commits = []
    for r in commit_rows:
        snap = StageSnapshot.model_validate_json(r["snapshot_json"])
        commits.append(
            ExportedStageCommit(
                version=r["version"],
                stage=r["stage"],
                action_ids=list(snap.action_ids),
                anchor_observations=list(snap.anchor_observations),
                chain_observations=list(snap.chain_observations),
                committed_at=datetime.fromisoformat(r["committed_at"]),
                snapshot=snap,
            )
        )

    finished = version >= stage_count
    handed_over = row["lifecycle"] == "handed_over"
    return ExecutionExport(
        execution_id=row["execution_id"],
        lifecycle=row["lifecycle"],  # type: ignore[arg-type]
        version=version,
        stage_count=stage_count,
        completed_stages=list(range(1, version + 1)),
        next_stage=None if (finished or handed_over) else version + 1,
        finished=finished,
        successor_execution_id=row["successor_id"],
        predecessor_execution_id=row["predecessor_id"],
        predecessor_version=row["predecessor_version"],
        created_at=datetime.fromisoformat(row["created_at"]),
        scenario=scenario,
        plan=plan,
        commits=commits,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_execution(req: ExecutionRegisterRequest) -> ExecutionStatus:
    sc = build_scenario(req)
    batches = _structural_batches(sc, req.plan)
    # A registered plan must be executable: replay it fully before storing
    # anything, so an invalid/unsafe plan is never made an immutable record.
    replay_plan(sc, batches)

    execution_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    stages_dto = [
        PlannedStage(
            stage=num, action_ids=[sc.action_ids[i] for i in batch]
        )
        for num, batch in enumerate(batches, start=1)
    ]
    plan_blob = json.dumps(
        [s.model_dump() for s in stages_dto], separators=(",", ":")
    )

    initial_snapshot = _build_snapshot(
        sc,
        initial_state(sc),
        version=0,
        stage=0,
        action_ids=[],
        anchor_observations=[],
        chain_observations=[],
        committed_at=now,
    )

    # Store the scenario verbatim (datetimes as ISO-8601 via mode="json");
    # the plan is stored separately and never changes after registration.
    scenario_blob = json.dumps(
        req.model_dump(mode="json", exclude={"plan"}),
        separators=(",", ":"),
    )

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO plan_executions "
            "(execution_id, scenario_json, plan_json, stage_count, "
            "current_version, lifecycle, successor_id, predecessor_id, "
            "predecessor_version, created_at) "
            "VALUES (?, ?, ?, ?, 0, 'active', NULL, NULL, NULL, ?)",
            (
                execution_id,
                scenario_blob,
                plan_blob,
                len(batches),
                now.isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO stage_commits "
            "(execution_id, version, stage, payload_json, snapshot_json, "
            "committed_at) VALUES (?, 0, 0, ?, ?, ?)",
            (
                execution_id,
                json.dumps({"kind": "initial"}, separators=(",", ":")),
                initial_snapshot.model_dump_json(),
                now.isoformat(),
            ),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return get_execution(execution_id)


# ---------------------------------------------------------------------------
# Stage commit
# ---------------------------------------------------------------------------


def _known_clients(sc: Scenario) -> set[str]:
    clients = {rel.client_id for rel in sc.relationships}
    clients.update(edge.client_id for edge in sc.initial_anchors)
    for edge in sc.add_edges.values():
        clients.add(edge.client_id)
    for edge in sc.remove_edges.values():
        clients.add(edge.client_id)
    return clients


def _load_execution(
    conn: sqlite3.Connection, execution_id: str
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM plan_executions WHERE execution_id = ?",
        (execution_id,),
    ).fetchone()
    if row is None:
        raise UnknownExecution(execution_id)
    return row


def _canonical_payload(req: StageCommitRequest) -> str:
    """Order-independent content identity of a commit, for idempotent replay."""

    payload = {
        "action_ids": sorted(req.action_ids),
        "anchor_observations": sorted(
            (
                [o.client_id, o.anchor_cert_id, o.present]
                for o in req.anchor_observations
            ),
            key=lambda t: (t[0], t[1]),
        ),
        "chain_observations": sorted(
            ([o.endpoint_id, list(o.chain)] for o in req.chain_observations),
            key=lambda t: t[0],
        ),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _conflict(
    reason: str,
    message: str,
    *,
    expected_version: Optional[int] = None,
    current_version: Optional[int] = None,
    stage: Optional[int] = None,
) -> CommitConflict:
    return CommitConflict(
        reason=reason,
        message=message,
        expected_version=expected_version,
        current_version=current_version,
        stage=stage,
    )


def commit_stage(
    execution_id: str, req: StageCommitRequest
) -> ExecutionStatus:
    # ---- load + scenario-level reference validation (outside the txn) ------
    read_conn = _connect()
    try:
        row = _load_execution(read_conn, execution_id)
        scenario_blob = row["scenario_json"]
        plan_blob = row["plan_json"]
    finally:
        read_conn.close()

    scenario_req = PlanRequest.model_validate(json.loads(scenario_blob))
    sc = build_scenario(scenario_req)

    plan_stages = [PlannedStage.model_validate(d) for d in json.loads(plan_blob)]
    batches = _structural_batches(sc, plan_stages)

    # Malformed references are client errors (422), not optimistic conflicts.
    index = {aid: i for i, aid in enumerate(sc.action_ids)}
    for aid in req.action_ids:
        if aid not in index:
            raise ScenarioValidationError(
                f"action_ids: unknown action_id '{aid}'"
            )
    clients = _known_clients(sc)
    for obs in req.anchor_observations:
        if obs.client_id not in clients:
            raise ScenarioValidationError(
                f"anchor_observations: unknown client_id '{obs.client_id}'"
            )
        if obs.anchor_cert_id not in sc.certs:
            raise ScenarioValidationError(
                f"anchor_observations: unknown certificate "
                f"'{obs.anchor_cert_id}'"
            )
    for obs in req.chain_observations:
        if obs.endpoint_id not in sc.endpoint_chains:
            raise ScenarioValidationError(
                f"chain_observations: unknown endpoint_id '{obs.endpoint_id}'"
            )
        for cid in obs.chain:
            if cid not in sc.certs:
                raise ScenarioValidationError(
                    f"chain_observations: unknown certificate '{cid}' in "
                    f"observed chain of endpoint '{obs.endpoint_id}'"
                )

    payload = _canonical_payload(req)

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _load_execution(conn, execution_id)
        current_version = row["current_version"]
        stage_count = row["stage_count"]
        stage = req.stage

        # -- optimistic lock first: stale or concurrent commits lose --------
        # This precedes idempotent replay: a duplicate is recognized only when
        # the caller still bases it on the version the stage advanced from.
        if req.expected_version != current_version:
            raise CommitRejected(
                _conflict(
                    "version_conflict",
                    f"expected_version {req.expected_version} does not match "
                    f"the current version {current_version}; a concurrent or "
                    f"stale commit must be retried with "
                    f"expected_version={current_version} against the latest "
                    f"state.",
                    expected_version=current_version,
                    current_version=current_version,
                    stage=stage,
                )
            )

        # -- already committed? idempotent replay or content conflict -------
        committed_row = conn.execute(
            "SELECT payload_json, snapshot_json FROM stage_commits "
            "WHERE execution_id = ? AND stage = ?",
            (execution_id, stage),
        ).fetchone()
        if committed_row is not None:
            if committed_row["payload_json"] == payload:
                conn.execute("ROLLBACK")
                return get_execution(execution_id)
            raise CommitRejected(
                _conflict(
                    "duplicate_payload_mismatch",
                    f"stage {stage} is already committed; re-submitting it "
                    f"with different content is forbidden (the audit record "
                    f"is immutable). Re-send the identical commit to receive "
                    f"the same snapshot.",
                    expected_version=current_version,
                    current_version=current_version,
                    stage=stage,
                )
            )

        # -- handed over? no unfinished stage may land on the old execution --
        # An already-committed stage is handled above (idempotent replay), so a
        # request reaching here is for a not-yet-committed stage, which must
        # fail now that a successor continues from this execution's snapshot.
        if row["lifecycle"] == "handed_over":
            raise CommitRejected(
                _conflict(
                    "execution_handed_over",
                    f"execution '{execution_id}' has been handed over to "
                    f"successor '{row['successor_id']}' at version "
                    f"{current_version}; commit the remaining stages against "
                    f"the successor instead.",
                    expected_version=current_version,
                    current_version=current_version,
                    stage=stage,
                )
            )

        # -- plan ordering: the next consecutive stage only -----------------
        if stage != current_version + 1 or stage > stage_count:
            raise CommitRejected(
                _conflict(
                    "unexpected_stage",
                    f"stage {stage} cannot be committed next: expected "
                    f"stage {current_version + 1} of {stage_count}.",
                    expected_version=current_version,
                    current_version=current_version,
                    stage=stage,
                )
            )

        # -- action set must be exactly this plan stage ---------------------
        planned_actions = [sc.action_ids[i] for i in batches[stage - 1]]
        if sorted(req.action_ids) != sorted(planned_actions):
            raise CommitRejected(
                _conflict(
                    "action_set_mismatch",
                    f"stage {stage} plan actions "
                    f"[{', '.join(sorted(planned_actions)) or 'none'}] do not "
                    f"match the committed action set "
                    f"[{', '.join(sorted(req.action_ids)) or 'none'}].",
                    expected_version=current_version,
                    current_version=current_version,
                    stage=stage,
                )
            )

        # -- replay the prefix to obtain the exact pre-stage state ----------
        # The immutable plan fixes prior stages; rebuild the state they
        # produced so the new stage is validated against the true prefix.
        pre_state = initial_state(sc)
        for num in range(1, stage):
            prior_bits = 0
            for i in batches[num - 1]:
                prior_bits |= 1 << i
            pre_state = state_for_mask_applied(sc, pre_state, prior_bits)

        bits = 0
        for aid in req.action_ids:
            bits |= 1 << index[aid]

        # -- batch validation: allowed_stages, contradictions, trust paths --
        rf = check_batch(sc, pre_state, bits, stage)
        if rf is not None:
            failure = _replay_failure_dto(sc, rf, status="invalid")
            raise CommitRejected(
                CommitConflict(
                    reason=rf.reason,
                    message=failure.message,
                    expected_version=current_version,
                    current_version=current_version,
                    stage=stage,
                    relationship=failure.relationship,
                    blocking_action=failure.blocking_action,
                    chain=failure.chain,
                    trusted_anchors=failure.trusted_anchors,
                )
            )

        end_state = state_for_mask_applied(sc, pre_state, bits)

        # -- observations must attest the computed field state --------------
        for obs in req.anchor_observations:
            edge = TrustEdge(obs.client_id, obs.anchor_cert_id)
            if (edge in end_state.anchors) != obs.present:
                expected_present = edge in end_state.anchors
                raise CommitRejected(
                    CommitConflict(
                        reason="observation_mismatch",
                        message=(
                            f"anchor observation ({obs.client_id}, "
                            f"{obs.anchor_cert_id}) present="
                            f"{str(obs.present).lower()} does not match the "
                            f"post-stage state (present="
                            f"{str(expected_present).lower()})."
                        ),
                        expected_version=current_version,
                        current_version=current_version,
                        stage=stage,
                        observation={
                            "kind": "anchor",
                            "client_id": obs.client_id,
                            "anchor_cert_id": obs.anchor_cert_id,
                            "submitted_present": obs.present,
                            "expected_present": expected_present,
                        },
                    )
                )
        for obs in req.chain_observations:
            expected_chain = list(end_state.chains[obs.endpoint_id])
            if list(obs.chain) != expected_chain:
                raise CommitRejected(
                    CommitConflict(
                        reason="observation_mismatch",
                        message=(
                            f"chain observation for endpoint "
                            f"'{obs.endpoint_id}' ['"
                            + "' -> '".join(obs.chain)
                            + "'] does not match the post-stage chain ['"
                            + "' -> '".join(expected_chain)
                            + "']."
                        ),
                        expected_version=current_version,
                        current_version=current_version,
                        stage=stage,
                        observation={
                            "kind": "endpoint_chain",
                            "endpoint_id": obs.endpoint_id,
                            "submitted_chain": list(obs.chain),
                            "expected_chain": expected_chain,
                        },
                    )
                )

        # Defense in depth: the registered plan was fully replayed at
        # registration, but every relationship is re-checked here inside the
        # transaction before any version advance.
        broken = first_broken_relationship(sc, end_state)
        if broken is not None:
            ridx, pf = broken
            rel = sc.relationships[ridx]
            raise CommitRejected(
                CommitConflict(
                    reason=pf.reason,
                    message=(
                        f"stage {stage}: relationship {rel.client_id} -> "
                        f"{rel.endpoint_id} has no valid trust path after "
                        f"the committed actions."
                    ),
                    expected_version=current_version,
                    current_version=current_version,
                    stage=stage,
                    relationship=RelationshipRef(
                        client_id=rel.client_id, endpoint_id=rel.endpoint_id
                    ),
                    chain=[
                        ChainCert(
                            cert_id=cid,
                            subject=sc.certs[cid].subject,
                            issuer=sc.certs[cid].issuer,
                        )
                        for cid in pf.chain
                    ],
                    trusted_anchors=pf.trusted_anchors,
                )
            )

        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(
            sc,
            end_state,
            version=stage,
            stage=stage,
            action_ids=[sc.action_ids[i] for i in batches[stage - 1]],
            anchor_observations=list(req.anchor_observations),
            chain_observations=list(req.chain_observations),
            committed_at=now,
        )

        conn.execute(
            "INSERT INTO stage_commits "
            "(execution_id, version, stage, payload_json, snapshot_json, "
            "committed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                execution_id,
                stage,
                stage,
                payload,
                snapshot.model_dump_json(),
                now.isoformat(),
            ),
        )
        cur = conn.execute(
            "UPDATE plan_executions SET current_version = ? "
            "WHERE execution_id = ? AND current_version = ?",
            (stage, execution_id, current_version),
        )
        # The conditional UPDATE is the final optimistic-lock guarantee: under
        # the write lock the row must still be at the version we validated.
        if cur.rowcount != 1:
            raise CommitRejected(
                _conflict(
                    "version_conflict",
                    f"execution '{execution_id}' advanced concurrently while "
                    f"stage {stage} was being committed; no progress was "
                    f"applied.",
                    expected_version=None,
                    current_version=None,
                    stage=stage,
                )
            )
        conn.execute("COMMIT")
    except CommitRejected:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()

    return get_execution(execution_id)


# ---------------------------------------------------------------------------
# Handover (接棒): continue an unfinished execution under a new plan
#
# When a rotation still in progress must switch to a new plan, the progress
# already on the books becomes the new execution's starting point. One SQLite
# transaction both flips the original execution to lifecycle 'handed_over' and
# registers the successor at version 0, whose initial snapshot is copied from
# the original execution's latest snapshot. Either both land or neither does;
# a losing concurrent handover fails without inserting any child row.
# ---------------------------------------------------------------------------


def _state_from_snapshot(snapshot: StageSnapshot) -> TrustState:
    """Rebuild the engine TrustState captured by a stored snapshot."""

    return TrustState(
        anchors=frozenset(
            TrustEdge(a.client_id, a.anchor_cert_id)
            for a in snapshot.trust_anchors
        ),
        chains={
            ep.endpoint_id: list(ep.chain) for ep in snapshot.endpoint_chains
        },
    )


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _cert_attribute_mismatches(
    parent_req: PlanRequest, child_req: HandoverRequest
) -> list[HandoverMismatchItem]:
    """A reused cert_id must carry identical attributes in the new scenario."""

    parent_certs = {c.cert_id: c for c in parent_req.certificates}
    mismatches: list[HandoverMismatchItem] = []
    for c in sorted(child_req.certificates, key=lambda x: x.cert_id):
        old = parent_certs.get(c.cert_id)
        if old is None:
            continue  # newly introduced certificate: nothing to rewrite against
        attrs: list[tuple[str, Any, Any]] = [
            ("subject", old.subject, c.subject),
            ("issuer", old.issuer, c.issuer),
            ("not_valid_before", _utc(old.not_valid_before),
             _utc(c.not_valid_before)),
            ("not_valid_after", _utc(old.not_valid_after),
             _utc(c.not_valid_after)),
        ]
        diverged = [name for name, want, got in attrs if want != got]
        if diverged:
            details = {
                name: [want.isoformat() if hasattr(want, "isoformat") else want,
                       got.isoformat() if hasattr(got, "isoformat") else got]
                for name, want, got in attrs if name in diverged
            }
            mismatches.append(
                HandoverMismatchItem(
                    component="certificate",
                    message=(
                        f"reused certificate '{c.cert_id}' must keep its "
                        f"original attributes; changed: "
                        f"{', '.join(diverged)}"
                    ),
                    cert_id=c.cert_id,
                    attribute=",".join(diverged),
                    expected={k: v[0] for k, v in details.items()},
                    found={k: v[1] for k, v in details.items()},
                )
            )
    return mismatches


def _starting_point_mismatches(
    parent_req: PlanRequest,
    child_req: HandoverRequest,
    snapshot: StageSnapshot,
) -> list[HandoverMismatchItem]:
    """Every way the new scenario fails to start exactly at ``snapshot``."""

    mismatches: list[HandoverMismatchItem] = []

    # Initial trust anchors must equal the snapshot's complete anchor set.
    snapshot_anchors = sorted(
        (a.client_id, a.anchor_cert_id) for a in snapshot.trust_anchors
    )
    child_anchors = sorted(
        (a.client_id, a.cert_id) for a in child_req.trust_anchors
    )
    if child_anchors != snapshot_anchors:
        only_snapshot = sorted(set(snapshot_anchors) - set(child_anchors))
        only_child = sorted(set(child_anchors) - set(snapshot_anchors))
        for client_id, anchor_cert_id in only_snapshot:
            mismatches.append(
                HandoverMismatchItem(
                    component="trust_anchors",
                    message=(
                        f"new scenario is missing the snapshot's trust anchor "
                        f"({client_id}, {anchor_cert_id})"
                    ),
                    client_id=client_id,
                    anchor_cert_id=anchor_cert_id,
                    expected={"present": True},
                    found={"present": False},
                )
            )
        for client_id, anchor_cert_id in only_child:
            mismatches.append(
                HandoverMismatchItem(
                    component="trust_anchors",
                    message=(
                        f"new scenario declares an initial trust anchor "
                        f"({client_id}, {anchor_cert_id}) absent from the "
                        f"snapshot"
                    ),
                    client_id=client_id,
                    anchor_cert_id=anchor_cert_id,
                    expected={"present": False},
                    found={"present": True},
                )
            )

    # Every endpoint's current chain must be the chain served in the snapshot.
    snapshot_chains = {
        ep.endpoint_id: list(ep.chain) for ep in snapshot.endpoint_chains
    }
    child_endpoints = {ep.endpoint_id: ep for ep in child_req.endpoints}
    for endpoint_id in sorted(set(snapshot_chains) | set(child_endpoints)):
        if endpoint_id not in child_endpoints:
            mismatches.append(
                HandoverMismatchItem(
                    component="endpoint_chains",
                    message=(
                        f"new scenario drops endpoint '{endpoint_id}' that "
                        f"serves a chain in the snapshot"
                    ),
                    endpoint_id=endpoint_id,
                    expected=snapshot_chains.get(endpoint_id),
                    found=None,
                )
            )
            continue
        if endpoint_id not in snapshot_chains:
            mismatches.append(
                HandoverMismatchItem(
                    component="endpoint_chains",
                    message=(
                        f"new scenario adds endpoint '{endpoint_id}' that is "
                        f"not present at the snapshot"
                    ),
                    endpoint_id=endpoint_id,
                    expected=None,
                    found=list(
                        child_endpoints[endpoint_id].current_chain
                    ),
                )
            )
            continue
        want = snapshot_chains[endpoint_id]
        got = list(child_endpoints[endpoint_id].current_chain)
        if got != want:
            mismatches.append(
                HandoverMismatchItem(
                    component="endpoint_chains",
                    message=(
                        f"endpoint '{endpoint_id}' current_chain must be the "
                        f"snapshot chain"
                    ),
                    endpoint_id=endpoint_id,
                    expected=want,
                    found=got,
                )
            )

    # The required communication relationships must be exactly the snapshot's.
    snapshot_rels = sorted(
        (r.client_id, r.endpoint_id) for r in snapshot.relationships
    )
    child_rels = sorted(
        (r.client_id, r.endpoint_id) for r in child_req.relationships
    )
    if child_rels != snapshot_rels:
        only_snapshot = sorted(set(snapshot_rels) - set(child_rels))
        only_child = sorted(set(child_rels) - set(snapshot_rels))
        for client_id, endpoint_id in only_snapshot:
            mismatches.append(
                HandoverMismatchItem(
                    component="relationships",
                    message=(
                        f"new scenario drops the required relationship "
                        f"{client_id} -> {endpoint_id} from the snapshot"
                    ),
                    client_id=client_id,
                    endpoint_id=endpoint_id,
                    expected={"required": True},
                    found={"required": False},
                )
            )
        for client_id, endpoint_id in only_child:
            mismatches.append(
                HandoverMismatchItem(
                    component="relationships",
                    message=(
                        f"new scenario adds a required relationship "
                        f"{client_id} -> {endpoint_id} that is not in the "
                        f"snapshot"
                    ),
                    client_id=client_id,
                    endpoint_id=endpoint_id,
                    expected={"required": False},
                    found={"required": True},
                )
            )

    mismatches.extend(_cert_attribute_mismatches(parent_req, child_req))

    # Stable ordering: component (anchors, chains, relationships, certs),
    # then the involved identities.
    component_order = {
        "trust_anchors": 0,
        "endpoint_chains": 1,
        "relationships": 2,
        "certificate": 3,
    }
    mismatches.sort(
        key=lambda m: (
            component_order[m.component],
            m.client_id or "",
            m.endpoint_id or "",
            m.anchor_cert_id or "",
            m.cert_id or "",
        )
    )
    return mismatches


def handover_execution(
    execution_id: str, req: HandoverRequest
) -> ExecutionStatus:
    """Atomically hand an unfinished execution over to a new continuation.

    Outside the transaction the new scenario/plan are parsed, structurally
    validated and fully replayed from the claimed starting state; inside one
    ``BEGIN IMMEDIATE`` transaction the optimistic version is re-checked, the
    original execution is marked ``handed_over`` and the version-0 successor is
    inserted. Both changes commit together or not at all.
    """

    # ---- load the original execution (outside the write txn) ---------------
    read_conn = _connect()
    try:
        row = _load_execution(read_conn, execution_id)
        parent_req = PlanRequest.model_validate(json.loads(row["scenario_json"]))
        snapshot_blob = read_conn.execute(
            "SELECT snapshot_json FROM stage_commits WHERE execution_id = ? "
            "AND version = ?",
            (execution_id, row["current_version"]),
        ).fetchone()[0]
        current_version = row["current_version"]
        stage_count = row["stage_count"]
        lifecycle = row["lifecycle"]
        successor = row["successor_id"]
    finally:
        read_conn.close()

    # ---- state guards before any payload content work ----------------------
    # These mirror the authoritative in-transaction checks below. Evaluating
    # them first means an impossible handover (the original execution moved on,
    # was already handed over, or already finished) always fails on that ground
    # regardless of what the new scenario contains — and creates no child.
    if req.expected_version != current_version:
        raise HandoverRejected(
            HandoverConflict(
                reason="version_conflict",
                message=(
                    f"expected_version {req.expected_version} does not match "
                    f"execution '{execution_id}' current version "
                    f"{current_version}; it committed another stage first. "
                    f"Base the handover on version {current_version} and its "
                    f"latest snapshot."
                ),
                execution_id=execution_id,
                expected_version=current_version,
                current_version=current_version,
            )
        )
    if lifecycle == "handed_over":
        raise HandoverRejected(
            HandoverConflict(
                reason="already_handed_over",
                message=(
                    f"execution '{execution_id}' was already handed over to "
                    f"successor '{successor}' at version {current_version}; "
                    f"a later handover creates no successor."
                ),
                execution_id=execution_id,
                expected_version=current_version,
                current_version=current_version,
                successor_execution_id=successor,
            )
        )
    if current_version >= stage_count:
        raise HandoverRejected(
            HandoverConflict(
                reason="execution_finished",
                message=(
                    f"execution '{execution_id}' already finished its plan at "
                    f"version {current_version}; nothing remains to hand over."
                ),
                execution_id=execution_id,
                expected_version=current_version,
                current_version=current_version,
            )
        )

    latest_snapshot = StageSnapshot.model_validate_json(snapshot_blob)

    # ---- parse + fully validate the new scenario ---------------------------
    child_sc = build_scenario(req)
    child_batches = _structural_batches(child_sc, req.plan)

    # The continuation must start exactly at the progress already on the books.
    mismatches = _starting_point_mismatches(parent_req, req, latest_snapshot)
    if mismatches:
        raise HandoverConsistencyError(
            HandoverConsistencyDetail(
                message=(
                    f"handover starting point does not match execution "
                    f"'{execution_id}' version {req.expected_version} latest "
                    f"snapshot: {len(mismatches)} divergence(s)"
                ),
                execution_id=execution_id,
                expected_version=req.expected_version,
                current_version=req.expected_version,
                mismatches=mismatches,
            )
        )

    # Replay the continuation plan from the snapshot state itself. The new
    # scenario's declared initial state was just proven equal to it, so a plan
    # that breaks a relationship is rejected here before anything is stored.
    start_state = _state_from_snapshot(latest_snapshot)
    broken = first_broken_relationship(child_sc, start_state)
    if broken is not None:
        from .engine import ReplayFailure

        ridx, pf = broken
        rf = ReplayFailure(
            stage=0,
            relationship_index=ridx,
            blocking_action_index=None,
            reason=pf.reason,
            chain=pf.chain,
            failure_index=pf.failure_index,
            trusted_anchors=pf.trusted_anchors,
        )
        raise ExecutionReplayError(
            _replay_failure_dto(child_sc, rf, status="invalid")
        )
    for num, batch in enumerate(child_batches, start=1):
        bits = 0
        for i in batch:
            bits |= 1 << i
        rf = check_batch(child_sc, start_state, bits, num)
        if rf is not None:
            raise ExecutionReplayError(
                _replay_failure_dto(child_sc, rf, status="invalid")
            )
        start_state = state_for_mask_applied(child_sc, start_state, bits)

    successor_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    stages_dto = [
        PlannedStage(
            stage=num, action_ids=[child_sc.action_ids[i] for i in batch]
        )
        for num, batch in enumerate(child_batches, start=1)
    ]
    plan_blob = json.dumps(
        [s.model_dump() for s in stages_dto], separators=(",", ":")
    )
    scenario_blob = json.dumps(
        req.model_dump(mode="json", exclude={"plan", "expected_version"}),
        separators=(",", ":"),
    )

    # The successor's version-0 snapshot continues the original snapshot:
    # same anchors, chains and relationship status, rebuilt with the new
    # scenario's certificate metadata.
    initial_snapshot = _build_snapshot(
        child_sc,
        _state_from_snapshot(latest_snapshot),
        version=0,
        stage=0,
        action_ids=[],
        anchor_observations=[],
        chain_observations=[],
        committed_at=now,
    )

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _load_execution(conn, execution_id)
        current_version = row["current_version"]
        stage_count = row["stage_count"]

        # Optimistic lock first: the original execution committing its next
        # stage first advances the version and loses this handover.
        if req.expected_version != current_version:
            raise HandoverRejected(
                HandoverConflict(
                    reason="version_conflict",
                    message=(
                        f"expected_version {req.expected_version} does not "
                        f"match execution '{execution_id}' current version "
                        f"{current_version}; it committed another stage first. "
                        f"Base the handover on version {current_version} and "
                        f"its latest snapshot."
                    ),
                    execution_id=execution_id,
                    expected_version=current_version,
                    current_version=current_version,
                )
            )

        # Only one successor may ever take over: a handover that lands second
        # fails here and inserts no child row.
        if row["lifecycle"] == "handed_over":
            raise HandoverRejected(
                HandoverConflict(
                    reason="already_handed_over",
                    message=(
                        f"execution '{execution_id}' was already handed over "
                        f"to successor '{row['successor_id']}' at version "
                        f"{current_version}; a later handover creates no "
                        f"successor."
                    ),
                    execution_id=execution_id,
                    expected_version=current_version,
                    current_version=current_version,
                    successor_execution_id=row["successor_id"],
                )
            )

        # A finished rotation has no unfinished stage to continue from.
        if current_version >= stage_count:
            raise HandoverRejected(
                HandoverConflict(
                    reason="execution_finished",
                    message=(
                        f"execution '{execution_id}' already finished its "
                        f"plan at version {current_version}; nothing remains "
                        f"to hand over."
                    ),
                    execution_id=execution_id,
                    expected_version=current_version,
                    current_version=current_version,
                )
            )

        # Re-derive the snapshot under the lock so the claim is checked
        # against the exact snapshot the version points to (defense in depth:
        # the optimistic guard above already pins current_version).
        locked_blob = conn.execute(
            "SELECT snapshot_json FROM stage_commits WHERE execution_id = ? "
            "AND version = ?",
            (execution_id, current_version),
        ).fetchone()[0]
        locked_snapshot = StageSnapshot.model_validate_json(locked_blob)
        locked_mismatches = _starting_point_mismatches(
            parent_req, req, locked_snapshot
        )
        if locked_mismatches:
            raise HandoverConsistencyError(
                HandoverConsistencyDetail(
                    message=(
                        f"handover starting point does not match execution "
                        f"'{execution_id}' version {current_version} latest "
                        f"snapshot under the write lock"
                    ),
                    execution_id=execution_id,
                    expected_version=req.expected_version,
                    current_version=current_version,
                    mismatches=locked_mismatches,
                )
            )

        # ---- the two effects of a handover land in this one transaction ----
        conn.execute(
            "INSERT INTO plan_executions "
            "(execution_id, scenario_json, plan_json, stage_count, "
            "current_version, lifecycle, successor_id, predecessor_id, "
            "predecessor_version, created_at) "
            "VALUES (?, ?, ?, ?, 0, 'active', NULL, ?, ?, ?)",
            (
                successor_id,
                scenario_blob,
                plan_blob,
                len(child_batches),
                execution_id,
                current_version,
                now.isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO stage_commits "
            "(execution_id, version, stage, payload_json, snapshot_json, "
            "committed_at) VALUES (?, 0, 0, ?, ?, ?)",
            (
                successor_id,
                json.dumps(
                    {
                        "kind": "handover_initial",
                        "predecessor_execution_id": execution_id,
                        "predecessor_version": current_version,
                    },
                    separators=(",", ":"),
                ),
                initial_snapshot.model_dump_json(),
                now.isoformat(),
            ),
        )
        cur = conn.execute(
            "UPDATE plan_executions SET lifecycle = 'handed_over', "
            "successor_id = ? WHERE execution_id = ? AND current_version = ? "
            "AND lifecycle = 'active'",
            (successor_id, execution_id, current_version),
        )
        # Final guard: the version/lifecycle state the checks ran against must
        # still hold; otherwise the whole handover rolls back atomically.
        if cur.rowcount != 1:
            raise HandoverRejected(
                HandoverConflict(
                    reason="already_handed_over",
                    message=(
                        f"execution '{execution_id}' changed while the "
                        f"handover was being committed; no successor was "
                        f"created."
                    ),
                    execution_id=execution_id,
                )
            )
        conn.execute("COMMIT")
    except HandoverRejected:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    except HandoverConsistencyError:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()

    return get_execution(successor_id)
