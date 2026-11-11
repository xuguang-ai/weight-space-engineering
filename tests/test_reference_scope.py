"""Vector references resolve against the experiment's own stored results, whatever engine runs the operation.

An analysis or a preview of experiment B may reference only results that B
produced (``wse://analyses/<sha256>#<pointer>``). A result of experiment A
fails exactly like a result that does not exist, also through an operator
(unbound) engine, so that A's vector can never reach B's results or patches,
which B's run can read. Models are the tiny offline fixtures of
``tests/hf_fixtures.py``.
"""

from __future__ import annotations

import pytest

from conftest import HF_SKIP_REASON
from wse.api import Engine
from wse.schemas import ErrorCode, WSEError

pytest.importorskip("transformers", reason=HF_SKIP_REASON)

REPO = "example-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DOWN = "model.layers.0.mlp.down_proj.weight"  # [16, 32]
FULL = [{"start": 0, "stop": 16, "step": 1}, {"start": 0, "stop": 32, "step": 1}]


def _expect(code: ErrorCode, func, *args, **kwargs) -> WSEError:
    with pytest.raises(WSEError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code, excinfo.value
    return excinfo.value


def _state(engine: Engine, experiment_id: str) -> tuple:
    """What a run could see of its experiment: consumed allowance, owned results, and patches."""
    budget = {name: (entry.consumed, entry.reserved)
              for name, entry in engine.get_experiment(experiment_id).budget.items() if name != "compute_seconds"}
    conn = engine.store.db.conn
    owned = [row["uri"] for row in conn.execute(
        "SELECT uri FROM analysis_results WHERE experiment_id = ? ORDER BY uri", (experiment_id,))]
    patches = [row["patch_id"] for row in conn.execute(
        "SELECT patch_id FROM patches WHERE experiment_id = ? ORDER BY patch_id", (experiment_id,))]
    return budget, owned, patches


def test_references_resolve_against_the_experiments_own_results_bound_or_not(hf_snapshots, workspace):
    from wse.policy import QWEN_LOOP_POLICY_ID

    with Engine(workspace) as engine:  # an operator engine, bound to no run
        base = engine.import_hf_snapshot(hf_snapshots["llama_untied"], repo=REPO, revision=REVISION).checkpoint_id
        grants = [engine.grant_run("scope_v1", QWEN_LOOP_POLICY_ID, base) for _ in range(2)]
        a, b = (engine.create_experiment(grant.run_id, "scope_v1", QWEN_LOOP_POLICY_ID, "create").experiment_id
                for grant in grants)
        # Per-row norms (16 values, the output axis) and per-column norms (32 values, the input axis) of A.
        rows = engine.analyze_model(a, base, "tensor_norms", DOWN, FULL, parameters={"axis": "rows"}).result_uri
        columns = engine.analyze_model(a, base, "tensor_norms", DOWN, FULL, parameters={"axis": "columns"}).result_uri
        projection = {"direction": f"{columns}#/norms", "axis": "rows"}

        def spec(direction: str) -> dict:
            return {"base_checkpoint_id": base, "hypothesis": "Removing this direction changes the outputs.",
                    "operations": [{"op": "project_out", "tensor": DOWN, "selection": FULL, "direction": direction,
                                    "columns": [0, 1]}]}

        # A's references resolve in A, through this engine as through A's run.
        assert engine.analyze_model(a, base, "direction_projection", DOWN, FULL, parameters=projection).usage \
            .analysis_calls == 1
        assert engine.preview_patch(a, spec(f"{rows}#/norms"), "a-preview").changed_elements > 0
        # In B they fail exactly like a result that does not exist, and B gets nothing: no charge, no stored result
        # or patch that B's run could read.
        before = _state(engine, b)
        missing = "wse://analyses/" + "0" * 64
        absent = _expect(ErrorCode.NOT_FOUND, engine.analyze_model, b, base, "direction_projection", DOWN, FULL,
                         parameters={**projection, "direction": f"{missing}#/norms"}).message
        error = _expect(ErrorCode.NOT_FOUND, engine.analyze_model, b, base, "direction_projection", DOWN, FULL,
                        parameters=projection)
        assert error.message.replace(columns, missing) == absent
        _expect(ErrorCode.NOT_FOUND, engine.preview_patch, b, spec(f"{rows}#/norms"), "b-preview")
        assert _state(engine, b) == before
    with Engine(workspace, run_id=grants[1].run_id) as bound:
        _expect(ErrorCode.NOT_FOUND, bound.read_analysis_result, columns)
        _expect(ErrorCode.NOT_FOUND, bound.analyze_model, b, base, "direction_projection", DOWN, FULL,
                parameters=projection)
    with Engine(workspace, run_id=grants[0].run_id) as bound:
        assert bound.read_analysis_result(columns)["norms"]
