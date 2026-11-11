"""The README's agent example, ``examples/signed_sum/agent_api.py``, run offline with the scripted controller.

The example writes out the setup that ``build_fixture.setup_workspace``
performs for the trial harness and the tests. These tests run the example
and check that both set up the same model, task, development suite, final
suite configuration, and policy values, so the two cannot drift apart.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from wse.api import Engine
from wse.policy import SMALL_MODEL_POLICY_ID, builtin_experiment_policies

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "signed_sum" / "agent_api.py"
MODEL_FIELDS = ("weights_hash", "config_hash", "adapter", "config", "tensors", "parameter_count", "input_semantics",
                "output_semantics", "provenance", "license")


def _run_example(out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(EXAMPLE), str(out)], capture_output=True, text=True, timeout=600,
                          check=False)


@pytest.fixture(scope="module")
def example_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("example") / "demo"
    result = _run_example(out)
    assert result.returncode == 0, result.stderr
    return out, result.stdout


def test_the_example_repairs_the_model_and_reports_the_edit(example_run: tuple[Path, str]) -> None:
    _, stdout = example_run
    assert "scripted_search@1 ended stopped: target_met" in stdout
    assert "edit: scale_slice@1 on layers.1.weight[1:2, 0:4], factor -1.0" in stdout
    assert ("layers.1.weight: [[-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]] -> "
            "[[-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]]") in stdout
    assert "final test, 1,024 examples the agent never saw: 50% correct before the edit, 100% after" in stdout
    assert "the export replays to byte-identical weights: True" in stdout


def test_the_example_sets_up_what_the_fixture_builder_sets_up(example_run: tuple[Path, str], signed_sum,
                                                               tmp_path: Path) -> None:
    out, _ = example_run
    with Engine(out / "workspace") as example, Engine(tmp_path / "fixture") as fixture:
        ids = signed_sum.setup_workspace(fixture, tmp_path / "bundles", final_seed=1)
        imported = [m for m in example.list_models() if m.provenance.kind == "import"]
        assert len(imported) == 1
        reference = fixture.load_manifest(ids["task_checkpoint_id"])
        for field in MODEL_FIELDS:
            assert getattr(imported[0], field) == getattr(reference, field), field

        assert example.get_task(ids["task_id"]) == fixture.get_task(ids["task_id"])
        assert example.get_suite(ids["development_suite_id"]) == fixture.get_suite(ids["development_suite_id"])
        final, reference_final = example.get_suite(ids["final_suite_id"]), fixture.get_suite(ids["final_suite_id"])
        assert final.model_dump(exclude={"dataset"}) == reference_final.model_dump(exclude={"dataset"})
        assert (final.dataset.example_count, final.dataset.cluster_count) == (1024, 512)

        policy = example.get_policy("signed_sum_policy_v1").model_dump(mode="json")
        builtin = builtin_experiment_policies()[SMALL_MODEL_POLICY_ID].model_dump(mode="json")
        assert {**policy, "policy_id": SMALL_MODEL_POLICY_ID} == builtin


def test_the_example_refuses_a_directory_that_exists(example_run: tuple[Path, str]) -> None:
    out, _ = example_run
    result = _run_example(out)
    assert result.returncode == 2
    assert "exists; each run needs a new directory" in result.stderr
