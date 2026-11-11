"""Task 12's expressiveness check: the known repair, expressed through the managed tools (operator use only).

The M2 plan's Q2 exit asks whether the tools can express the article's repair:
on a fixed practice subset, a ``project_out`` of p50c's recovered loop
direction from p50c's 64 changed columns of ``mlp.down`` in layers 15 and 19,
with alpha 6, must lower the development limit-hit rate relative to the base.
The direction of each layer is the top left singular vector of ``ΔW[:, C]``
(``C`` the changed columns of p50c's published diff), sign-fixed so that its
entry of largest magnitude is positive. It is an operator-only artifact: it
reaches the run only as this script's inline patch values, never through an
agent-visible surface.

A scripted client (the registered strategy ``expressiveness_check@1``
defined here) drives the managed tools of the run-bound engine through the
reference runner, which starts one worker on ``--device`` and settles and
freezes the experiment however the run ends:

1. a generation pass of the base on the practice suite (``wse_submit_evaluation``
   and ``wse_get_job``);
2. one patch of two ``project_out`` operations, each with its layer's
   direction inline (1,024 values), its 64 columns, and ``--alpha``
   (``wse_preview_patch``), applied as a candidate (``wse_apply_patch``);
3. a generation pass of the candidate on the same problems, and a paired
   comparison with the base (``wse_compare``).

The directions come from ``--directions NPZ`` (keys ``layer15``, ``layer19``,
``columns15``, ``columns19``), or are recomputed from the base and p50c
checkpoints of the workspace with ``--recompute``; p50c is imported first from
its published edit directory (read only; ``--p50c-edit``), which returns the
existing candidate when it is already there. With both, the two are compared
and the recomputed directions are used. The directions used are written to
``OUT/directions.npz``.

Output: ``OUT/expressiveness-<run_id>.json`` (and a copy as
``OUT/expressiveness.json``): the setup, the directions' provenance, the
verdict (the limit-hit rates of base and candidate and whether it fell), and
the experiment's full summary (``loop_setup.experiment_summary``). The
workspace setup is idempotent, so the script can be restarted; each
invocation grants a new run.

    uv run --no-sync python examples/qwen_loops/check_expressiveness.py OUT \\
        --workspace ~/Desktop/wse-models/workspaces/q2 --recompute

A short smoke run: ``--problems 4 --max-new-tokens 256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loop_setup  # noqa: E402 - the shared setup next to this script

LAYERS = (15, 19)
STRATEGY = "expressiveness_check@1"
DEFAULT_P50C_EDIT = loop_setup.MODELS / "lm_understand-logs" / "exp8_math" / "runs" / "p50c" / "edit"
DEFAULT_P50C_REPO = "RikiyaT/lm_understand-logs"
DEFAULT_P50C_REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"


# ---------------------------------------------------------------------------
# The scripted client
# ---------------------------------------------------------------------------


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ExpressivenessCheck:
    """Base pass, one patch of the given ``project_out`` operations, candidate pass, comparison; then stop.

    ``operations`` are complete ``project_out`` operations (tensor, selection,
    inline direction, columns, alpha). The selection stays on the base.
    """

    strategy_id = "expressiveness_check"
    version = "1"

    def __init__(self, suite_id: str, operations: Sequence[Mapping[str, Any]], wait_seconds: float = 10.0) -> None:
        self.suite_id = suite_id
        self.operations = [dict(operation) for operation in operations]
        self.wait_seconds = float(wait_seconds)
        self._plan: Generator[Any, Any, None] | None = None
        self._seen = 0
        self._step = 0

    def config(self) -> dict[str, Any]:
        def recorded(direction: list[float]) -> dict[str, Any]:
            return {"length": len(direction),
                    "sha256": hashlib.sha256(json.dumps(direction).encode("utf-8")).hexdigest()}

        return {"suite_id": self.suite_id, "wait_seconds": self.wait_seconds,
                "operations": [{**operation, "direction": recorded(operation["direction"])}
                               for operation in self.operations]}

    def next_action(self, objective: Any, observations: Sequence[Any], remaining_budget: Mapping[str, Any]) -> Any:
        from wse.strategies.base import Action

        try:
            if self._plan is None:
                self._seen = len(observations)
                self._plan = self._run(observations[0])
                return next(self._plan)
            new = list(observations[self._seen:])
            self._seen = len(observations)
            return self._plan.send(new[-1])
        except _Stop as stop:
            return Action.stop(stop.reason[:200])
        except StopIteration:
            return Action.stop("the check is complete")

    def _key(self, label: str) -> str:
        self._step += 1
        return f"expressiveness-{self._step:03d}-{label}"

    @staticmethod
    def _ok(observation: Any, what: str) -> Any:
        if not observation.ok:
            error = observation.error or {}
            raise _Stop(f"{what} failed with {error.get('code')}: {error.get('message')}")
        return observation.result

    def _pass(self, eid: str, checkpoint_id: str, config: dict[str, int], what: str) -> Generator[Any, Any, dict]:
        from wse.strategies.base import Action

        job = self._ok((yield Action.call("wse_submit_evaluation", f"Generate on the practice problems with {what}.",
                                          experiment_id=eid, checkpoint_id=checkpoint_id, suite_id=self.suite_id,
                                          eval_config=config, idempotency_key=self._key("pass"))), what)
        while job["state"] in ("queued", "running"):
            job = self._ok((yield Action.call("wse_get_job", f"Wait for the pass of {what}.", experiment_id=eid,
                                              job_id=job["job_id"], wait_seconds=self.wait_seconds)), "a poll")
        if job["state"] != "succeeded":
            raise _Stop(f"the pass of {what} ended {job['state']}: {job.get('error')}")
        return job

    def _run(self, first: Any) -> Generator[Any, Any, None]:
        from wse.strategies.base import Action

        description = self._ok(first, "reading the experiment")
        experiment = description["experiment"]
        eid, base = experiment["experiment_id"], experiment["base_checkpoint_id"]
        suite = next(s for s in description["suites"] if s["suite_id"] == self.suite_id)
        config = {"sample_count": max(suite["sample_counts"]), "seed": suite["seeds"][0]}
        base_job = yield from self._pass(eid, base, config, "the base")
        layers = ", ".join(operation["tensor"].split(".layers.")[1].split(".")[0] for operation in self.operations)
        hypothesis = (f"Expressiveness check (operator): removing a known loop direction from {len(self.operations)} "
                      f"mlp.down tensors (layers {layers}) lowers the development limit-hit rate.")
        preview = self._ok((yield Action.call(
            "wse_preview_patch", hypothesis, experiment_id=eid, idempotency_key=self._key("preview"),
            patch_spec={"base_checkpoint_id": base, "hypothesis": hypothesis,
                        "expected_behavior": "The development limit-hit rate falls relative to the base.",
                        "development_suite_id": self.suite_id, "operations": self.operations})), "the preview")
        candidate = self._ok((yield Action.call(
            "wse_apply_patch", f"Commit the previewed candidate ({preview['changed_elements']} changed weights).",
            experiment_id=eid, patch_id=preview["patch_id"], idempotency_key=self._key("apply"))), "the commit")
        candidate_job = yield from self._pass(eid, candidate["checkpoint_id"], config, "the candidate")
        yield Action.call("wse_compare", "Pair the candidate's grades with the base's.", experiment_id=eid,
                          base_report_id=base_job["report_id"], candidate_report_id=candidate_job["report_id"])
        before, after = base_job["result"]["metrics"], candidate_job["result"]["metrics"]
        raise _Stop(f"limit-hit rate {before['limit_hit_rate']:.3g} (base) -> {after['limit_hit_rate']:.3g} "
                    f"(candidate); accuracy {before['accuracy']:.3g} -> {after['accuracy']:.3g}")


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------


def _sign_fixed(vector: np.ndarray) -> np.ndarray:
    """The sign convention of the recovered directions: the entry of largest magnitude is positive."""
    return vector if vector[int(np.argmax(np.abs(vector)))] > 0 else -vector


def load_directions(path: Path) -> dict[int, dict[str, Any]]:
    with np.load(path) as data:
        return {layer: {"direction": np.asarray(data[f"layer{layer}"], dtype=np.float32),
                        "columns": [int(c) for c in np.asarray(data[f"columns{layer}"]).reshape(-1)]}
                for layer in LAYERS}


def recompute_directions(engine: Any, base_id: str, p50c_id: str) -> dict[int, dict[str, Any]]:
    """Per layer: the top left singular vector of ``ΔW[:, C]`` (float64 SVD, kept as float32), columns, and fit."""
    from wse.adapters.families import role_tensor

    manifest = engine.load_manifest(base_id)
    base, p50c = engine.open_checkpoint(base_id), engine.open_checkpoint(p50c_id)
    recovered = {}
    for layer in LAYERS:
        name = role_tensor(manifest, "mlp.down", layer, registry=engine.registry).tensor
        w = base.tensor(name).double().numpy()
        p = p50c.tensor(name).double().numpy()
        changed = w != p
        columns = np.flatnonzero(changed.any(axis=0))
        delta = (p - w)[:, columns]
        left, singular, _ = np.linalg.svd(delta, full_matrices=False)
        u = _sign_fixed(left[:, 0])
        projected, along = u @ w[:, columns], u @ delta
        recovered[layer] = {
            "direction": u.astype(np.float32), "columns": [int(c) for c in columns], "tensor": name,
            "changed_entries": int(changed.sum()),
            "rank1_energy_fraction": float(singular[0] ** 2 / np.sum(singular ** 2)),
            "alpha_least_squares": float(-(along @ projected) / (projected @ projected))}
    return recovered


def _agreement(first: Mapping[int, Mapping[str, Any]], second: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for layer in LAYERS:
        a, b = first[layer]["direction"].astype(np.float64), second[layer]["direction"].astype(np.float64)
        result[str(layer)] = {"columns_equal": first[layer]["columns"] == second[layer]["columns"],
                              "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                              "max_abs_difference": float(np.max(np.abs(a - b)))}
    return result


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the output directory")
    loop_setup.add_setup_arguments(parser)
    parser.add_argument("--directions", type=Path, default=None, help="an .npz of recovered directions and columns")
    parser.add_argument("--recompute", action="store_true", help="recompute the directions from base and p50c")
    parser.add_argument("--p50c-edit", type=Path, default=DEFAULT_P50C_EDIT, help="p50c's edit directory (read only)")
    parser.add_argument("--p50c-repo", default=DEFAULT_P50C_REPO)
    parser.add_argument("--p50c-revision", default=DEFAULT_P50C_REVISION)
    parser.add_argument("--alpha", type=float, default=6.0, help="the projection's alpha (p50c: 6)")
    parser.add_argument("--wall-clock-hours", type=float, default=3.0, help="the controller's wall-clock limit")
    parser.add_argument("--max-steps", type=int, default=3000, help="the controller's step limit (job polls count)")
    args = parser.parse_args(argv)
    loop_setup.check_setup_arguments(parser, args)
    if args.directions is None and not args.recompute:
        parser.error("give --directions NPZ, --recompute, or both")
    if not math.isfinite(args.alpha):
        parser.error("--alpha must be finite")
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

    from wse.api import Engine
    from wse.operators.tensor import shortest_floats
    from wse.registry import builtin_registry
    from wse.runner import ReferenceRunner, StrategyConfig

    out, workspace = args.out.expanduser(), args.workspace.expanduser()
    provenance: dict[str, Any] = {}
    with Engine(workspace) as engine:
        setup = loop_setup.prepare(engine, args)
        base = setup["base_checkpoint_id"]
        given = recomputed = None
        if args.directions is not None:
            given = load_directions(args.directions.expanduser())
            provenance["npz"] = {"path": str(args.directions.expanduser()),
                                 "sha256": _sha256(args.directions.expanduser())}
        if args.recompute:
            imported = engine.import_patch(args.p50c_edit.expanduser(), base_checkpoint_id=base,
                                           format="lm-understand-v1", repo=args.p50c_repo,
                                           revision=args.p50c_revision)
            recomputed = recompute_directions(engine, base, imported.checkpoint_id)
            provenance["recomputed"] = {
                "p50c_checkpoint_id": imported.checkpoint_id, "edit_dir": str(args.p50c_edit.expanduser()),
                "repo": args.p50c_repo, "revision": args.p50c_revision,
                "layers": {str(layer): {key: value for key, value in entry.items() if key != "direction"}
                           for layer, entry in recomputed.items()}}
        if given is not None and recomputed is not None:
            provenance["agreement"] = _agreement(given, recomputed)
        directions = recomputed if recomputed is not None else given
        provenance["used"] = "recomputed" if recomputed is not None else "npz"
        from wse.adapters.families import role_tensor

        manifest = engine.load_manifest(base)
        operations = []
        for layer in LAYERS:
            role = role_tensor(manifest, "mlp.down", layer, registry=engine.registry)
            direction = directions[layer]["direction"]
            if direction.size != role.shape[0]:
                parser.error(f"layer {layer}'s direction has {direction.size} values; {role.tensor} has "
                             f"{role.shape[0]} rows")
            operations.append({"op": "project_out", "tensor": role.tensor, "selection": role.selection(),
                               "direction": shortest_floats(direction), "columns": directions[layer]["columns"],
                               "alpha": args.alpha})
        grant = engine.grant_run(setup["task_id"], setup["policy_id"], base)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "directions.npz", **{f"layer{layer}": directions[layer]["direction"] for layer in LAYERS},
             **{f"columns{layer}": np.asarray(directions[layer]["columns"]) for layer in LAYERS})
    registry = builtin_registry()
    registry.register("strategy", ExpressivenessCheck.strategy_id, ExpressivenessCheck.version, ExpressivenessCheck)
    config = StrategyConfig(strategy=STRATEGY, parameters={"suite_id": setup["suite_id"], "operations": operations},
                            max_steps=args.max_steps, wall_clock_seconds=args.wall_clock_hours * 3600)
    print(f"run {grant.run_id}: {STRATEGY} on {base}, suite {setup['suite_id']}, device {args.device}",
          file=sys.stderr)
    runner = ReferenceRunner(workspace, registry=registry, worker_args=["--device", args.device],
                             log_dir=out / "logs")
    status, error, started = 0, None, time.monotonic()
    try:
        runner.run_experiment(grant.run_id, setup["task_id"], setup["policy_id"], config)
    except KeyboardInterrupt as exc:
        status, error = 130, f"interrupted: {exc}"
    except Exception as exc:  # noqa: BLE001 - the record of a failed run is still written
        status, error = 1, f"{type(exc).__name__}: {exc}"
    record: dict[str, Any] = {"script": "examples/qwen_loops/check_expressiveness.py", "workspace": str(workspace),
                              "setup": setup, "run_id": grant.run_id, "alpha": args.alpha,
                              "directions": provenance, "error": error,
                              "elapsed_seconds": round(time.monotonic() - started, 1)}
    if runner.last_experiment_id is not None:
        with Engine(workspace) as engine:
            summary = loop_setup.experiment_summary(engine, runner.last_experiment_id)
        record["experiment"] = summary
        metrics = {job["checkpoint_id"]: job.get("metrics") for job in summary["jobs"]
                   if job["kind"] == "evaluation" and job["state"] == "succeeded"}
        before = metrics.get(base)
        after = next((value for key, value in metrics.items() if key != base), None)
        if before and after:
            record["verdict"] = {
                "base": before, "candidate": after,
                "limit_hit_rate_fell": after["limit_hit_rate"] < before["limit_hit_rate"],
                "comparison": next((c for candidate in summary["candidates"] for c in candidate["comparisons"]),
                                   None)}
    loop_setup.write_json(out / f"expressiveness-{grant.run_id}.json", record)
    loop_setup.write_json(out / "expressiveness.json", record)
    print(json.dumps({"run_id": grant.run_id, "experiment_id": runner.last_experiment_id,
                      "verdict": {key: value for key, value in (record.get("verdict") or {}).items()
                                  if key != "comparison"},
                      "summary": str(out / f"expressiveness-{grant.run_id}.json"), "error": error}, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
