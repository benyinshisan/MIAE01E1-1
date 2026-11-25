"""Issue #28: traceable aggregation of completed run logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from miae01_dqn.summary import DEFAULT_CAPACITY, _verdict, build_summary, main


def _write_run(
    runs_dir: Path,
    root: Path,
    seed: int,
    mean: float,
    *,
    capacity: int = DEFAULT_CAPACITY,
    steps: int = 2_500_000,
    games: int = 30,
    lr: float = 2.5e-4,
    name_suffix: str = "",
) -> Path:
    run_dir = runs_dir / f"seed{seed}_cap{capacity}_complete{name_suffix}"
    run_dir.mkdir(parents=True)
    config = {
        "env": {"env_id": "ALE/Pong-v5", "action_repeat": 4},
        "optim": {"lr": lr, "kind": "rmsprop"},
        "algo": {
            "replay_capacity": capacity,
            "total_agent_steps": steps,
            "action_repeat": 4,
            "gamma": 0.99,
        },
        "eval": {"episodes": games, "every_frames": 1_000_000},
        "run": {"seed": seed, "run_name": run_dir.name, "device": "cuda"},
    }
    final_scores = [mean - 1.0] * (games // 2) + [mean + 1.0] * (games - games // 2)
    middle_scores = [-20.0] * games
    evaluations = [
        {"at_frame": steps * 2, "scores": middle_scores, "mean": -20.0, "std": 0.0},
        {"at_frame": steps * 4, "scores": final_scores, "mean": mean, "std": 1.0},
    ]
    result = {
        "status": "complete",
        "train": {
            "agent_steps": steps,
            "frames": steps * 4,
            "updates": steps // 4,
            "episodes": 10,
            "wall_clock_s": 12.5,
            "final_return_mean": mean,
        },
        "evaluation": evaluations[-1],
        "wall_clock_s": 13.5,
        "rss_bytes": 5 * 2**30,
        "run_dir": str(run_dir),
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "evaluations.json").write_text(json.dumps(evaluations), encoding="utf-8")
    return run_dir


def test_summary_writes_tables_and_uses_seed_sample_std(tmp_path):
    runs_dir = tmp_path / "runs"
    for seed, mean in [(1, 2.0), (2, 4.0), (3, 6.0)]:
        _write_run(runs_dir, tmp_path, seed, mean)
    output = tmp_path / "report"

    summary = build_summary(runs_dir, output)

    assert summary["seeds_present"] == [1, 2, 3]
    assert summary["paper_vs_ours"]["mean"] == pytest.approx(4.0)
    assert summary["paper_vs_ours"]["sample_std"] == pytest.approx(2.0)
    assert summary["paper_vs_ours"]["normalized"] == pytest.approx(
        100.0 * (4.0 - (-20.7)) / (9.3 - (-20.7))
    )
    assert not any("缺少种子" in warning for warning in summary["warnings"])
    for name in (
        "summary.json",
        "summary.md",
        "per_seed.csv",
        "learning_curve_per_seed.csv",
        "learning_curve_aggregate.csv",
        "learning_curve.png",
    ):
        assert (output / name).is_file(), name
    assert (output / "learning_curve.png").stat().st_size > 0

    with (output / "per_seed.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["final_mean"]) for row in rows] == [2.0, 4.0, 6.0]
    assert (output / "summary.md").read_text(encoding="utf-8").count("| 2.000") >= 1


def test_verdict_uses_preregistered_thresholds():
    assert _verdict(9.4).startswith("supported:")
    assert _verdict(0.1).startswith("partially supported:")
    assert _verdict(-0.1).startswith("not supported:")
    assert _verdict(-21.0).startswith("not supported:")


def test_missing_seed_and_unfinished_run_are_explicit(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, tmp_path, 1, 2.0)
    partial = runs_dir / "seed2_cap200000_incomplete"
    partial.mkdir()
    config = json.loads((runs_dir / "seed1_cap200000_complete" / "config.json").read_text())
    config["run"]["seed"] = 2
    config["run"]["run_name"] = partial.name
    (partial / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (partial / "progress.json").write_text(json.dumps({"agent_steps": 120_000}), encoding="utf-8")

    summary = build_summary(runs_dir, tmp_path / "report")

    warnings = "\n".join(summary["warnings"])
    assert "seed 2" in warnings and "120000" in warnings
    assert "缺少种子 3" in warnings
    assert "1/3" in warnings
    assert summary["seeds_present"] == [1]


def test_unfinished_training_budget_is_rejected(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _write_run(runs_dir, tmp_path, 1, 2.0)
    result = json.loads((run_dir / "result.json").read_text())
    result["train"]["agent_steps"] = 1000
    result["train"]["frames"] = 4000
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    summary = build_summary(runs_dir, tmp_path / "report")

    warnings = "\n".join(summary["warnings"])
    assert "训练未跑满" in warnings
    assert summary["records"] == []


def test_non_30_game_final_evaluation_is_rejected(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, tmp_path, 1, 2.0, games=29)

    summary = build_summary(runs_dir, tmp_path / "report")

    warnings = "\n".join(summary["warnings"])
    assert "29 games" in warnings
    assert summary["records"] == []


def test_config_mismatch_is_reported(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, tmp_path, 1, 2.0)
    _write_run(runs_dir, tmp_path, 2, 4.0, lr=1e-4)

    summary = build_summary(runs_dir, tmp_path / "report", seeds=(1, 2))

    warnings = "\n".join(summary["warnings"])
    assert "配置不一致" in warnings
    assert "optim.lr" in warnings


def test_capacity_ablation_is_paired_by_seed(tmp_path):
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, tmp_path, 1, 10.0)
    _write_run(runs_dir, tmp_path, 2, 12.0)
    _write_run(runs_dir, tmp_path, 1, 8.0, capacity=50_000)
    _write_run(runs_dir, tmp_path, 2, 9.0, capacity=50_000)
    _write_run(runs_dir, tmp_path, 1, 7.0, capacity=100_000)

    summary = build_summary(runs_dir, tmp_path / "report", seeds=(1, 2))

    rows = {row["capacity"]: row for row in summary["capacity_ablation"]}
    assert rows[50_000]["seeds"] == [1, 2]
    assert rows[50_000]["mean_paired_difference"] == pytest.approx(-2.5)
    assert rows[100_000]["seeds"] == [1]
    assert rows[100_000]["mean_paired_difference"] == pytest.approx(-3.0)
    assert rows[50_000]["mean_rss_bytes"] == pytest.approx(5 * 2**30)


def test_checkpoint_sha_and_strict_cli(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _write_run(runs_dir, tmp_path, 1, 2.0)
    _write_run(runs_dir, tmp_path, 2, 4.0)
    _write_run(runs_dir, tmp_path, 3, 6.0)
    (run_dir / "verification.json").write_text(
        json.dumps({"checkpoint_sha256": "a" * 64}), encoding="utf-8"
    )

    assert main(["--runs-dir", str(runs_dir), "--output-dir", str(tmp_path / "strict")]) == 0
    assert main(
        [
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(tmp_path / "strict"),
            "--seeds",
            "1,2",
            "--strict",
        ]
    ) == 1
    summary = build_summary(runs_dir, tmp_path / "report", seeds=(1, 2, 3))
    assert summary["records"][0]["checkpoint_sha256"] == "a" * 64
