"""Issue #13 (X-05): independent, non-author verification of the runner.

Author: 2260032112 黄俊皓. I implemented the replay buffer (#4) and the training
loop (#6); the runner (#8) was implemented by a different team member, so this
file satisfies the "written by a non-implementer" rule for X-05. No production
code is changed by this branch: findings go to ``docs/cross-tests-runner.md``.

Scope beyond the author's ``test_config_network_runner.py``: that file exercises
``build_config`` and ``run_experiment`` directly. Here the whole CLI path
(``main`` -> ``build_config`` -> ``run_experiment`` -> artifacts) is exercised,
including the YAML ``--config`` route, same-seed determinism and a
repository-wide scan for machine-specific absolute paths.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from miae01_dqn.runner import build_config, main, run_experiment

RUN_NAME = re.compile(
    r"^(?P<env>.+?)__seed(?P<seed>\d+)__cap(?P<cap>\d+)__\d{8}T\d{12}Z$"
)
# Small but legal budgets: warm-up 200 frames / action_repeat 4 = 50 steps,
# which is >= the default batch size of 32.
SMALL = dict(
    total_agent_steps=400,
    replay_start_frames=200,
    replay_capacity=2000,
)
# The CLI has no eval knob for the episode budget, so it goes through --config.
TINY_EVAL_YAML = "eval:\n  episodes: 1\n  max_episode_frames: 120\n"
DOCUMENTED_ARTIFACTS = (
    "config.json",
    "result.json",
    "environment.json",
    "metrics.jsonl",
    "curves.png",
)
# A real drive path starts a token: the letter is not preceded by another
# alphanumeric or by URL scheme punctuation. Without the lookbehind this regex
# also matches "https://" and the escaped newline in "algo:\n".
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9+.\-])[A-Za-z]:[\\/]")


def _run_cli(log_dir: Path, seed: int = 1, **overrides) -> Path:
    """Run the real CLI entry point and return the single run directory."""
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = log_dir / "tiny-eval.yaml"
    config_path.write_text(TINY_EVAL_YAML, encoding="utf-8")
    argv = [
        "--log-dir",
        str(log_dir),
        "--device",
        "cpu",
        "--seed",
        str(seed),
        "--config",
        str(config_path),
    ]
    for name, value in {**SMALL, **overrides}.items():
        argv += ["--" + name.replace("_", "-"), str(value)]
    assert main(argv) == 0, "runner CLI failed"
    runs = [p for p in log_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1, f"expected exactly one run directory, got {runs}"
    return runs[0]


def _scalars(run: Path) -> list[tuple[str, float, int]]:
    """Every logged scalar except SPS, which depends on wall-clock speed."""
    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [(r["tag"], r["value"], r["step"]) for r in rows if r["tag"] != "SPS"]


def test_cli_run_writes_every_documented_artifact(tmp_path):
    """Issue #13 item 2: a finished run leaves the documented output files."""
    run = _run_cli(tmp_path)
    for name in DOCUMENTED_ARTIFACTS:
        assert (run / name).is_file(), f"missing {name}"
    assert list(run.glob("events.out.tfevents.*")), "no TensorBoard event file"

    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["train"]["agent_steps"] == SMALL["total_agent_steps"]
    assert result["train"]["frames"] == SMALL["total_agent_steps"] * 4
    assert len(result["evaluation"]["scores"]) == 1
    assert result["wall_clock_s"] > 0


def test_run_directory_name_is_reversible(tmp_path):
    """Issue #13 item 1: the run name can be read back into (env, seed, capacity)."""
    run = _run_cli(tmp_path, seed=4, replay_capacity=3072)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    match = RUN_NAME.match(run.name)
    assert match, f"unexpected run name {run.name!r}"
    assert match["env"] == config["env"]["env_id"].replace("/", "-")
    assert int(match["seed"]) == config["run"]["seed"] == 4
    assert int(match["cap"]) == config["algo"]["replay_capacity"] == 3072


def test_cli_overrides_reach_the_config(tmp_path):
    """Issue #13 item 5: --seed / --replay-capacity (and --log-dir) take effect."""
    cfg = build_config(
        {
            "seed": 7,
            "replay_capacity": 4096,
            "device": "cpu",
            "log_dir": str(tmp_path),
            "total_agent_steps": 8,
            "replay_start_frames": 8,
            "batch_size": 2,
            "episodes": 1,
            "max_episode_frames": 16,
        }
    )
    assert cfg.run.seed == 7
    assert cfg.algo.replay_capacity == 4096
    assert cfg.run.device == "cpu"
    assert Path(cfg.run.log_dir) == tmp_path
    assert "seed7__cap4096__" in cfg.run.run_name


def test_build_config_rejects_unknown_fields_and_bad_run_name():
    """Issue #13 item 1: typos fail loudly instead of being ignored."""
    with pytest.raises(ValueError):
        build_config({"bogus": 1})
    with pytest.raises(ValueError):
        build_config({"run_name": ".."})


def test_smoke_profile_matches_the_documented_budget():
    """The --smoke shortcut is an explicit profile, not an implicit default."""
    cfg = build_config({"smoke": True})
    assert cfg.algo.total_agent_steps == 5000
    assert cfg.algo.replay_start_frames == 1000
    assert cfg.algo.replay_capacity == 5000
    assert cfg.eval.episodes == 2


def test_same_seed_reproduces_scalars_and_evaluation(tmp_path):
    """Re-running with an identical seed reproduces logs and evaluation scores."""
    first = _run_cli(tmp_path / "a", seed=11)
    second = _run_cli(tmp_path / "b", seed=11)
    assert _scalars(first) == _scalars(second)
    eval_a = json.loads((first / "result.json").read_text(encoding="utf-8"))["evaluation"]
    eval_b = json.loads((second / "result.json").read_text(encoding="utf-8"))["evaluation"]
    assert eval_a["scores"] == eval_b["scores"]
    assert eval_a["mean"] == eval_b["mean"]


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="a CUDA build is present, so the guard is not reachable",
)
def test_cuda_request_is_rejected_without_cuda(tmp_path):
    """Requesting a device that cannot run must fail before writing any run."""
    cfg = build_config(
        {
            "device": "cuda",
            "log_dir": str(tmp_path),
            "total_agent_steps": 8,
            "replay_start_frames": 8,
            "batch_size": 2,
            "episodes": 1,
            "max_episode_frames": 16,
        }
    )
    with pytest.raises(ValueError, match="CUDA was requested but is unavailable"):
        run_experiment(cfg)
    assert not (tmp_path / cfg.run.run_name).exists()


def test_no_machine_specific_paths_in_tracked_sources():
    """Issue #13 item 3: no absolute drive paths in versioned code or config."""
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git")
    if git is None:  # pragma: no cover - developer machine without git on PATH
        pytest.skip("git is not available")
    tracked = subprocess.run(
        [git, "ls-files", "*.py", "*.yml", "*.toml"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked, "git returned no tracked files"
    offenders = [
        rel
        for rel in tracked
        if not rel.startswith("runs/")
        and DRIVE_PATH.search((root / rel).read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, f"machine-specific paths in {offenders}"


def test_committed_run_configs_use_relative_paths():
    """Committed evidence must not embed one developer's absolute log directory."""
    root = Path(__file__).resolve().parents[1]
    configs = sorted(root.glob("runs/*/config.json"))
    assert configs, "no committed run configuration found"
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        log_dir = Path(str(config["run"]["log_dir"]))
        assert not log_dir.is_absolute(), path
        assert not log_dir.drive, path
