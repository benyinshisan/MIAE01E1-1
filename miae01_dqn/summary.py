"""Traceable result aggregation for completed runs (Issue #28).

The command reads the raw run directories under ``runs/`` and writes a
machine-readable summary, per-seed and learning-curve CSVs, a Markdown report
and a learning-curve figure.  It never re-runs training and never reads a
checkpoint: the final checkpoint stays outside Git and only its recorded
SHA-256 is reported.

Usage::

    python -m miae01_dqn.summary \
        --runs-dir runs \
        --output-dir docs/evidence/issue-28 \
        --capacity 200000 \
        --seeds 1,2,3

The main table is computed from each seed's final 30-game mean.  The across-seed
value is the mean of those seed means and the *sample* standard deviation
(``ddof=1``).  A missing seed, an unfinished run, a final evaluation that is not
30 games, or a configuration mismatch is reported explicitly instead of being
silently dropped.  Capacity-ablation rows are paired by seed against the
reference capacity; the failure-search section remains explicitly unavailable
until Issues #25/#26 provide their artifacts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_CAPACITY = 200_000
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_OUTPUT_DIR = Path("docs/evidence/issue-28")
PAPER_DQN_MEAN = 18.9
PAPER_DQN_STD = 1.3
HUMAN_MEAN = 9.3
RANDOM_MEAN = -20.7
REQUIRED_EVAL_GAMES = 30


class RunValidationError(ValueError):
    """A run exists but cannot contribute to the final comparison."""


@dataclass(frozen=True)
class RunRecord:
    """One completed run, with statistics recomputed from its raw scores."""

    path: Path
    seed: int
    capacity: int
    agent_steps: int
    frames: int
    final_frame: int
    games: int
    final_mean: float
    final_population_std: float
    normalized: float | None
    wall_clock_s: float
    rss_bytes: int | None
    checkpoint_sha256: str | None
    evaluation_frames: tuple[int, ...]
    evaluation_means: tuple[float, ...]
    evaluation_stds: tuple[float, ...]
    config: dict[str, Any] = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "capacity": self.capacity,
            "run_dir": str(self.path),
            "agent_steps": self.agent_steps,
            "frames": self.frames,
            "final_frame": self.final_frame,
            "games": self.games,
            "final_mean": self.final_mean,
            "final_population_std": self.final_population_std,
            "normalized": self.normalized,
            "wall_clock_s": self.wall_clock_s,
            "rss_bytes": self.rss_bytes,
            "checkpoint_sha256": self.checkpoint_sha256,
            "evaluation_frames": list(self.evaluation_frames),
            "evaluation_means": list(self.evaluation_means),
            "evaluation_stds": list(self.evaluation_stds),
        }


@dataclass(frozen=True)
class _Candidate:
    path: Path
    config: dict[str, Any]
    result: dict[str, Any] | None
    progress: dict[str, Any] | None
    failure: dict[str, Any] | None

    @property
    def seed(self) -> int:
        return int(self.config["run"]["seed"])

    @property
    def capacity(self) -> int:
        return int(self.config["algo"]["replay_capacity"])

    @property
    def status(self) -> str:
        if self.result is not None:
            return str(self.result.get("status", "missing-status"))
        return "incomplete"

    @property
    def reported_steps(self) -> int | None:
        if self.result is not None:
            value = self.result.get("train", {}).get("agent_steps")
            if value is not None:
                return int(value)
        if self.progress is not None:
            value = self.progress.get("agent_steps")
            if value is not None:
                return int(value)
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read_json(path)
    if not isinstance(value, dict):
        raise RunValidationError(f"{path} must contain a JSON object")
    return value


def _scan_candidates(runs_dir: Path) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for path in sorted(Path(runs_dir).iterdir()):
        if not path.is_dir() or not (path / "config.json").is_file():
            continue
        try:
            config = _read_json(path / "config.json")
            int(config["run"]["seed"])
            int(config["algo"]["replay_capacity"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        candidates.append(
            _Candidate(
                path=path,
                config=config,
                result=_optional_json(path / "result.json"),
                progress=_optional_json(path / "progress.json"),
                failure=_optional_json(path / "failure.json"),
            )
        )
    return candidates


def _recompute_final_statistics(
    candidate: _Candidate,
) -> tuple[dict[str, Any], tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    if candidate.result is None:
        raise RunValidationError(
            f"{candidate.path.name}: 训练未跑满（缺少 result.json）"
        )
    result = candidate.result
    if result.get("status") != "complete":
        raise RunValidationError(
            f"{candidate.path.name}: 训练未跑满（result status={result.get('status')!r}）"
        )

    try:
        train = result["train"]
        final = result["evaluation"]
        scores = [float(score) for score in final["scores"]]
        agent_steps = int(train["agent_steps"])
        frames = int(train["frames"])
        expected_steps = int(candidate.config["algo"]["total_agent_steps"])
        action_repeat = int(candidate.config["algo"]["action_repeat"])
        expected_frames = expected_steps * action_repeat
        expected_games = int(candidate.config["eval"]["episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunValidationError(f"{candidate.path.name}: incomplete run metadata ({exc})") from exc

    if agent_steps != expected_steps:
        raise RunValidationError(
            f"{candidate.path.name}: 训练未跑满 "
            f"({agent_steps}/{expected_steps} agent steps)"
        )
    if frames != expected_frames:
        raise RunValidationError(
            f"{candidate.path.name}: frame count {frames} != configured {expected_frames}"
        )
    if expected_games != REQUIRED_EVAL_GAMES:
        raise RunValidationError(
            f"{candidate.path.name}: configured final evaluation is {expected_games} games, "
            f"expected {REQUIRED_EVAL_GAMES}"
        )
    if len(scores) != REQUIRED_EVAL_GAMES:
        raise RunValidationError(
            f"{candidate.path.name}: final evaluation contains {len(scores)} games, "
            f"expected {REQUIRED_EVAL_GAMES}"
        )
    if not all(math.isfinite(score) for score in scores):
        raise RunValidationError(f"{candidate.path.name}: non-finite final score")

    final_frame = int(final.get("at_frame", -1))
    if final_frame != frames:
        raise RunValidationError(
            f"{candidate.path.name}: final evaluation frame {final_frame} != training frames {frames}"
        )

    mean = float(np.mean(scores))
    population_std = float(np.std(scores))
    recorded_mean = final.get("mean")
    recorded_std = final.get("std")
    if recorded_mean is not None and not math.isclose(
        mean, float(recorded_mean), rel_tol=0.0, abs_tol=1e-9
    ):
        raise RunValidationError(
            f"{candidate.path.name}: recorded mean {recorded_mean} != recomputed {mean}"
        )
    if recorded_std is not None and not math.isclose(
        population_std, float(recorded_std), rel_tol=0.0, abs_tol=1e-9
    ):
        raise RunValidationError(
            f"{candidate.path.name}: recorded std {recorded_std} != recomputed {population_std}"
        )

    evaluations_path = candidate.path / "evaluations.json"
    evaluations = _read_json(evaluations_path) if evaluations_path.is_file() else []
    if not isinstance(evaluations, list) or not evaluations:
        raise RunValidationError(f"{candidate.path.name}: evaluations.json is missing or empty")

    evaluation_frames: list[int] = []
    evaluation_means: list[float] = []
    evaluation_stds: list[float] = []
    for index, evaluation in enumerate(evaluations):
        try:
            frame = int(evaluation["at_frame"])
            game_scores = [float(score) for score in evaluation["scores"]]
            game_mean = float(evaluation["mean"])
            game_std = float(evaluation["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RunValidationError(
                f"{candidate.path.name}: evaluation round {index} is malformed"
            ) from exc
        if len(game_scores) != REQUIRED_EVAL_GAMES:
            raise RunValidationError(
                f"{candidate.path.name}: evaluation round {index} has "
                f"{len(game_scores)} games, expected {REQUIRED_EVAL_GAMES}"
            )
        if not math.isclose(game_mean, float(np.mean(game_scores)), rel_tol=0.0, abs_tol=1e-9):
            raise RunValidationError(
                f"{candidate.path.name}: evaluation round {index} mean does not match its scores"
            )
        if not math.isclose(
            game_std, float(np.std(game_scores)), rel_tol=0.0, abs_tol=1e-9
        ):
            raise RunValidationError(
                f"{candidate.path.name}: evaluation round {index} std does not match its scores"
            )
        evaluation_frames.append(frame)
        evaluation_means.append(game_mean)
        evaluation_stds.append(game_std)

    if evaluation_frames != sorted(evaluation_frames):
        raise RunValidationError(f"{candidate.path.name}: evaluation frames are not increasing")
    if evaluation_frames[-1] != frames:
        raise RunValidationError(
            f"{candidate.path.name}: last evaluation frame {evaluation_frames[-1]} != {frames}"
        )
    if not math.isclose(evaluation_means[-1], mean, rel_tol=0.0, abs_tol=1e-9):
        raise RunValidationError(
            f"{candidate.path.name}: final evaluation does not match result.json"
        )

    return final, tuple(evaluation_frames), tuple(evaluation_means), tuple(evaluation_stds)


def _load_record(candidate: _Candidate) -> RunRecord:
    final, evaluation_frames, evaluation_means, evaluation_stds = (
        _recompute_final_statistics(candidate)
    )
    scores = [float(score) for score in final["scores"]]
    mean = float(np.mean(scores))
    normalized = final.get("normalized")
    if normalized is not None:
        normalized = float(normalized)
    wall_clock_s = candidate.result.get("wall_clock_s")
    if wall_clock_s is None:
        wall_clock_s = candidate.result["train"].get("wall_clock_s", float("nan"))
    rss_bytes = candidate.result.get("rss_bytes")
    if rss_bytes is not None:
        rss_bytes = int(rss_bytes)
    checkpoint_sha256 = None
    verification_path = candidate.path / "verification.json"
    if verification_path.is_file():
        verification = _read_json(verification_path)
        if isinstance(verification, dict):
            value = verification.get("checkpoint_sha256")
            if isinstance(value, str):
                checkpoint_sha256 = value
            peak = verification.get("observed_peak_rss_bytes")
            if peak is not None:
                rss_bytes = int(peak)
    return RunRecord(
        path=candidate.path,
        seed=candidate.seed,
        capacity=candidate.capacity,
        agent_steps=int(candidate.result["train"]["agent_steps"]),
        frames=int(candidate.result["train"]["frames"]),
        final_frame=int(final["at_frame"]),
        games=len(scores),
        final_mean=mean,
        final_population_std=float(np.std(scores)),
        normalized=normalized,
        wall_clock_s=float(wall_clock_s),
        rss_bytes=rss_bytes,
        checkpoint_sha256=checkpoint_sha256,
        evaluation_frames=evaluation_frames,
        evaluation_means=evaluation_means,
        evaluation_stds=evaluation_stds,
        config=candidate.config,
    )


def _config_differences(
    reference: dict[str, Any],
    other: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    differences: list[str] = []
    if isinstance(reference, dict) and isinstance(other, dict):
        for key in sorted(set(reference) | set(other)):
            path = f"{prefix}.{key}" if prefix else key
            if path in {"run.seed", "run.run_name"}:
                continue
            if key not in reference:
                differences.append(f"{path}: missing in reference")
            elif key not in other:
                differences.append(f"{path}: missing in run")
            else:
                differences.extend(
                    _config_differences(reference[key], other[key], path)
                )
    elif reference != other:
        differences.append(f"{prefix}: {reference!r} != {other!r}")
    return differences


def _describe_incomplete(candidate: _Candidate) -> str:
    steps = candidate.reported_steps
    progress = f", {steps} agent steps recorded" if steps is not None else ""
    failure = candidate.failure
    if failure:
        progress += f", failure={failure.get('type', 'unknown')}"
    return f"{candidate.path.name} (status={candidate.status}{progress})"


def _select_records(
    candidates: Sequence[_Candidate],
    capacity: int,
    seeds: Sequence[int],
) -> tuple[dict[int, RunRecord], list[str]]:
    warnings: list[str] = []
    records: dict[int, RunRecord] = {}
    for seed in seeds:
        matches = [
            candidate
            for candidate in candidates
            if candidate.seed == seed and candidate.capacity == capacity
        ]
        valid: list[RunRecord] = []
        incomplete: list[_Candidate] = []
        invalid_messages: list[str] = []
        for candidate in matches:
            try:
                valid.append(_load_record(candidate))
            except RunValidationError as exc:
                incomplete.append(candidate)
                invalid_messages.append(f"seed {seed}: {exc}")
        if not valid:
            # Only surface unfinished runs when they are the reason a seed is
            # absent.  A historical failed attempt next to a completed rerun
            # must not make --strict fail forever.
            warnings.extend(invalid_messages)
            if incomplete:
                details = "; ".join(_describe_incomplete(item) for item in incomplete)
                warnings.append(
                    f"seed {seed}: 没有可用的完整运行（{details}）；主表跳过该种子"
                )
            else:
                warnings.append(
                    f"缺少种子 {seed}: 未找到容量 {capacity} 的完整运行目录"
                )
            continue
        if len(valid) > 1:
            chosen = max(valid, key=lambda record: record.path.stat().st_mtime)
            warnings.append(
                f"seed {seed}: 找到 {len(valid)} 个完整运行，选用最新的 "
                f"{chosen.path.name}；其余被忽略"
            )
            records[seed] = chosen
        else:
            records[seed] = valid[0]
    return records, warnings


def _check_config_consistency(
    records: dict[int, RunRecord],
) -> list[str]:
    if len(records) < 2:
        return []
    reference_seed = min(records)
    reference = records[reference_seed].config
    warnings: list[str] = []
    for seed in sorted(records):
        if seed == reference_seed:
            continue
        differences = _config_differences(reference, records[seed].config)
        if differences:
            shown = "; ".join(differences[:8])
            more = f"; (+{len(differences) - 8} more)" if len(differences) > 8 else ""
            warnings.append(
                f"配置不一致: seed {seed} 与 seed {reference_seed} 不同: {shown}{more}"
            )
    return warnings


def _aggregate_curve(
    records: dict[int, RunRecord],
) -> tuple[list[int], list[float | None], list[int]]:
    if not records:
        return [], [], []
    frame_sets = [set(record.evaluation_frames) for record in records.values()]
    common = sorted(set.intersection(*frame_sets))
    means: list[float | None] = []
    counts: list[int] = []
    by_seed = {
        seed: dict(zip(record.evaluation_frames, record.evaluation_means))
        for seed, record in records.items()
    }
    for frame in common:
        values = [by_seed[seed][frame] for seed in records]
        means.append(float(np.mean(values)))
        counts.append(len(values))
    return common, means, counts


def _sample_std(values: Iterable[float]) -> float | None:
    values = list(values)
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def _verdict(mean: float) -> str:
    # README's pre-registered rule: >9.3 supported; positive but below 9.3
    # partially supported; otherwise not supported.  Being above the random
    # baseline alone is not enough for the middle category.
    if mean > HUMAN_MEAN:
        return "supported: above professional human"
    if mean > 0:
        return "partially supported: positive but below professional human"
    if mean > RANDOM_MEAN:
        return "not supported: above random but not positive"
    return "not supported: at or below random"


def _paper_vs_ours(records: dict[int, RunRecord]) -> dict[str, Any]:
    seed_means = {seed: record.final_mean for seed, record in sorted(records.items())}
    values = list(seed_means.values())
    pooled_mean = float(np.mean(values)) if values else None
    pooled_std = _sample_std(values)
    normalized = (
        None
        if pooled_mean is None
        else 100.0 * (pooled_mean - RANDOM_MEAN) / (HUMAN_MEAN - RANDOM_MEAN)
    )
    return {
        "seeds": seed_means,
        "n_seeds": len(values),
        "mean": pooled_mean,
        "sample_std": pooled_std,
        "normalized": normalized,
        "verdict": None if pooled_mean is None else _verdict(pooled_mean),
        "paper": {
            "dqn_mean": PAPER_DQN_MEAN,
            "dqn_std": PAPER_DQN_STD,
            "human_mean": HUMAN_MEAN,
            "random_mean": RANDOM_MEAN,
            "normalized": 100.0
            * (PAPER_DQN_MEAN - RANDOM_MEAN)
            / (HUMAN_MEAN - RANDOM_MEAN),
        },
    }


def _capacity_ablation(
    candidates: Sequence[_Candidate],
    reference_capacity: int,
    reference_steps: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    by_capacity: dict[int, dict[int, RunRecord]] = {}
    seen_capacities: set[int] = set()
    for candidate in candidates:
        seen_capacities.add(candidate.capacity)
        if candidate.result is None:
            continue
        try:
            record = _load_record(candidate)
        except RunValidationError:
            continue
        if reference_steps is not None and record.agent_steps != reference_steps:
            continue
        by_capacity.setdefault(record.capacity, {})[record.seed] = record

    reference = by_capacity.get(reference_capacity, {})
    other_capacities = sorted(capacity for capacity in by_capacity if capacity != reference_capacity)
    if not other_capacities:
        warnings.append(
            f"容量消融: 未找到 {reference_capacity} 以外的完整运行（依赖 Issue #27）"
        )
        return [], warnings

    rows: list[dict[str, Any]] = []
    for capacity in other_capacities:
        records = by_capacity[capacity]
        paired_seeds = sorted(set(reference) & set(records))
        if not paired_seeds:
            warnings.append(
                f"容量消融: capacity {capacity} 没有与 {reference_capacity} 配对的完整种子"
            )
            continue
        differences = [
            records[seed].final_mean - reference[seed].final_mean
            for seed in paired_seeds
        ]
        rows.append(
            {
                "capacity": capacity,
                "seeds": paired_seeds,
                "paired_seeds": len(paired_seeds),
                "mean_final_return": float(
                    np.mean([records[seed].final_mean for seed in paired_seeds])
                ),
                "mean_paired_difference": float(np.mean(differences)),
                "paired_differences": {
                    str(seed): float(difference)
                    for seed, difference in zip(paired_seeds, differences)
                },
                "mean_wall_clock_s": float(
                    np.mean([records[seed].wall_clock_s for seed in paired_seeds])
                ),
                "mean_rss_bytes": _mean_or_none(
                    [records[seed].rss_bytes for seed in paired_seeds]
                ),
            }
        )
    if not rows:
        warnings.append("容量消融: 没有可配对的容量组")
    return rows, warnings


def _mean_or_none(values: Sequence[int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(
    summary: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"# Pong {summary['capacity'] // 1000}k 结果汇总",
        "",
        f"生成时间（UTC）：`{summary['generated_at_utc']}`",
        f"扫描目录：`{summary['runs_dir']}`",
        f"参考容量：`{summary['capacity']}`｜请求种子：`{summary['seeds_requested']}`"
        f"｜实际种子：`{summary['seeds_present']}`",
        "",
    ]
    warnings = summary["warnings"]
    if warnings:
        lines.extend(["## 需要处理", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    ours = summary["paper_vs_ours"]
    lines.extend(
        [
            "## Paper vs Ours",
            "",
            "| 项目 | 最终 30 局原始分 | 种子数 | 说明 |",
            "|---|---:|---:|---|",
            f"| 论文 DQN | {PAPER_DQN_MEAN:.1f} ± {PAPER_DQN_STD:.1f} | — | Extended Data Table 2 |",
            f"| **本复现** | **{_fmt(ours['mean'])} ± {_fmt(ours['sample_std'])}** "
            f"| {ours['n_seeds']} | 种子均值的均值 ± 样本标准差（ddof=1） |",
            f"| 专业人类 | {HUMAN_MEAN:.1f} | — | 论文基线 |",
            f"| 随机策略 | {RANDOM_MEAN:.1f} | — | 论文基线 |",
            "",
            f"人类归一化分数：**{_fmt(ours['normalized'], 1)}%**"
            f"（论文 DQN：{_fmt(ours['paper']['normalized'], 1)}%）；"
            f"预注册判定：**{ours['verdict'] or '—'}**。",
            "",
            "## 逐种子最终结果",
            "",
            "| Seed | 运行目录 | 最终帧 | 局数 | 最终均值 | 总体标准差 | 归一化 | 耗时 (s) | 峰值 RSS (GiB) | 检查点 SHA-256 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in summary["records"]:
        sha = record["checkpoint_sha256"]
        sha_text = f"`{sha[:12]}…`" if sha else "未记录"
        rss = (
            f"{record['rss_bytes'] / 2**30:.2f}"
            if record["rss_bytes"] is not None
            else "—"
        )
        lines.append(
            f"| {record['seed']} | `{record['run_dir']}` | {record['final_frame']} "
            f"| {record['games']} | {_fmt(record['final_mean'])} "
            f"| {_fmt(record['final_population_std'])} "
            f"| {_fmt(record['normalized'], 1)}% | {_fmt(record['wall_clock_s'], 1)} "
            f"| {rss} | {sha_text} |"
        )
    if not summary["records"]:
        lines.append("| — | 没有可用运行 | — | — | — | — | — | — | — | — |")

    lines.extend(["", "## 学习曲线", ""])
    if summary["records"]:
        lines.append(
            "![每种子与跨种子评测曲线](learning_curve.png)"
        )
        lines.append("")
        lines.append(
            "曲线按每 1M 名义训练帧的 30 局评测均值绘制；"
            "阴影/误差条为跨种子样本标准差。"
        )
    else:
        lines.append("没有可用运行，未生成学习曲线。")

    lines.extend(["", "## 容量消融", ""])
    if summary["capacity_ablation"]:
        lines.extend(
            [
                "| 容量 | 配对种子 | 最终均值 | 配对差值（相对参考容量） | 平均耗时 (s) | 平均峰值 RSS (GiB) |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary["capacity_ablation"]:
            rss = (
                f"{row['mean_rss_bytes'] / 2**30:.2f}"
                if row["mean_rss_bytes"] is not None
                else "—"
            )
            paired = ", ".join(str(seed) for seed in row["seeds"])
            lines.append(
                f"| {row['capacity']} | {paired} | {_fmt(row['mean_final_return'])} "
                f"| {_fmt(row['mean_paired_difference'])} "
                f"| {_fmt(row['mean_wall_clock_s'], 1)} | {rss} |"
            )
    else:
        lines.append(
            "未接入：没有 50k/100k 的完整配对运行；该部分依赖 Issue #27。"
        )

    lines.extend(["", "## 失败搜索", ""])
    lines.append(
        f"未接入：{summary['failure_search']['reason']}"
    )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `summary.json`：机器可读的完整汇总与警告",
            "- `per_seed.csv`：逐种子最终结果",
            "- `learning_curve_per_seed.csv`：每种子逐轮评测",
            "- `learning_curve_aggregate.csv`：共同评测帧上的跨种子均值与样本标准差",
            "- `learning_curve.png`：学习曲线图",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _plot_learning_curve(
    records: dict[int, RunRecord],
    aggregate_frames: Sequence[int],
    aggregate_means: Sequence[float | None],
    aggregate_counts: Sequence[int],
    output_path: Path,
    capacity: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for index, (seed, record) in enumerate(sorted(records.items())):
        axis.plot(
            np.asarray(record.evaluation_frames) / 1e6,
            record.evaluation_means,
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=colors[index % len(colors)],
            label=f"seed {seed}",
        )
    if records and len(records) >= 2:
        common_indices = [index for index, count in enumerate(aggregate_counts) if count >= 2]
        x = [aggregate_frames[index] / 1e6 for index in common_indices]
        means = [aggregate_means[index] for index in common_indices]
        stds: list[float] = []
        by_seed = {
            seed: dict(zip(record.evaluation_frames, record.evaluation_means))
            for seed, record in records.items()
        }
        for index in common_indices:
            frame = aggregate_frames[index]
            values = [by_seed[seed][frame] for seed in records]
            stds.append(float(np.std(values, ddof=1)))
        axis.errorbar(
            x,
            means,
            yerr=stds,
            marker="s",
            markersize=3,
            linewidth=2,
            capsize=3,
            color="black",
            label=f"seed mean ± sample std (n={len(records)})",
        )
    axis.axhline(HUMAN_MEAN, color="#2ca02c", linestyle="--", linewidth=1,
                 label=f"professional human ({HUMAN_MEAN})")
    axis.axhline(PAPER_DQN_MEAN, color="#ff7f0e", linestyle=":", linewidth=1,
                 label=f"paper DQN ({PAPER_DQN_MEAN})")
    axis.axhline(RANDOM_MEAN, color="#7f7f7f", linestyle="-.", linewidth=1,
                 label=f"random ({RANDOM_MEAN})")
    axis.set_xlabel("Nominal training frames (millions)")
    axis.set_ylabel("Evaluation raw return (30 games)")
    axis.set_title(f"Pong {capacity // 1000}k evaluation learning curve")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_summary(
    runs_dir: str | Path,
    output_dir: str | Path,
    *,
    capacity: int = DEFAULT_CAPACITY,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Read completed runs and write the Issue #28 artifacts.

    Returns the same dictionary that is written to ``summary.json``.
    """
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    candidates = _scan_candidates(runs_path)
    records, warnings = _select_records(candidates, capacity, seeds)
    warnings.extend(_check_config_consistency(records))

    reference_steps = (
        next(iter(records.values())).agent_steps if records else None
    )
    ablation_rows, ablation_warnings = _capacity_ablation(
        candidates, capacity, reference_steps
    )
    warnings.extend(ablation_warnings)

    missing = [seed for seed in seeds if seed not in records]
    if missing:
        warnings.append(
            f"主表缺少种子 {missing}（{len(records)}/{len(seeds)}）；"
            "样本标准差至少需要 2 个种子，最终验收需要 3 个"
        )

    curve_frames, curve_means, curve_counts = _aggregate_curve(records)
    paper = _paper_vs_ours(records)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "runs_dir": str(runs_path),
        "output_dir": str(output_path),
        "capacity": capacity,
        "seeds_requested": list(seeds),
        "seeds_present": sorted(records),
        "records": [records[seed].as_dict() for seed in sorted(records)],
        "paper_vs_ours": paper,
        "learning_curve": {
            "frames": list(curve_frames),
            "mean": list(curve_means),
            "n_seeds": list(curve_counts),
        },
        "capacity_ablation": ablation_rows,
        "failure_search": {
            "available": False,
            "reason": "尚未找到 Issue #25/#26 的失败开局清单与等预算对照产物",
        },
        "warnings": warnings,
    }

    # Machine-readable summary first, then the human-readable artifacts.
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    per_seed_rows = []
    for seed in sorted(records):
        record = records[seed].as_dict()
        per_seed_rows.append(record)
    _write_csv(
        output_path / "per_seed.csv",
        per_seed_rows,
        [
            "seed",
            "capacity",
            "run_dir",
            "agent_steps",
            "frames",
            "final_frame",
            "games",
            "final_mean",
            "final_population_std",
            "normalized",
            "wall_clock_s",
            "rss_bytes",
            "checkpoint_sha256",
        ],
    )

    curve_rows: list[dict[str, Any]] = []
    for seed in sorted(records):
        record = records[seed]
        for frame, mean, std in zip(
            record.evaluation_frames, record.evaluation_means, record.evaluation_stds
        ):
            curve_rows.append(
                {
                    "seed": seed,
                    "frame": frame,
                    "mean": mean,
                    "population_std": std,
                    "games": record.games,
                }
            )
    _write_csv(
        output_path / "learning_curve_per_seed.csv",
        curve_rows,
        ["seed", "frame", "mean", "population_std", "games"],
    )

    aggregate_rows = []
    for frame, mean, count in zip(curve_frames, curve_means, curve_counts):
        values = [
            records[seed].evaluation_means[
                records[seed].evaluation_frames.index(frame)
            ]
            for seed in records
        ]
        aggregate_rows.append(
            {
                "frame": frame,
                "n_seeds": count,
                "mean": mean,
                "sample_std": _sample_std(values) if count >= 2 else None,
            }
        )
    _write_csv(
        output_path / "learning_curve_aggregate.csv",
        aggregate_rows,
        ["frame", "n_seeds", "mean", "sample_std"],
    )

    _plot_learning_curve(
        records,
        curve_frames,
        curve_means,
        curve_counts,
        output_path / "learning_curve.png",
        capacity,
    )
    (output_path / "summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    return summary


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate completed Pong runs into the Issue #28 report"
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on missing seeds, unfinished runs or config mismatches",
    )
    args = parser.parse_args(argv)
    summary = build_summary(
        args.runs_dir,
        args.output_dir,
        capacity=args.capacity,
        seeds=args.seeds,
    )
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}", flush=True)
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "seeds_present": summary["seeds_present"],
                "paper_vs_ours": summary["paper_vs_ours"],
                "warnings": len(summary["warnings"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if args.strict and summary["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
