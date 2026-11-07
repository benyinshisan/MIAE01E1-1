"""Configuration dataclasses. Implementation: issue I-01.

Validation is deliberately small:

* ``_check_types`` reads the annotations and rejects wrong runtime types
  (Python does not enforce annotations by itself) and raises ``TypeError``.
* range and cross-field invariants use ``assert``, one statement each.

.. warning::
   The ``assert`` checks disappear under ``python -O``. Run without ``-O``
   (the default) so that invalid configurations always fail fast.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any


def _check_types(obj: Any) -> None:
    """Reject values whose runtime type does not match the annotation."""
    hints = typing.get_type_hints(type(obj))
    for f in fields(obj):
        expected = hints.get(f.name)
        if expected is None:
            continue
        value = getattr(obj, f.name)
        if not _matches(value, expected):
            raise TypeError(
                f"{f.name} must be {_name(expected)}, got {type(value).__name__}: {value!r}"
            )


def _matches(value: Any, expected: Any) -> bool:
    origin = typing.get_origin(expected)
    if origin is typing.Union:                     # e.g. float | None
        return any(_matches(value, arg) for arg in typing.get_args(expected))
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def _name(expected: Any) -> str:
    origin = typing.get_origin(expected)
    if origin is typing.Union:
        return " or ".join(_name(a) for a in typing.get_args(expected))
    return getattr(expected, "__name__", str(expected))


@dataclass(frozen=True)
class EnvSpec:
    env_id: str = "ALE/Pong-v5"
    frame_size: int = 84
    frame_stack: int = 4
    action_repeat: int = 4
    noop_max: int = 30
    frameskip: int = 1
    repeat_action_probability: float = 0.0
    episodic_life: bool = True
    clip_reward: bool = True
    grayscale: bool = True
    max_and_skip: bool = True

    def __post_init__(self) -> None:
        _check_types(self)
        assert self.env_id, "env_id must not be empty"
        assert self.frame_size >= 1, f"frame_size must be >= 1, got {self.frame_size}"
        assert self.frame_stack >= 1, f"frame_stack must be >= 1, got {self.frame_stack}"
        assert self.action_repeat >= 1, (
            f"action_repeat must be >= 1, got {self.action_repeat}"
        )
        assert self.noop_max >= 0, f"noop_max must be >= 0, got {self.noop_max}"
        assert self.frameskip >= 1, f"frameskip must be >= 1, got {self.frameskip}"
        assert 0.0 <= self.repeat_action_probability <= 1.0, (
            f"repeat_action_probability must be in [0, 1], "
            f"got {self.repeat_action_probability}"
        )
        # The ALE env must not skip frames itself: MaxAndSkip already repeats actions.
        assert not (self.frameskip > 1 and self.max_and_skip), (
            f"frameskip must be 1 when max_and_skip is enabled, got {self.frameskip} "
            f"(action repeat would compound to {self.frameskip * self.action_repeat})"
        )

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        """Channel-first shape fed to the network: (frame_stack, H, W)."""
        return (self.frame_stack, self.frame_size, self.frame_size)


@dataclass(frozen=True)
class OptimSpec:
    kind: str = "rmsprop"
    lr: float = 2.5e-4
    momentum: float = 0.95
    alpha: float = 0.95
    eps: float = 0.01
    td_error_clip: float = 1.0
    grad_clip_norm: float | None = None

    def __post_init__(self) -> None:
        _check_types(self)
        assert self.kind in ("rmsprop", "adam"), (
            f"kind must be 'rmsprop' or 'adam', got {self.kind!r}"
        )
        assert self.lr > 0, f"lr must be > 0, got {self.lr}"
        assert 0 <= self.momentum < 1, f"momentum must be in [0, 1), got {self.momentum}"
        assert 0 <= self.alpha < 1, f"alpha must be in [0, 1), got {self.alpha}"
        assert self.eps >= 0, f"eps must be >= 0, got {self.eps}"
        assert self.td_error_clip >= 0, (
            f"td_error_clip must be >= 0 (0 disables clipping), got {self.td_error_clip}"
        )
        assert self.grad_clip_norm is None or self.grad_clip_norm > 0, (
            f"grad_clip_norm must be None or > 0, got {self.grad_clip_norm}"
        )


@dataclass(frozen=True)
class AlgoSpec:
    gamma: float = 0.99
    batch_size: int = 32
    replay_capacity: int = 200_000
    replay_start_frames: int = 50_000
    learn_every_steps: int = 4
    target_update_every_updates: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_frames: int = 1_000_000
    total_agent_steps: int = 2_500_000
    action_repeat: int = 4

    def __post_init__(self) -> None:
        _check_types(self)
        assert 0 < self.gamma <= 1, f"gamma must be in (0, 1], got {self.gamma}"
        assert self.batch_size >= 1, f"batch_size must be >= 1, got {self.batch_size}"
        assert self.replay_capacity >= 1, (
            f"replay_capacity must be >= 1, got {self.replay_capacity}"
        )
        assert self.replay_start_frames >= 0, (
            f"replay_start_frames must be >= 0, got {self.replay_start_frames}"
        )
        assert self.learn_every_steps >= 1, (
            f"learn_every_steps must be >= 1, got {self.learn_every_steps}"
        )
        assert self.target_update_every_updates >= 1, (
            f"target_update_every_updates must be >= 1, "
            f"got {self.target_update_every_updates}"
        )
        assert 0 <= self.eps_start <= 1, (
            f"eps_start must be in [0, 1], got {self.eps_start}"
        )
        assert 0 <= self.eps_end <= 1, f"eps_end must be in [0, 1], got {self.eps_end}"
        assert self.eps_end <= self.eps_start, (
            f"eps_end ({self.eps_end}) must be <= eps_start ({self.eps_start})"
        )
        assert self.eps_decay_frames >= 0, (
            f"eps_decay_frames must be >= 0, got {self.eps_decay_frames}"
        )
        assert self.total_agent_steps >= 1, (
            f"total_agent_steps must be >= 1, got {self.total_agent_steps}"
        )
        assert self.action_repeat >= 1, (
            f"action_repeat must be >= 1, got {self.action_repeat}"
        )

        total_frames = self.total_agent_steps * self.action_repeat
        assert self.replay_start_frames <= total_frames, (
            f"replay_start_frames ({self.replay_start_frames}) exceeds the total "
            f"frame budget ({total_frames}); learning would never start"
        )
        warmup_steps = self.replay_start_frames // self.action_repeat
        assert self.batch_size <= warmup_steps, (
            f"batch_size ({self.batch_size}) exceeds the steps collected before "
            f"learning starts ({warmup_steps}); the first update could not be sampled"
        )
        assert self.batch_size <= self.replay_capacity, (
            f"batch_size ({self.batch_size}) must not exceed replay_capacity "
            f"({self.replay_capacity})"
        )

    @property
    def total_frames(self) -> int:
        """Training budget in Atari frames (agent steps x action repeat)."""
        return self.total_agent_steps * self.action_repeat

    @property
    def replay_start_steps(self) -> int:
        """Warm-up length converted from frames to agent steps."""
        return self.replay_start_frames // self.action_repeat


@dataclass(frozen=True)
class EvalSpec:
    every_frames: int = 1_000_000
    episodes: int = 30
    epsilon: float = 0.05
    max_episode_frames: int = 18_000

    def __post_init__(self) -> None:
        _check_types(self)
        assert self.every_frames >= 1, (
            f"every_frames must be >= 1, got {self.every_frames}"
        )
        assert self.episodes >= 1, f"episodes must be >= 1, got {self.episodes}"
        assert 0 <= self.epsilon <= 1, f"epsilon must be in [0, 1], got {self.epsilon}"
        assert self.max_episode_frames >= 1, (
            f"max_episode_frames must be >= 1, got {self.max_episode_frames}"
        )


@dataclass(frozen=True)
class RunSpec:
    seed: int = 1
    device: str = "cpu"
    run_name: str = ""
    log_dir: Path = field(default_factory=lambda: Path("runs"))
    save_checkpoints: bool = False

    def __post_init__(self) -> None:
        _check_types(self)
        assert self.seed >= 0, f"seed must be >= 0, got {self.seed}"
        assert self.device == "cpu" or self.device.startswith("cuda"), (
            f"device must be 'cpu' or 'cuda[:N]', got {self.device!r}"
        )
        assert "/" not in self.run_name and "\\" not in self.run_name, (
            f"run_name must not contain path separators, got {self.run_name!r}"
        )

    @property
    def run_dir(self) -> Path:
        """Directory of this run: log_dir / run_name."""
        return Path(self.log_dir) / self.run_name


_SPEC_CLASSES: dict[str, type] = {
    "env": EnvSpec,
    "optim": OptimSpec,
    "algo": AlgoSpec,
    "eval": EvalSpec,
    "run": RunSpec,
}
_SPEC_FIELDS: dict[str, frozenset[str]] = {
    name: frozenset(f.name for f in fields(cls)) for name, cls in _SPEC_CLASSES.items()
}


def _cast(raw: Any, expected: Any) -> Any:
    """Coerce an override value (often a CLI string) to the annotated type."""
    origin = typing.get_origin(expected)
    if origin is typing.Union:                     # e.g. float | None
        if raw is None:
            return None
        inner = [a for a in typing.get_args(expected) if a is not type(None)]
        return _cast(raw, inner[0]) if len(inner) == 1 else raw
    if expected is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text in ("1", "true", "yes", "on"):
                return True
            if text in ("0", "false", "no", "off"):
                return False
        raise ValueError(f"cannot interpret {raw!r} as bool")
    if expected is int:
        return int(raw)
    if expected is float:
        return float(raw)
    if expected is str:
        return str(raw)
    if expected is Path:
        return Path(raw)
    return raw


def _coerce(spec_name: str, field_name: str, raw: Any) -> Any:
    expected = typing.get_type_hints(_SPEC_CLASSES[spec_name])[field_name]
    try:
        return _cast(raw, expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cannot set {spec_name}.{field_name}={raw!r}: {exc}"
        ) from exc


@dataclass(frozen=True)
class RunConfig:
    env: EnvSpec = field(default_factory=EnvSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    algo: AlgoSpec = field(default_factory=AlgoSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    run: RunSpec = field(default_factory=RunSpec)

    def __post_init__(self) -> None:
        assert self.algo.action_repeat == self.env.action_repeat, (
            "action_repeat mismatch: "
            f"algo={self.algo.action_repeat} env={self.env.action_repeat}. "
            "The frame/step conversion in AlgoSpec must match the environment's "
            "action repeat."
        )

    def with_overrides(self, **flat: Any) -> "RunConfig":
        """Return a new config with the given fields replaced.

        Keys may be qualified (``algo.gamma=0.9``) or bare (``gamma=0.9``).
        Bare keys must be unambiguous across the five specs. Values are coerced
        to the annotated type, so CLI strings such as ``"64"`` work.
        """
        specs = {
            "env": self.env,
            "optim": self.optim,
            "algo": self.algo,
            "eval": self.eval,
            "run": self.run,
        }
        patches: dict[str, dict[str, Any]] = {name: {} for name in specs}

        for key, raw in flat.items():
            if "." in key:
                spec_name, _, field_name = key.partition(".")
                if spec_name not in specs:
                    raise ValueError(
                        f"unknown spec {spec_name!r} in {key!r}; "
                        f"expected one of {sorted(specs)}"
                    )
                if field_name not in _SPEC_FIELDS[spec_name]:
                    raise ValueError(f"unknown field {key!r}")
            else:
                candidates = [
                    name for name, names in _SPEC_FIELDS.items() if key in names
                ]
                if not candidates:
                    raise ValueError(f"unknown field {key!r}")
                if len(candidates) > 1:
                    raise ValueError(
                        f"ambiguous field {key!r} exists in {candidates}; "
                        f"qualify it as '<spec>.{key}'"
                    )
                spec_name, field_name = candidates[0], key
            patches[spec_name][field_name] = _coerce(spec_name, field_name, raw)

        updated = {
            name: (replace(spec, **patches[name]) if patches[name] else spec)
            for name, spec in specs.items()
        }
        return replace(self, **updated)
