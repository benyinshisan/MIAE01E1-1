"""CLI entry point and experiment assembly. Implementation: issue I-08."""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Protocol

from miae01_dqn.config import RunConfig


class Logger(Protocol):
    def scalar(self, tag: str, value: float, step: int) -> None: ...
    def text(self, tag: str, text: str) -> None: ...
    def close(self) -> None: ...


def build_config(cli_or_yaml: Mapping[str, Any]) -> RunConfig:
    raise NotImplementedError("I-08: build_config")


def run_experiment(cfg: RunConfig) -> dict[str, Any]:
    raise NotImplementedError("I-08: run_experiment")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="miae01_dqn.runner",
        description="Run the DQN replication (see README).",
    )
    parser.add_argument("--smoke", action="store_true", help="short smoke run (5k steps)")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--config", default=None, help="path to a YAML/JSON config")
    args = parser.parse_args(argv)

    cfg = build_config(vars(args))
    run_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
