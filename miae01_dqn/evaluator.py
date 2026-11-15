"""Independent evaluation on a separate environment. Implementation: issue I-07."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import torch

from miae01_dqn.agent import Agent
from miae01_dqn.config import EvalSpec
from miae01_dqn.envs import EnvFactory


class EvalResult(NamedTuple):
    at_frame: int
    scores: Sequence[float]
    mean: float
    std: float
    random_score: float | None = None
    human_score: float | None = None

    @property
    def normalized(self) -> float:
        if self.random_score is None or self.human_score is None:
            raise ValueError("normalization requires random_score and human_score")
        if self.human_score == self.random_score:
            raise ValueError("human_score must differ from random_score")
        return 100.0 * (self.mean - self.random_score) / (
            self.human_score - self.random_score
        )


class Evaluator:
    """Evaluate without consuming the training agent's exploration RNG.

    ``make_eval`` supplies fresh no-op starts and unclipped rewards. Exploration
    uses a private generator; only ``act_greedy`` is called on the agent.
    Baselines are optional, game-specific caller inputs, never Pong constants.

    ``action_repeat`` must match the environment (4 under the project contract).
    On ALE, the episode frame counter includes reset/no-op/FIRE frames. Stop
    before a full repeated action would exceed the limit, leaving at most
    ``action_repeat - 1`` unused frames. Other EnvHandle implementations use
    decision count times action_repeat, since they expose no raw frame counter.
    """
    def __init__(
        self,
        agent: Agent,
        env_factory: EnvFactory,
        spec: EvalSpec,
        *,
        seed: int | None = None,
        random_score: float | None = None,
        human_score: float | None = None,
        action_repeat: int = 4,
    ) -> None:
        if (random_score is None) != (human_score is None):
            raise ValueError("provide both random_score and human_score, or neither")
        if random_score is not None:
            if not np.isfinite([random_score, human_score]).all():
                raise ValueError("normalization baselines must be finite")
            if human_score == random_score:
                raise ValueError("human_score must differ from random_score")
        if not isinstance(action_repeat, int) or isinstance(action_repeat, bool):
            raise TypeError("action_repeat must be an integer")
        if action_repeat < 1 or action_repeat > spec.max_episode_frames:
            raise ValueError("action_repeat must be within the episode frame budget")
        self._agent = agent
        self._env_factory = env_factory
        self._spec = spec
        self._rng = np.random.default_rng(seed)
        self._random_score = random_score
        self._human_score = human_score
        self._action_repeat = action_repeat

    def evaluate(self, at_frame: int) -> EvalResult:
        """Return per-game raw returns, population mean/std, and frame label.

        A fresh environment is owned and closed by each call, including when
        action selection or stepping fails. Time-limit truncation ends a game;
        life loss alone does not, because make_eval omits EpisodicLife.
        """
        env_seed = int(self._rng.integers(0, 2**31))
        env = self._env_factory.make_eval(env_seed)
        scores: list[float] = []
        try:
            ale = getattr(getattr(env, "unwrapped", None), "ale", None)
            frame_counter = getattr(ale, "getEpisodeFrameNumber", None)
            with torch.inference_mode():
                for _ in range(self._spec.episodes):
                    obs, _ = env.reset()
                    frames = int(frame_counter()) if frame_counter else 0
                    if frames + self._action_repeat > self._spec.max_episode_frames:
                        raise ValueError("episode frame budget is exhausted during reset")
                    score = 0.0
                    while frames + self._action_repeat <= self._spec.max_episode_frames:
                        if self._rng.random() < self._spec.epsilon:
                            action = int(self._rng.integers(env.action_space.n))
                        else:
                            action = self._agent.act_greedy(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        score += float(reward)
                        next_frames = (
                            int(frame_counter()) if frame_counter
                            else frames + self._action_repeat
                        )
                        if not 0 < next_frames - frames <= self._action_repeat:
                            raise ValueError("ALE frame advance does not match action_repeat")
                        frames = next_frames
                        if terminated or truncated:
                            break
                    scores.append(score)
        finally:
            env.close()
        return EvalResult(
            at_frame, tuple(scores), float(np.mean(scores)), float(np.std(scores)),
            self._random_score, self._human_score,
        )
