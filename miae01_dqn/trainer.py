"""Step-driven training loop. Implementation: issue I-06.

One environment, one agent, one replay buffer. The loop is the one specified in
``docs/architecture.md`` section 4, with the counting units of section 3: the
loop counts *agent steps*, one buffer push per step, one update every
``learn_every_steps`` steps, one target sync every ``target_update_every_updates``
updates.

Invariants enforced here:

* every env step produces exactly one ``buffer.push`` - warm-up steps included,
  no transition is dropped;
* ``done`` is ``terminated`` only. A time-limit truncation is *not* a terminal
  state, so the TD target keeps bootstrapping from the last observation;
* the successor stored for a truncated step is ``info["final_observation"]``
  when the environment provides it (auto-reset environments return the first
  observation of the next episode as ``next_obs``), otherwise ``next_obs``,
  which is what a plain single env returns;
* episode returns are logged raw and un-normalised.

Logging goes through ``runner.Logger.scalar(tag, value, step)``; the tags are
exactly the ones promised in the README: ``episodic_return``, ``epsilon``,
``td_loss``, ``q_values`` and ``SPS``.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from miae01_dqn.agent import Agent
from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import RunConfig
from miae01_dqn.envs import EnvFactory

if TYPE_CHECKING:
    from miae01_dqn.runner import Logger


class TrainResult(NamedTuple):
    agent_steps: int  # steps actually executed (equals total_agent_steps)
    frames: int  # agent steps x action repeat
    updates: int  # gradient updates performed
    episodes: int  # episodes that ended inside the budget
    wall_clock_s: float
    final_return_mean: float  # mean of the last 100 episode returns


class Trainer:
    """Runs the training loop and returns a :class:`TrainResult`.

    ``log_every_steps`` (agent steps) controls the periodic scalars. The optional
    ``on_eval`` hook fires every ``cfg.eval.every_frames`` frames (converted to
    agent steps), so the caller can evaluate mid-training without the trainer
    knowing what an evaluator is; it must not touch the env, agent state or
    buffer.
    """

    def __init__(
        self,
        cfg: RunConfig,
        agent: Agent,
        env_factory: EnvFactory,
        buffer: ReplayBuffer,
        logger: "Logger",
        *,
        log_every_steps: int = 1000,
        on_eval: Callable[[int], None] | None = None,
    ) -> None:
        if log_every_steps < 1:
            raise ValueError(f"log_every_steps must be >= 1, got {log_every_steps}")
        self._cfg = cfg
        self._agent = agent
        self._env_factory = env_factory
        self._buffer = buffer
        self._logger = logger
        self._log_every_steps = int(log_every_steps)
        self._on_eval = on_eval

    def run(self) -> TrainResult:
        """Run ``cfg.algo.total_agent_steps`` steps and report the run summary."""
        cfg, algo = self._cfg, self._cfg.algo
        agent, buffer, logger = self._agent, self._buffer, self._logger
        eval_every = max(1, cfg.eval.every_frames // algo.action_repeat)

        env = self._env_factory.make_train(cfg.run.seed)
        returns: deque[float] = deque(maxlen=100)
        steps = 0
        updates = 0
        episodes = 0
        episode_return = 0.0
        last_loss: float | None = None
        started = time.perf_counter()
        last_step, last_time = 0, started

        try:
            observation, _ = env.reset()
            for step in range(algo.total_agent_steps):
                epsilon = agent.epsilon_at(step, algo)
                action = agent.act(observation, epsilon)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Truncation cuts the episode but not the return: the target must
                # still bootstrap, while the stored successor must be the true
                # last frame of the episode, not the first frame after reset.
                done = bool(terminated)
                stored_next = (
                    info.get("final_observation", next_obs) if truncated else next_obs
                )
                buffer.push(observation, action, float(reward), stored_next, done)
                episode_return += float(reward)
                steps = step + 1

                if (
                    step >= algo.replay_start_steps
                    and step % algo.learn_every_steps == 0
                ):
                    loss = agent.update()
                    if loss is not None:
                        updates += 1
                        last_loss = float(loss)
                        if updates % algo.target_update_every_updates == 0:
                            agent.sync_target()

                if terminated or truncated:
                    episodes += 1
                    returns.append(episode_return)
                    logger.scalar("episodic_return", episode_return, steps)
                    observation, _ = env.reset()
                    episode_return = 0.0
                else:
                    observation = next_obs

                if steps % self._log_every_steps == 0:
                    now = time.perf_counter()
                    elapsed = now - last_time
                    logger.scalar("epsilon", float(epsilon), steps)
                    logger.scalar(
                        "q_values", float(np.max(agent.q_values(observation))), steps
                    )
                    logger.scalar(
                        "SPS", (steps - last_step) / elapsed if elapsed > 0 else 0.0, steps
                    )
                    if last_loss is not None:
                        logger.scalar("td_loss", last_loss, steps)
                    last_step, last_time = steps, now

                if self._on_eval is not None and steps % eval_every == 0:
                    self._on_eval(steps)
        finally:
            env.close()

        return TrainResult(
            agent_steps=steps,
            frames=steps * algo.action_repeat,
            updates=updates,
            episodes=episodes,
            wall_clock_s=time.perf_counter() - started,
            final_return_mean=float(np.mean(returns)) if returns else 0.0,
        )
