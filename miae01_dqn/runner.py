"""CLI, logging and experiment assembly (Issue #8)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping

import numpy as np
import torch

from miae01_dqn.config import RunConfig


def write_json(path: Path, data: Any) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding='utf-8')
    # Windows readers/antivirus can briefly deny replacement of an open file.
    # Keep the old complete JSON visible and retry instead of aborting training.
    for attempt in range(100):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.05)


class Logger:
    def __init__(self, run_dir: Path):
        from torch.utils.tensorboard import SummaryWriter
        self.run_dir = Path(run_dir)
        self.writer = SummaryWriter(str(run_dir))
        self.file = (self.run_dir / 'metrics.jsonl').open('w', encoding='utf-8')
        self.started = time.perf_counter()

    def scalar(self, tag: str, value: float, step: int) -> None:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'non-finite metric {tag}')
        self.writer.add_scalar(tag, value, step)
        self.file.write(json.dumps({'tag': tag, 'value': value, 'step': step}) + '\n')
        if tag == 'SPS':
            import psutil
            memory = psutil.Process().memory_info().rss
            self.writer.add_scalar('rss_bytes', memory, step)
            write_json(self.run_dir / 'progress.json', {'agent_steps': step,
                       'sps': value, 'rss_bytes': memory,
                       'wall_clock_s': time.perf_counter() - self.started})
            self.file.flush()
            self.writer.flush()
            if step % 25000 == 0 or step == 5000:
                print(f'step={step} SPS={value:.1f} RSS={memory / 2**30:.2f} GiB', flush=True)

    def text(self, tag: str, text: str) -> None:
        self.writer.add_text(tag, text)

    def close(self) -> None:
        self.writer.close()
        self.file.close()


def build_config(cli_or_yaml: Mapping[str, Any]) -> RunConfig:
    values = dict(cli_or_yaml)
    config_path = values.pop('config', None)
    smoke = values.pop('smoke', False)
    flat = {}
    if config_path:
        import yaml
        loaded = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        if not isinstance(loaded, dict):
            raise ValueError('configuration file must contain a mapping')
        for key, value in loaded.items():
            if isinstance(value, dict):
                flat.update({f'{key}.{k}': v for k, v in value.items()})
            else:
                flat[key] = value
    flat.update({k.replace('-', '_'): v for k, v in values.items() if v is not None})
    if smoke:
        # Explicit engineering profile: learn during smoke without allocating 5.6GB.
        flat.update({'algo.total_agent_steps': 5000, 'algo.replay_start_frames': 1000})
        if 'replay_capacity' not in flat and 'algo.replay_capacity' not in flat:
            flat['algo.replay_capacity'] = 5000
        if 'episodes' not in flat and 'eval.episodes' not in flat:
            flat['eval.episodes'] = 2
    cfg = RunConfig().with_overrides(**flat)
    if cfg.algo.replay_capacity < 2:
        raise ValueError('replay capacity must be >= 2')
    if cfg.env.frameskip != 1 or not cfg.env.max_and_skip:
        raise ValueError('runner requires ALE frameskip=1 and MaxAndSkip enabled')
    if not cfg.run.run_name:
        name = re.sub(r'[^A-Za-z0-9_.-]', '-', cfg.env.env_id)
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        cfg = cfg.with_overrides(run_name=f'{name}__seed{cfg.run.seed}__cap{cfg.algo.replay_capacity}__{ts}')
    if cfg.run.run_name in ('.', '..'):
        raise ValueError('run_name must identify a child directory')
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def run_experiment(cfg: RunConfig) -> dict[str, Any]:
    from miae01_dqn.agent import Agent
    from miae01_dqn.buffer import ReplayBuffer
    from miae01_dqn.envs import EnvFactory
    from miae01_dqn.evaluator import Evaluator
    from miae01_dqn.network import QNetwork
    from miae01_dqn.trainer import Trainer

    if cfg.run.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    seed_everything(cfg.run.seed)
    run_dir = cfg.run.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / 'config.json', asdict(cfg))
    import gymnasium, ale_py, psutil
    write_json(run_dir / 'environment.json', {
        'torch': torch.__version__, 'numpy': np.__version__, 'gymnasium': gymnasium.__version__,
        'ale': ale_py.__version__, 'device': cfg.run.device,
        'device_name': torch.cuda.get_device_name() if cfg.run.device.startswith('cuda') else 'cpu',
        'available_memory_bytes': psutil.virtual_memory().available,
    })
    logger = Logger(run_dir)
    evaluations = []
    started = time.perf_counter()
    try:
        factory = EnvFactory(cfg.env)
        buffer = ReplayBuffer(cfg.algo.replay_capacity, factory.obs_shape(), seed=cfg.run.seed)
        agent = Agent(QNetwork(factory.obs_shape(), factory.n_actions()), buffer, cfg.algo,
                      cfg.optim, cfg.run.device, seed=cfg.run.seed)
        baselines = {'random_score': -20.7, 'human_score': 9.3} if cfg.env.env_id == 'ALE/Pong-v5' else {}
        evaluator = Evaluator(agent, factory, cfg.eval, seed=cfg.run.seed + 10000,
                              action_repeat=cfg.env.action_repeat, **baselines)

        def evaluate(agent_steps):
            result = evaluator.evaluate(agent_steps * cfg.algo.action_repeat)
            record = dict(result._asdict())
            if baselines:
                record['normalized'] = result.normalized
            evaluations.append(record)
            write_json(run_dir / 'evaluations.json', evaluations)
            logger.scalar('eval/mean', result.mean, result.at_frame)
            logger.scalar('eval/std', result.std, result.at_frame)
            print(f'evaluation frame={result.at_frame} mean={result.mean:.3f}', flush=True)

        trained = Trainer(cfg, agent, factory, buffer, logger, on_eval=evaluate).run()
        if not evaluations or evaluations[-1]['at_frame'] != trained.frames:
            evaluate(trained.agent_steps)
        result = {'status': 'complete', 'train': trained._asdict(), 'evaluation': evaluations[-1],
                  'wall_clock_s': time.perf_counter() - started,
                  'rss_bytes': psutil.Process().memory_info().rss,
                  'run_dir': str(run_dir), 'evaluations': evaluations}
        if cfg.run.save_checkpoints:
            torch.save(agent.state_dict(), run_dir / 'agent.pt')
        logger.file.flush()
        plot_run(run_dir)
        write_json(run_dir / 'result.json', result)
    except BaseException as exc:
        write_json(run_dir / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        logger.close()
    return result


def plot_run(run_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    records = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for axis, tag, label in zip(axes, ['episodic_return', 'td_loss'], ['Raw episode return', 'Mean TD Huber loss']):
        points = [r for r in records if r['tag'] == tag]
        axis.plot([r['step'] for r in points], [r['value'] for r in points], linewidth=1)
        axis.set(xlabel='Agent steps', ylabel=label)
        axis.grid(alpha=0.25)
    fig.savefig(run_dir / 'curves.png', dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Nature DQN replication')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--config')
    for name, kind in [('seed', int), ('replay-capacity', int), ('device', str),
                       ('total-agent-steps', int), ('log-dir', str), ('run-name', str),
                       ('episodes', int), ('replay-start-frames', int)]:
        parser.add_argument('--' + name, type=kind, default=None)
    parser.add_argument('--save-checkpoints', action='store_true', default=None)
    cfg = build_config(vars(parser.parse_args(argv)))
    result = run_experiment(cfg)
    print(json.dumps(result, default=str), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
