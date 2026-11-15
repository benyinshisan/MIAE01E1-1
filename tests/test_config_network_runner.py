"""Issue #9/#13 regression coverage; not a substitute for non-author review."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import torch
from miae01_dqn.config import RunConfig
from miae01_dqn.network import QNetwork
from miae01_dqn.runner import build_config


def test_full_defaults_and_overrides():
    cfg=RunConfig()
    assert asdict(cfg.optim)==dict(kind='rmsprop',lr=.00025,momentum=.95,alpha=.95,
        eps=.01,td_error_clip=1.0,grad_clip_norm=None)
    assert (cfg.algo.gamma,cfg.algo.batch_size,cfg.algo.replay_start_frames,
            cfg.algo.learn_every_steps,cfg.algo.target_update_every_updates,
            cfg.algo.eps_decay_frames)==(.99,32,50000,4,10000,1000000)
    assert (cfg.eval.episodes,cfg.eval.epsilon,cfg.eval.max_episode_frames)==(30,.05,18000)
    assert cfg.with_overrides(gamma=.9)==cfg.with_overrides(**{'algo.gamma':.9})
    with pytest.raises(ValueError):cfg.with_overrides(typo=1)
    with pytest.raises(AssertionError):cfg.with_overrides(**{'algo.action_repeat':2})
    assert cfg.with_overrides(grad_clip_norm='2.0').optim.grad_clip_norm==2
    assert cfg.with_overrides(grad_clip_norm='null').optim.grad_clip_norm is None
    with pytest.raises(ValueError):cfg.with_overrides(batch_size=1.5)


def test_network_contract():
    torch.set_num_threads(1)
    network=QNetwork((4,84,84),6)
    x=torch.randint(0,256,(2,4,84,84),dtype=torch.uint8)
    y=network(x)
    assert y.shape==(2,6)
    torch.testing.assert_close(y,network(x.float()),rtol=0,atol=0)
    assert sum(p.numel() for p in network.parameters())==1687206
    y.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters())


def test_build_config_cli_yaml_and_names(tmp_path):
    config=tmp_path/'run.yaml'
    config.write_text('algo:\n  replay_capacity: 9000\nrun:\n  seed: 3\n')
    cfg=build_config({'config':config,'seed':2,'replay_capacity':7000,'smoke':True})
    assert cfg.run.seed==2 and cfg.algo.replay_capacity==7000
    assert cfg.algo.total_agent_steps==5000
    assert 'seed2__cap7000__' in cfg.run.run_name
    with pytest.raises(ValueError):build_config({'bogus':1})
    with pytest.raises(ValueError):build_config({'run_name':'..'})


def test_no_fixed_machine_paths_in_production_code():
    root=Path(__file__).resolve().parents[1]/'miae01_dqn'
    import re
    for path in root.glob('*.py'):
        assert not re.search(r'[CDE]:[\\/]',path.read_text(encoding='utf-8-sig')),path


def test_real_runner_artifacts_and_overwrite_protection(tmp_path):
    import json
    from miae01_dqn.runner import run_experiment
    cfg=build_config({'total_agent_steps':16,'replay_start_frames':8,'batch_size':2,
        'replay_capacity':32,'episodes':1,'max_episode_frames':100,
        'log_dir':tmp_path,'run_name':'tiny','device':'cpu','seed':3})
    result=run_experiment(cfg)
    assert result['status']=='complete' and result['train']['agent_steps']==16
    assert result['train']['updates']==3
    directory=tmp_path/'tiny'
    for name in ('config.json','result.json','environment.json','curves.png','metrics.jsonl'):
        assert (directory/name).is_file()
    assert list(directory.glob('events.out.tfevents.*'))
    saved=json.loads((directory/'config.json').read_text())
    assert saved['run']['seed']==3 and saved['algo']['replay_capacity']==32
    with pytest.raises(FileExistsError):run_experiment(cfg)
