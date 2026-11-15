"""Counting, truncation and evaluation isolation regression tests (#12/#13)."""
from types import SimpleNamespace
import numpy as np
import torch
from gymnasium import spaces
from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import RunConfig, EvalSpec
from miae01_dqn.trainer import Trainer
from miae01_dqn.evaluator import Evaluator
from test_agent import agent


def frame(value):
    return np.full((1,1,1),value,np.uint8)


class Env:
    action_space=spaces.Discrete(2)
    def __init__(self):
        self.closed=False
        self.resets=0
    def reset(self):
        self.resets+=1
        return frame(1),{}
    def step(self,action):
        return frame(9),1.0,False,True,{'final_observation':frame(5),'raw_reward':7.0}
    def close(self):self.closed=True


def test_trainer_truncation_updates_target_sync_and_eval_units():
    cfg=RunConfig().with_overrides(total_agent_steps=16,replay_start_frames=8,
        batch_size=2,learn_every_steps=2,target_update_every_updates=3,every_frames=16)
    env=Env()
    class Buffer:
        records=[]
        def push(self,*args,**kwargs):self.records.append((args,kwargs))
    class Agent:
        updates=syncs=0
        def act(self,*a):return 0
        def epsilon_at(self,*a):return .5
        def update(self):self.updates+=1;return 1.0
        def sync_target(self):self.syncs+=1
        def q_values(self,*a):return np.zeros(2)
    logs=[]
    logger=SimpleNamespace(scalar=lambda *a:logs.append(a))
    a,b=Agent(),Buffer()
    eval_steps=[]
    result=Trainer(cfg,a,SimpleNamespace(make_train=lambda seed:env),b,logger,
                   log_every_steps=4,on_eval=eval_steps.append).run()
    assert result.agent_steps==16 and result.frames==64 and result.episodes==16
    assert result.updates==7 and a.syncs==2
    assert eval_steps==[4,8,12,16]
    assert env.resets==1 and env.closed  # same-step autoreset is not reset twice
    assert len(b.records)==16
    for args,kwargs in b.records:
        np.testing.assert_array_equal(args[3],frame(5))
        assert args[4] is False and kwargs['truncated'] is True
    assert all(value==7 for tag,value,step in logs if tag=='episodic_return')


def test_truncated_successor_survives_reset_and_ring_eviction():
    b=ReplayBuffer(4,(1,1,1),seed=1)
    b.push(frame(1),0,0,frame(5),False,truncated=True)
    b.push(frame(9),1,0,frame(10),False)
    seen=False
    for _ in range(20):
        batch=b.sample(2)
        for obs,nxt in zip(batch.obs,batch.next_obs):
            if obs.item()==1:
                seen=True
                assert nxt.item()==5
    assert seen
    for i in range(10,16):b.push(frame(i),1,0,frame(i+1),False)
    for _ in range(20):
        batch=b.sample(4)
        assert np.all(batch.next_obs==batch.obs+1)


def test_evaluation_preserves_actual_agent_parameters_optimizer_rng_and_buffer():
    a=agent()
    a.buffer.push(frame(1),0,1,frame(1),True)
    a.update()
    before=a.state_dict()
    class EvalEnv(Env):
        def step(self,action):return frame(1),2.0,True,False,{}
    result=Evaluator(a,SimpleNamespace(make_eval=lambda seed:EvalEnv()),EvalSpec(),seed=3).evaluate(40)
    after=a.state_dict()
    assert len(result.scores)==30 and result.mean==2 and len(a.buffer)==1
    assert before['rng']==after['rng']
    for group in ('policy','target'):
        for key in before[group]:torch.testing.assert_close(before[group][key],after[group][key],rtol=0,atol=0)
    for param,values in before['optimizer']['state'].items():
        for key,value in values.items():
            torch.testing.assert_close(value,after['optimizer']['state'][param][key],rtol=0,atol=0)
