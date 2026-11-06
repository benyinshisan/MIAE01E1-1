"""Smoke tests: the package imports and the paper defaults are in place."""



def test_package_imports():
    import miae01_dqn

    assert miae01_dqn.__version__ == "0.1.0"


def test_paper_defaults():
    from miae01_dqn import AlgoSpec, EnvSpec, OptimSpec

    env, optim, algo = EnvSpec(), OptimSpec(), AlgoSpec()
    assert env.env_id == "ALE/Pong-v5"
    assert env.frameskip == 1
    assert (optim.kind, optim.lr, optim.td_error_clip) == ("rmsprop", 2.5e-4, 1.0)
    assert (algo.gamma, algo.batch_size) == (0.99, 32)
    assert algo.target_update_every_updates == 10_000
    assert algo.eps_decay_frames == 1_000_000


def test_entry_point_exists():
    import importlib

    mod = importlib.import_module("miae01_dqn.runner")
    assert hasattr(mod, "main")
