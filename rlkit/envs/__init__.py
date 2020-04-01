import os
import importlib


ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn


# automatically import any envs in the envs/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        if 'walker' in module or 'hopper' in module:
            continue
        # if 'ant' in module or 'hopper' in module or 'half_cheetah' in module or 'mujoco' in module or 'humanoid' in module or 'walker' in module:
        #     continue
        importlib.import_module('rlkit.envs.' + module)
