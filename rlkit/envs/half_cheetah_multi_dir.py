import numpy as np

from .half_cheetah import HalfCheetahEnv
from . import register_env


@register_env('cheetah-multi-dir')
class HalfCheetahMultiDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a reward equal to its
    velocity in the target direction. The tasks are generated by sampling the
    target directions from a Bernoulli distribution on {-1, 1} with parameter
    0.5 (-1: backward, +1: forward).

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self, task={}, n_tasks=2, randomize_tasks=False, n_dirs=3, max_eps=700, seed=0):
        directions = [-1, 1]
        self._max_eps = max_eps
        self.tasks = self.sample_tasks(n_tasks, n_dirs)
        self._num_steps = 0
        self._task = task
        self._goal_dirs = self.tasks[0]['dir']
        self._goal_steps = self.tasks[0]['step']
        self._goal_dir = self.tasks[0].get('dir', [1])[0]
        self._goal = self._goal_dir
        # self._goal_dir = task.get('direction', 1)
        # self._goal = self._goal_dir
        super(HalfCheetahMultiDirEnv, self).__init__()
        self.seed(seed)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        self._num_steps += 1

        self._goal_dir = self._goal_dirs[np.searchsorted(self._goal_steps, self._num_steps)]
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks, num_dirs):
        directions = 2 * np.random.binomial(1, p=0.5, size=(num_tasks, num_dirs)) - 1
        change_steps = np.sort(np.array([self._max_eps * i / num_dirs for i in range(1, num_dirs)]) + np.random.uniform(-0.05*self._max_eps, 0.05*self._max_eps, size=(num_tasks, num_dirs - 1)))
        tasks = []
        for i in range(num_tasks):
            tasks.append({'dir': directions[i], 'step': change_steps[i]})
        # tasks = [{'direction': direction} for direction in directions]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._num_steps = 0
        self._goal_steps = self._task['step']
        self._goal_dirs = self._task['dir']
        self._goal_dir = self._goal_dirs[np.searchsorted(self._goal_steps, self._num_steps)]
        self._goal = self._goal_dir
        # self._goal_dir = self._task['direction']
        # self._goal = self._goal_dir
        self.reset()

    def save_all_tasks(self, save_dir):
        import pickle
        import os
        with open(os.path.join(save_dir, 'goals.pkl'), 'wb') as f:
            pickle.dump(self.tasks, f)