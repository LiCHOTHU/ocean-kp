import numpy as np
from gym import spaces
from gym import Env

from . import register_env

import os

@register_env('point-robot')
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, randomize_tasks=False, n_tasks=2, seed=0):
        self.seed(seed)
        if randomize_tasks:
            # np.random.seed(1337)
            goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(n_tasks)]
        else:
            # some hand-coded goals for debugging
            goals = [np.array([10, -10]),
                     np.array([10, 10]),
                     np.array([-10, 10]),
                     np.array([-10, -10]),
                     np.array([0, 0]),

                     np.array([7, 2]),
                     np.array([0, 4]),
                     np.array([-6, 9])
                     ]
            goals = [g / 10. for g in goals]
        self.goals = goals

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)


@register_env('sparse-point-robot')
class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2, seed=0):
        super().__init__(randomize_tasks, n_tasks, seed)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angles = np.linspace(0, np.pi, num=n_tasks)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d

@register_env('sparse-dirichlet-point-robot')
class SparseDirichletPointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2, seed=0, alpha=0.5):
        super().__init__(randomize_tasks, n_tasks, seed)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angle1 = np.pi/8
            angle2 = 7*np.pi/8
            dirichlets = np.random.dirichlet([alpha]*2, n_tasks)
            angles = []
            for diri in dirichlets:
                angles.append(angle1*diri[0]+angle2*diri[1])
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()
                
        # if randomize_tasks:
        #     np.random.seed(1337)
        #     radius = 1.0
        #     angles = np.linspace(0, np.pi, num=n_tasks)
        #     xs = radius * np.cos(angles)
        #     ys = radius * np.sin(angles)
        #     goals = np.stack([xs, ys], axis=1)
        #     np.random.shuffle(goals)
        #     goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d

@register_env('sparse-direction-point-robot')
class SparseDirectionPointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, n_train_tasks=20, n_test_tasks=12, goal_radius=0.2, seed=0):
        super().__init__(randomize_tasks, n_tasks, seed)
        self.goal_radius = goal_radius
        assert n_tasks == n_train_tasks + n_test_tasks, "n_tasks != n_train_tasks + n_test_tasks"
        assert n_train_tasks % 4 == 0
        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angle_init = [0, np.pi/2, np.pi, 3*np.pi/2]
            angle_test = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
            delta = np.pi/12
            angles = []
            for i in angle_init:
                angles.extend(np.linspace(i-delta, i+delta, num=n_train_tasks//len(angle_init)))
            for i in angle_test:
                angles.extend(np.linspace(i-delta, i+delta, num=n_test_tasks//len(angle_test)))

            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            # np.random.shuffle(goals)
            goals = goals.tolist()
                
        # if randomize_tasks:
        #     np.random.seed(1337)
        #     radius = 1.0
        #     angles = np.linspace(0, np.pi, num=n_tasks)
        #     xs = radius * np.cos(angles)
        #     ys = radius * np.sin(angles)
        #     goals = np.stack([xs, ys], axis=1)
        #     np.random.shuffle(goals)
        #     goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d

@register_env('categorical-point-robot')
class CategoricalPointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2, seed=0, n_categories=2):
        super().__init__(randomize_tasks, n_tasks, seed)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            if n_categories == 1:
                angles = np.linspace(0, np.pi, num=n_tasks)
            else:
                assert n_tasks % n_categories == 0, "n tasks should be multiply of n categories"
                angles = np.array([])
                for category in range(n_categories):
                    angles = np.concatenate((angles, np.linspace((category*2+1)/(n_categories*2+1)*np.pi, (category*2+2)/(n_categories*2+1)*np.pi, num=int(n_tasks/n_categories))), axis=0)

            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()


        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d

        
@register_env('two-categorical-point-robot')
class TwoCategoricalPointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2, seed=0, n_categories=2):
        super().__init__(randomize_tasks, n_tasks, seed)
        self.goal_radius = goal_radius
        n_tasks *= 2

        if randomize_tasks:
            np.random.seed(1337)
            radius = [0.5, 1.0]
            if n_categories == 1:
                angles = [np.linspace(0, 2*np.pi, num=n_tasks//2), np.linspace(0, 2*np.pi, num=n_tasks//2)]
                
            else:
                assert (n_tasks // 2) % n_categories == 0, "n tasks should be multiply of n categories"
                angles = []
                for i in range(2):
                    angle = np.array([])
                    for category in range(n_categories):
                        angle = np.concatenate((angle, np.linspace((category*2+i)/(n_categories*2)*np.pi*2, (category*2+1+i)/(n_categories*2)*np.pi*2, num=int(n_tasks/n_categories))), axis=0)
                    angles.append(angle)

            xs = np.concatenate([radius[0] * np.cos(angles[0]), radius[1] * np.cos(angles[1])], axis=0)
            ys = np.concatenate([radius[0] * np.sin(angles[0]), radius[1] * np.sin(angles[1])], axis=0)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
