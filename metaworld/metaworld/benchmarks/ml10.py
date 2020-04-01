
from metaworld.benchmarks.base import Benchmark
from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS, MEDIUM_MODE_CLS_DICT


class AncientML10(MultiClassMultiTaskEnv, Benchmark, Serializable):

    def __init__(self, env_type='train', sample_all=False):
        assert env_type == 'train' or env_type == 'test'
        Serializable.quick_init(self, locals())

        cls_dict = MEDIUM_MODE_CLS_DICT[env_type]
        args_kwargs = MEDIUM_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)

class OldML10(MultiClassMultiTaskEnv, Benchmark, Serializable): #! randomize goals when calling reset_task, one task has one replay buffer

    def __init__(self, sample_all=False, seed=0):
        task_names = [
            'reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1', 'drawer-close-v1', \
            'button-press-topdown-v1', 'peg-insert-side-v1', 'window-open-v1', 'sweep-v1', 'basketball-v1', \
            'drawer-open-v1', 'door-close-v1', 'shelf-place-v1', 'sweep-into-v1', 'lever-pull-v1'
        ]
        Serializable.quick_init(self, locals())

        medium_cls_dict, medium_args_kwargs = {}, {}
        for task in task_names:
            if task in MEDIUM_MODE_CLS_DICT['train']:
                medium_cls_dict.update({task: MEDIUM_MODE_CLS_DICT['train'][task]})
                medium_args_kwargs.update({task: MEDIUM_MODE_ARGS_KWARGS['train'][task]})
            else:
                medium_cls_dict.update({task: MEDIUM_MODE_CLS_DICT['test'][task]})
                medium_args_kwargs.update({task: MEDIUM_MODE_ARGS_KWARGS['test'][task]})

        super().__init__(
            task_env_cls_dict=medium_cls_dict,
            task_args_kwargs=medium_args_kwargs,
            sample_goals=False,
            obs_type='plain',
            sample_all=sample_all)
        self.reorder_env(task_names)
        self.seed(seed)
        self.n_train_goals = 10
        self.n_test_goals = 5

    def reset_task(self, task_id):
        self.set_task(task_id)
        self.reset()

    @classmethod
    def get_all_tasks(cls, sample_all=False, seed=0):
        return cls(sample_all=sample_all, seed=seed)


class ML10(MultiClassMultiTaskEnv, Benchmark, Serializable): #! sampled a fixed set of goals prehand, one goal has one replay buffer

    def __init__(self, n_train_goals=30, n_test_goals=10, sample_all=False, seed=0):
        # assert env_type == 'train' or env_type == 'test'
        task_names = [
            'reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1', 'drawer-close-v1', \
            'button-press-topdown-v1', 'peg-insert-side-v1', 'window-open-v1', 'sweep-v1', 'basketball-v1', \
            'drawer-open-v1', 'door-close-v1', 'shelf-place-v1', 'sweep-into-v1', 'lever-pull-v1'
        ]
        # n_goals = n_train_goals + n_test_goals
        Serializable.quick_init(self, locals())

        medium_cls_dict, medium_args_kwargs = {}, {}
        for task in task_names:
            if task in MEDIUM_MODE_CLS_DICT['train']:
                medium_cls_dict.update({task: MEDIUM_MODE_CLS_DICT['train'][task]})
                medium_args_kwargs.update({task: MEDIUM_MODE_ARGS_KWARGS['train'][task]})
            else:
                medium_cls_dict.update({task: MEDIUM_MODE_CLS_DICT['test'][task]})
                medium_args_kwargs.update({task: MEDIUM_MODE_ARGS_KWARGS['test'][task]})
        for task in medium_args_kwargs:
            medium_args_kwargs[task]['kwargs']['random_init'] = False
        # medium_cls_dict = MEDIUM_MODE_CLS_DICT['train']
        # medium_cls_dict.update(MEDIUM_MODE_CLS_DICT['test'])
        # medium_args_kwargs = MEDIUM_MODE_ARGS_KWARGS['train']
        # medium_args_kwargs.update(MEDIUM_MODE_ARGS_KWARGS['test'])
        # cls_dict = MEDIUM_MODE_CLS_DICT['train']
        # args_kwargs = MEDIUM_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=medium_cls_dict,
            task_args_kwargs=medium_args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)
        self.reorder_env(task_names)
        self.seed(seed)
        goals = self._task_envs[0].sample_goals_(n_train_goals*10+n_test_goals*5)
        # print (goals[:10])
        # for i in range(15):
        #     print (self._task_envs[0].goal_space)
        # exit(-1)
        # for i in range(10):
        #     goals.append(self._task_envs[i].sample_goals_(n_train_goals))
        # for i in range(10, 15):
        #     goals.append(self._task_envs[i].sample_goals_(n_test_goals))
        # for i in range(15):
        #     print (goals[i])
        goal_dict = {}
        for i in range(10):
            # print (goals[i*n_train_goals:(i+1)*n_train_goals])
            goal_dict.update({task_names[i]: goals[i*n_train_goals:(i+1)*n_train_goals]})
        for i in range(10, 15):
            # print (goals[10*n_train_goals+(i-10)*n_test_goals:10*n_train_goals+(i-9)*n_test_goals])
            goal_dict.update({task_names[i]: goals[10*n_train_goals+(i-10)*n_test_goals:10*n_train_goals+(i-9)*n_test_goals]})
        self.discretize_goal_space(goal_dict, seed)
        self.n_train_goals = n_train_goals*10
        self.n_test_goals = n_test_goals*5
        self.n_train_goals_per_task = n_train_goals
        self.n_test_goals_per_task = n_test_goals
        # self.ML1 = False
        # self.ML10 = True
        # self.OldML10 = False
        self.setup_iter()
        # self.setup_ML10_test_goals()

    def reset_task(self, goal_id):
        # assert len(self._task_envs) == 1
        assert goal_id < self.n_train_goals + self.n_test_goals
        if goal_id >= self.n_train_goals:
            task_id = int((goal_id-self.n_train_goals)//self.n_test_goals_per_task)
            self.set_task({'task':10+task_id, 'goal':goal_id-(self.n_train_goals+task_id*self.n_test_goals_per_task)})
        else:
            task_id = int(goal_id//self.n_train_goals_per_task)
            self.set_task({'task':task_id, 'goal':goal_id-task_id*self.n_train_goals_per_task})
        self.reset()

    @classmethod
    def get_all_tasks(cls, n_train_goals=5, n_test_goals=2, sample_all=False, seed=0):
        return cls(n_train_goals=n_train_goals, n_test_goals=n_test_goals, sample_all=sample_all, seed=seed)


class PaperML10(MultiClassMultiTaskEnv, Benchmark, Serializable): #! sampled a fixed set of goals prehand, one goal has one replay buffer

    def __init__(self, n_train_goals=30, n_test_goals=10, sample_all=False, seed=0):
        # assert env_type == 'train' or env_type == 'test'
        task_names = [
            'reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1', 'drawer-close-v1', \
            'button-press-topdown-v1', 'peg-insert-side-v1', 'window-open-v1', 'sweep-v1', 'basketball-v1', \
            'drawer-open-v1', 'door-close-v1', 'shelf-place-v1', 'sweep-into-v1', 'lever-pull-v1'
        ]
        # n_goals = n_train_goals + n_test_goals
        Serializable.quick_init(self, locals())

        medium_cls_dict, medium_args_kwargs = {}, {}
        for task in task_names:
            if task in MEDIUM_MODE_CLS_DICT['train']:
                medium_cls_dict.update({task: MEDIUM_MODE_CLS_DICT['train'][task]})
                medium_args_kwargs.update({task: MEDIUM_MODE_ARGS_KWARGS['train'][task]})
            else:
                medium_cls_dict.update({task: MEDIUM_MODE_CLS_DICT['test'][task]})
                medium_args_kwargs.update({task: MEDIUM_MODE_ARGS_KWARGS['test'][task]})
        # medium_cls_dict = MEDIUM_MODE_CLS_DICT['train']
        # medium_cls_dict.update(MEDIUM_MODE_CLS_DICT['test'])
        # medium_args_kwargs = MEDIUM_MODE_ARGS_KWARGS['train']
        # medium_args_kwargs.update(MEDIUM_MODE_ARGS_KWARGS['test'])
        # cls_dict = MEDIUM_MODE_CLS_DICT['train']
        # args_kwargs = MEDIUM_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=medium_cls_dict,
            task_args_kwargs=medium_args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)
        self.reorder_env(task_names)
        self.seed(seed)
        import pdb
        pdb.set_trace()
        goals = self._task_envs[0].sample_goals_(n_train_goals*10+n_test_goals*5)
        # print (goals[:10])
        # for i in range(15):
        #     print (self._task_envs[0].goal_space)
        # exit(-1)
        # for i in range(10):
        #     goals.append(self._task_envs[i].sample_goals_(n_train_goals))
        # for i in range(10, 15):
        #     goals.append(self._task_envs[i].sample_goals_(n_test_goals))
        # for i in range(15):
        #     print (goals[i])
        goal_dict = {}
        for i in range(10):
            # print (goals[i*n_train_goals:(i+1)*n_train_goals])
            goal_dict.update({task_names[i]: goals[i*n_train_goals:(i+1)*n_train_goals]})
        for i in range(10, 15):
            # print (goals[10*n_train_goals+(i-10)*n_test_goals:10*n_train_goals+(i-9)*n_test_goals])
            goal_dict.update({task_names[i]: goals[10*n_train_goals+(i-10)*n_test_goals:10*n_train_goals+(i-9)*n_test_goals]})
        self.discretize_goal_space(goal_dict, seed)
        self.n_train_goals = n_train_goals*10
        self.n_test_goals = n_test_goals*5
        self.n_train_goals_per_task = n_train_goals
        self.n_test_goals_per_task = n_test_goals
        # self.ML1 = False
        # self.ML10 = True
        # self.OldML10 = False
        self.setup_iter()
        # self.setup_ML10_test_goals()

    def reset_task(self, goal_id):
        # assert len(self._task_envs) == 1
        assert goal_id < self.n_train_goals + self.n_test_goals
        if goal_id >= self.n_train_goals:
            task_id = int((goal_id-self.n_train_goals)//self.n_test_goals_per_task)
            self.set_task({'task':10+task_id, 'goal':goal_id-(self.n_train_goals+task_id*self.n_test_goals_per_task)})
        else:
            task_id = int(goal_id//self.n_train_goals_per_task)
            self.set_task({'task':task_id, 'goal':goal_id-task_id*self.n_train_goals_per_task})
        self.reset()

    @classmethod
    def get_all_tasks(cls, n_train_goals=5, n_test_goals=2, sample_all=False, seed=0):
        return cls(n_train_goals=n_train_goals, n_test_goals=n_test_goals, sample_all=sample_all, seed=seed)
