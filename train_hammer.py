"""
Launcher for experiments with PEARL
#! 7.6 change rl_algorithm replay_buffer_size from 1000000 to 100000
"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, SetEncoder, VRNNEncoder
from rlkit.torch.sac.tool_sac import PEARLSoftActorCritic
from rlkit.torch.sac.tool_agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.hammer_default import default_config
import random
from metaworld.benchmarks import ML1, ML10, ML45
from metaworld.benchmarks.ml10 import OldML10
import pickle

from dataset.dataset_hammer import Dataset
from libs.network import KeyNet
from libs.gnn import Net as GNN

def set_global_seeds(i):
    # try:
    #     import tensorflow as tf
    # except ImportError:
    #     pass
    # else:
    #     tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_dim(s):
    a, b, c, d, e = s.split('.')
    return [int(a), int(b), int(c), int(d), int(e)]

def gpu_optimizer(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

def experiment(variant):

    # create information for the dataset

    print(variant['dataset_params'])
    dataset = Dataset(**variant['dataset_params'])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # create multi-task environment and sample tasks
    print (variant['env_name'])
    # print (ENVS)
    print (variant['env_params'])
    # print (ENVS[variant['env_name']](**variant['env_params']))
    if variant['meta'] == 'meta':
        env = ML1.get_all_tasks(variant['task'], variant['n_train_tasks'], variant['n_eval_tasks'], seed=variant['seed'])
    elif variant['meta'] == 'meta10':
        assert variant['n_train_tasks'] % 10 == 0 and variant['n_eval_tasks'] % 5 == 0
        env = ML10.get_all_tasks(variant['n_train_tasks']//10, variant['n_eval_tasks']//5, seed=variant['seed'])
    elif variant['meta'] == 'oldmeta10':
        env = OldML10.get_all_tasks(seed=variant['seed'])
    else:
        print(variant['env_name'])
        print(variant['env_params'])


        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))

    env.set_kp_data(dataset)
    tasks = env.get_all_task_idx()

    obs_dim = 15

    #!! modify action space add if
    #!! action_type = env.action_space(type) something like that
    action_dim = int(np.prod(env.action_space.shape))

    # instantiate networks
    cont_latent_dim, num_cat, latent_dim, num_dir, dir_latent_dim = read_dim(variant['global_latent'])
    r_cont_dim, r_n_cat, r_cat_dim, r_n_dir, r_dir_dim = read_dim(variant['vrnn_latent'])
    # latent_dim = variant['latent_size']
    # num_cat = variant['num_cat']
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    glob = variant['algo_params']['glob']
    rnn = variant['rnn']
    vrnn_latent = variant['vrnn_latent']
    encoder_model = MlpEncoder # VRNNEncoder #RecurrentEncoder # if recurrent else MlpEncoder #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if recurrent:
        if variant['vrnn_constraint'] == 'logitnormal':
            output_size = r_cont_dim * 2 + r_n_cat * r_cat_dim + r_n_dir * r_dir_dim * 2
        else:
            output_size = r_cont_dim * 2 + r_n_cat * r_cat_dim + r_n_dir * r_dir_dim
        if variant['rnn_sample'] == 'batch_sampling':
            input_size = (obs_dim + action_dim + reward_dim) * variant['temp_res']
        else:
            input_size = (obs_dim + action_dim + reward_dim)
        if rnn == 'rnn':
            recurrent_model = RecurrentEncoder
            recurrent_context_encoder = recurrent_model(
                hidden_sizes=[net_size, net_size, net_size],
                input_size=input_size,
                output_size = output_size
            )
        elif rnn == 'vrnn':
            recurrent_model = VRNNEncoder
            recurrent_context_encoder = recurrent_model(
                hidden_sizes=[net_size, net_size, net_size],
                input_size=input_size,
                output_size=output_size,
                temperature=variant['temperature'],
                vrnn_latent=variant['vrnn_latent'],
                vrnn_constraint=variant['vrnn_constraint'],
                r_alpha=variant['vrnn_alpha'],
                r_var=variant['vrnn_var'],
            )

    else:
        recurrent_context_encoder = None
    # cont_latent_dim = variant['cont_latent_size']
    # dir_latent_dim = variant['dir_latent_size']
    # num_dir = variant['num_dir']
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if dir_latent_dim > 0 and variant['constraint'] == 'deepsets':
        raise Exception('Deprecated!')
        context_encoder2 = SetEncoder(
            hidden_sizes=[200, 200],
            input_size=obs_dim + action_dim + reward_dim,
            output_size=200,
            set_output_size=dir_latent_dim * num_dir,
            set_activation=torch.max
        )
        if latent_dim + cont_latent_dim > 0:
            context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=obs_dim + action_dim + reward_dim,
            output_size=latent_dim * num_cat + cont_latent_dim*2,
        )
        else:
            context_encoder = None
    else:
        if glob:
            context_encoder = encoder_model(
                hidden_sizes=[net_size, net_size, net_size],
                input_size=obs_dim + action_dim + reward_dim,
                output_size=latent_dim * num_cat + cont_latent_dim*2 + dir_latent_dim * num_dir * 2,
            )
        else:
            context_encoder = None
        context_encoder2 = None

    # create keypoint model
    kp_model = KeyNet(num_points=500, num_key=variant['num_key'], num_cates=1)
    gnn_model = GNN(variant['gnn_params'])


    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,
        output_size=1,
    )
    target_qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,
        output_size=1,
    )
    target_qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,
        output_size=1,
    )
    #!! add if here
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,
        latent_dim=latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        # latent_dim,
        # num_cat,
        # cont_latent_dim,
        # dir_latent_dim,
        # num_dir,
        kp_model,
        gnn_model,
        context_encoder,
        context_encoder2,
        recurrent_context_encoder,
        variant['global_latent'],
        variant['vrnn_latent'],
        policy,
        variant['temperature'],
        variant['unitkl'],
        variant['alpha'],
        # variant['prior'],
        variant['constraint'],
        variant['vrnn_constraint'],
        variant['var'],
        variant['vrnn_alpha'],
        variant['vrnn_var'],
        rnn,
        variant['temp_res'],
        variant['rnn_sample'],
        **variant['algo_params']
    )
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        with open(os.path.join(path, 'extra_data.pkl'), 'rb') as f:
            extra_data = pickle.load(f)
            variant['algo_params']['start_epoch'] = extra_data['epoch'] + 1
            replay_buffer = extra_data['replay_buffer']
            enc_replay_buffer = extra_data['enc_replay_buffer']
            variant['algo_params']['_n_train_steps_total'] = extra_data['_n_train_steps_total']
            variant['algo_params']['_n_env_steps_total'] = extra_data['_n_env_steps_total']
            variant['algo_params']['_n_rollouts_total'] = extra_data['_n_rollouts_total']
    else:
        replay_buffer=None
        enc_replay_buffer=None

    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, target_qf1, target_qf2],
        latent_dim=latent_dim,
        max_disc_cap=variant['max_disc_cap'],
        max_cont_cap=variant['max_cont_cap'],
        max_dir_cap=variant['max_dir_cap'],
        replay_buffer=replay_buffer,
        enc_replay_buffer=enc_replay_buffer,
        temp_res=variant['temp_res'],
        rnn_sample=variant['rnn_sample'],
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        if recurrent_context_encoder != None:
            recurrent_context_encoder.load_state_dict(torch.load(os.path.join(path, 'recurrent_context_encoder.pth')))
            # algorithm.recurrent_context_optimizer.load_state_dict(torch.load(os.path.join(path, 'recurrent_context_optimizer.pth')))
        if context_encoder != None:
            context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
            # algorithm.context_optimizer.load_state_dict(torch.load(os.path.join(path, 'context_optimizer.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        target_qf1.load_state_dict(torch.load(os.path.join(path, 'target_qf1.pth')))
        target_qf2.load_state_dict(torch.load(os.path.join(path, 'target_qf2.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))
        # algorithm.policy_optimizer.load_state_dict(torch.load(os.path.join(path, 'policy_optimizer.pth')))
        # algorithm.qf1_optimizer.load_state_dict(torch.load(os.path.join(path, 'qf1_optimizer.pth')))
        # algorithm.qf2_optimizer.load_state_dict(torch.load(os.path.join(path, 'qf2_optimizer.pth')))
        # algorithm.alpha_optimizer.load_state_dict(torch.load(os.path.join(path, 'alpha_optimizer.pth')))

    # optional GPU mode
    if ptu.gpu_enabled():
        algorithm.to()
        # gpu_optimizer(algorithm.qf1_optimizer)
        # gpu_optimizer(algorithm.qf2_optimizer)
        # if context_encoder != None:
        #     gpu_optimizer(algorithm.context_optimizer)
        # gpu_optimizer(algorithm.alpha_optimizer)
        # gpu_optimizer(algorithm.policy_optimizer)
        # if recurrent_context_encoder != None:
        #     gpu_optimizer(algorithm.recurrent_context_optimizer)

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))
    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    if variant.get('log_name', "") == "":
        log_name = variant['env_name']
    else:
        log_name = variant['log_name']
    experiment_log_dir = setup_logger(log_name, \
                            variant=variant, \
                            exp_id=exp_id, \
                            base_log_dir=variant['util_params']['base_log_dir'], \
                            config_log_dir=variant['util_params']['config_log_dir'], \
                            log_dir=variant['util_params']['log_dir'])
    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    if os.environ['DEBUG'] != '0' and variant['env_name'].endswith('point-robot'):
        import datetime
        import dateutil.tz
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(num=0, figsize=(8,8))
        axes = plt.axes()
        axes.set(aspect='equal')
        plt.axis([-1.25, 1.25, -1.25, 1.25])
        for g in env._wrapped_env.goals[:variant['n_train_tasks']]:
            circle = plt.Circle((g[0], g[1]), radius=env._wrapped_env.goal_radius)
            axes.add_artist(circle)

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        plt.savefig('./img-test/train-tasks-%s.pdf'%timestamp)

        plt.figure(num=1, figsize=(8,8))
        axes = plt.axes()
        axes.set(aspect='equal')
        plt.axis([-1.25, 1.25, -1.25, 1.25])
        for g in env._wrapped_env.goals[-variant['n_eval_tasks']:]:
            circle = plt.Circle((g[0], g[1]), radius=env._wrapped_env.goal_radius)
            axes.add_artist(circle)

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        plt.savefig('./img-test/test-tasks-%s.pdf'%timestamp)
        # exit(-1)

    # from pympler import asizeof
    # print (asizeof.asizeof(algorithm)/1e9)
    # exit(-1)
    print(experiment_log_dir)
    env.save_all_tasks(experiment_log_dir)

    # run the algorithm
    if variant['eval']:
        algorithm._try_to_eval(0, eval_all=True, eval_train_offline=False, animated=True)
    else:
        algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
@click.option('--seed', default=0)
@click.option('--kl_anneal', default="none", help="none or mono or cycle")
@click.option('--temperature', default=0.33)
@click.option('--logdir', default='output')
@click.option('--kl_lambda', default=1.)
# @click.option('--latent_size', default=5)
# @click.option('--num_cat', default=5)
@click.option('--unitkl', is_flag=True, default=False)
@click.option('--max_disc_cap', default=0.)
# @click.option('--cont_latent_size', default=5)
@click.option('--max_cont_cap', default=0.)
@click.option('--n_iteration', default=100)
@click.option('--alpha', default=0.7)
# @click.option('--dir_latent_size', default=5)
# @click.option('--num_dir', default=5)
@click.option('--constraint', default='deepsets')
@click.option('--env_alpha', default=0.7)
@click.option('--var', default=2.5)
@click.option('--max_dir_cap', default=0.)
@click.option('--task', default=-1)
@click.option('--eval', is_flag=True, default=False)
@click.option('--path_to_weights', default=None)
@click.option('--recurrent', is_flag=True, default=False)
@click.option('--vrnn_latent', default='2.0.0.2.4', help='gaus-dim.num-cat.cat-dim.num-dir.dir-dim')
@click.option('--global_latent', default='2.0.0.2.4', help='gaus-dim.num-cat.cat-dim.num-dir.dir-dim')
@click.option('--rnn', default='rnn', help='rnn or vrnn or None')
@click.option('--traj_batch_size', default=16)
@click.option('--vrnn_constraint', default='dirichlet', help="logitnormal or dirichlet")
@click.option('--vrnn_alpha', default=0.7)
@click.option('--vrnn_var', default=2.5)
@click.option('--temp_res', default=10)
@click.option('--rnn_sample', default="full", help="full or full_wo_sampling or single_sampling or batch_sampling")
@click.option('--resample_in_traj', is_flag=True, default=False)
# @click.option('--alpha_p', default=1.)
#! add window length
#! add flag online adaptation
def main(config, gpu, docker, debug, seed, kl_anneal, temperature, logdir, kl_lambda, \
            unitkl, max_disc_cap, max_cont_cap, n_iteration, alpha, \
            constraint, env_alpha, var, max_dir_cap, task, eval, path_to_weights, \
            recurrent, vrnn_latent, global_latent, rnn, traj_batch_size, vrnn_constraint, \
            vrnn_alpha, vrnn_var, temp_res, rnn_sample, resample_in_traj):
    cont_latent_size, num_cat, latent_size, num_dir, dir_latent_size = read_dim(global_latent)
    glob = latent_size * num_cat + cont_latent_size + dir_latent_size * num_dir > 0
    if resample_in_traj:
        assert glob
    assert kl_anneal in ['none', 'mono', 'cycle']
    if not recurrent:
        vrnn_latent = '0.0.0.0.0'
        rnn = 'None'
        traj_batch_size = -1
        vrnn_constraint = None
        vrnn_alpha = None
        vrnn_var = None
        if not resample_in_traj:
            temp_res = None
        rnn_sample = None
    r_cont_dim, r_n_cat, r_cat_dim, r_n_dir, r_dir_dim = read_dim(vrnn_latent)
    if recurrent:
        temp_res = int(temp_res)
        assert rnn_sample in ["full", "full_wo_sampling", "single_sampling", "batch_sampling"]
        if rnn_sample == 'full':
            temp_res = 1
        if r_dir_dim > 0:
            assert vrnn_constraint in ['logitnormal', 'dirichlet']
            if vrnn_constraint == 'logitnormal':
                vrnn_alpha = None
            else:
                vrnn_var = None
        else:
            vrnn_alpha = None
            vrnn_var = None
            vrnn_constraint = None
    if resample_in_traj:
        temp_res = int(temp_res)
    # assert latent_size * num_cat + cont_latent_size + dir_latent_size * num_dir > 0
    # assert prior in ['cat', 'dir']
    if latent_size == 0:
        num_cat = 0
    if dir_latent_size == 0:
        num_dir = 0
    if dir_latent_size > 0:
        assert constraint in ['deepsets', 'logitnormal']
        # assert not unitkl
        if constraint == 'deepsets':
            var = None
    else:
        constraint = None
        alpha = None
        var = None

    set_global_seeds(seed)
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    if gpu != -1:
        variant['util_params']['gpu_id'] = gpu
    else:
        variant['util_params']['use_gpu'] = False
    variant['seed'] = seed
    variant['temperature'] = temperature
    variant['env_params']['seed'] = seed
    # print (variant['env_params']['env_name'])
    if variant['env_name'] == 'sparse-dirichlet-point-robot':
        variant['env_params']['alpha'] = env_alpha
    else:
        env_alpha = None

    variant['algo_params']['kl_anneal'] = kl_anneal
    variant['util_params']['base_log_dir'] = logdir
    variant["algo_params"]['kl_lambda'] = kl_lambda
    # variant['latent_size'] = latent_size
    # variant['num_cat'] = num_cat
    variant['unitkl'] = unitkl
    variant['max_disc_cap'] = max_disc_cap
    # variant['cont_latent_size'] = cont_latent_size
    variant['max_cont_cap'] = max_cont_cap
    variant['alpha'] = alpha
    variant['var'] = var
    # variant['dir_latent_size'] = dir_latent_size
    # variant['num_dir'] = num_dir
    variant['constraint'] = constraint
    variant['max_dir_cap'] = max_dir_cap
    variant['eval'] = eval
    variant['path_to_weights'] = path_to_weights
    variant['algo_params']['recurrent'] = recurrent #! maybe add this to the naming process
    variant['algo_params']['glob'] = glob #! maybe add this to the naming process

    variant['vrnn_latent'] = vrnn_latent #! maybe add this to the naming process
    variant['global_latent'] = global_latent
    variant['rnn'] = rnn
    variant['algo_params']['traj_batch_size'] = traj_batch_size
    variant['vrnn_constraint'] = vrnn_constraint
    variant['vrnn_alpha'] = vrnn_alpha
    variant['vrnn_var'] = vrnn_var
    variant['util_params']['log_dir'] = None

    variant['temp_res'] = temp_res
    variant['rnn_sample'] = rnn_sample
    variant['algo_params']['resample_in_traj'] = resample_in_traj

    if 'meta' not in config:
        if not resample_in_traj:
            variant['util_params']['config_log_dir'] = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-var-%s-vrnn-%s-rnn-%s-vc-%s-va-%s-vvar-%s-res-%s-%s/seed-%s'%\
                (latent_size, num_cat, cont_latent_size, dir_latent_size, num_dir, kl_lambda, \
                    temperature, kl_anneal, unitkl, max_disc_cap, max_cont_cap, max_dir_cap, \
                    alpha, constraint, var, vrnn_latent, rnn, vrnn_constraint, \
                    vrnn_alpha, vrnn_var, temp_res, rnn_sample, seed)
        else:
            variant['util_params']['config_log_dir'] = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-var-%s-vrnn-%s-rnn-%s-vc-%s-va-%s-vvar-%s-res-%s-%s-resample/seed-%s'%\
                (latent_size, num_cat, cont_latent_size, dir_latent_size, num_dir, kl_lambda, \
                    temperature, kl_anneal, unitkl, max_disc_cap, max_cont_cap, max_dir_cap, \
                    alpha, constraint, var, vrnn_latent, rnn, vrnn_constraint, \
                    vrnn_alpha, vrnn_var, temp_res, rnn_sample, seed)
        variant['meta'] = None
    else:
        if 'meta.json' in config:
            assert task >= 0 and task < 15
            task_name = [
                            'reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1', 'drawer-close-v1', \
                            'button-press-topdown-v1', 'peg-insert-side-v1', 'window-open-v1', 'sweep-v1', 'basketball-v1', \
                            'drawer-open-v1', 'door-close-v1', 'shelf-place-v1', 'sweep-into-v1', 'lever-pull-v1'
                        ]
            task = task_name[task]
            variant['task'] = task
            variant['meta'] = 'meta'
        elif 'meta10.json' in config:
            task = 'meta10'
            variant['meta'] = 'meta10'
        elif 'oldmeta10.json' in config:
            task = 'oldmeta10'
            variant['meta'] = 'oldmeta10'
        variant['algo_params']['save_replay_buffer'] = True
        variant['algo_params']['save_enc_replay_buffer'] = True
        variant['util_params']['config_log_dir'] = '%s/dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-ea-%s-var-%s-vrnn-%s-rnn-%s-vc-%s-va-%s-vvar-%s-res-%s-%s/seed-%s'%\
            (task, latent_size, num_cat, cont_latent_size, dir_latent_size, num_dir, kl_lambda, \
                temperature, kl_anneal, unitkl, max_disc_cap, max_cont_cap, max_dir_cap, \
                alpha, constraint, env_alpha, var, vrnn_latent, rnn, vrnn_constraint, \
                vrnn_alpha, vrnn_var, temp_res, rnn_sample, seed)

    # if True:
    #     variant['algo_params']['save_replay_buffer'] = True
    #     variant['algo_params']['save_enc_replay_buffer'] = True
    #     variant['util_params']['log_dir'] = 'log-test/log-test'
    #     variant['util_params']['config_log_dir'] = 'log-test' #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!remove it
    if eval:
        variant['util_params']['config_log_dir'] = os.path.join('eval', variant['util_params']['config_log_dir'])
    variant['util_params']['debug'] = debug
    variant['algo_params']['num_iterations'] = int(n_iteration)
    experiment(variant)

if __name__ == "__main__":
    main()

