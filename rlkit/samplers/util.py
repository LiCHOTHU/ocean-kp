import numpy as np
import rlkit.torch.pytorch_util as ptu

def rollout(env, agent, max_path_length=np.inf, accum_context=True, resample_z=False, animated=False, recurrent=False, temp_res=1, rnn_sample=None, resample_in_traj=False): 
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param accum_context: if True, accumulate the collected context
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    sucs = []
    o = env.reset()
    next_o = None
    path_length = 0
    # if animated:
        # env.render()
    if recurrent:
        agent.clear_sequence_z()
        contexts = []
        if rnn_sample == 'batch_sampling':
            batch_contexts = []
    frames = []
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        if resample_in_traj and path_length % temp_res == (temp_res - 1):
            agent.sample_z()
        if recurrent:
            # import pdb
            # pdb.set_trace()
            if rnn_sample == 'full':
                agent.infer_step_posterior(ptu.FloatTensor(np.expand_dims(np.concatenate([o, a, [r]]), axis=0)), resample=True)
            elif rnn_sample == 'full_wo_sampling':
                if path_length % temp_res == (temp_res - 1):
                    agent.infer_step_posterior(ptu.FloatTensor(np.expand_dims(np.concatenate([o, a, [r]]), axis=0)), resample=True)
                else:
                    agent.infer_step_posterior(ptu.FloatTensor(np.expand_dims(np.concatenate([o, a, [r]]), axis=0)), resample=False)
            elif rnn_sample == 'single_sampling':
                if path_length % temp_res == (temp_res - 1):
                    agent.infer_step_posterior(ptu.FloatTensor(np.expand_dims(np.concatenate([o, a, [r]]), axis=0)), resample=True)
            elif rnn_sample == 'batch_sampling':
                batch_contexts.append(np.concatenate([o, a, [r]]))
                if path_length % temp_res == (temp_res - 1):
                    batch_contexts = np.array(batch_contexts).reshape(1, -1)
                    agent.infer_step_posterior(ptu.FloatTensor(batch_contexts), resample=True)
                    batch_contexts = []
            contexts.append(agent.seq_z.detach().cpu().numpy())
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        if 'success' in env_info:
            sucs.append(env_info['success'])
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            img = env.sim.render(height=512, width=512, camera_name='topview')[::-1]
            frames.append(img)
            # env.render()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    if recurrent:
        contexts = np.array(contexts)
    else:
        contexts = None

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        frames=frames,
        contexts = contexts,
        sucs = sucs
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """

    '''
    currently does not update wrt recurrent context
    '''
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
