import argparse
import torch
import time
import os
import numpy as np
import gymnasium as gym

from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = False  # torch.cuda.is_available()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            np.random.seed(seed + rank * 1000)
            env.reset(seed=seed + rank * 1000)
            return env
        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
def pad_or_clip_action(ac, act_space):
    expected_size = act_space.n if hasattr(act_space, 'n') else int(np.prod(act_space.shape))
    ac = np.array(ac, dtype=np.float32)
    if ac.size < expected_size:
        padded = np.zeros(expected_size, dtype=np.float32)
        padded[:ac.size] = ac
        ac = padded
    elif ac.size > expected_size:
        ac = ac[:expected_size]
    if isinstance(act_space, Box):
        ac = np.clip(ac, act_space.low, act_space.high)
    return ac


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)

    maddpg = MADDPG.init_from_env_with_delay(env,
                                             agent_alg=config.agent_alg,
                                             adversary_alg=config.adversary_alg,
                                             tau=config.tau,
                                             lr=config.lr,
                                             hidden_dim=config.hidden_dim,
                                             delay_step=1)
    
    delay_step = 1
    print(f"env: {dir(env)}")
    base_env = env.envs[0]
    #print(f"ENVS:{dir(env.envs)}")
    agents = base_env.agents
    nagents = len(agents)
    print(f"Agents: {agents}, nagents: {nagents}")
    print(f"base_env: {base_env}")
    print("[DEBUG] Base env agents:", base_env.agents)
    obs_sizes = []
    for agent in base_env.agents:
        obs_size = base_env.observation_space(agent).shape[0]
        
        # Add size of delayed actions
        act_space = base_env.action_space(agent)
        act_size = 1 if isinstance(act_space, Discrete) else int(np.prod(act_space.shape))
        
        obs_size += delay_step * act_size
        obs_sizes.append(obs_size)
    ac_dims = [
        base_env.action_space(agent).n if isinstance(base_env.action_space(agent), Discrete)
        else int(np.prod(base_env.action_space(agent).shape))
        for agent in base_env.agents
    ]

    
    replay_buffer = ReplayBuffer(
        config.buffer_length,
        maddpg.nagents,
        obs_sizes,
        ac_dims
    )


    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs_dict = env.reset()
        if isinstance(obs_dict, dict):
            agents = list(obs_dict.keys())
            obs = np.array([obs_dict[a] for a in agents], dtype=object).reshape(1, -1)
        else:
            # env is DummyVecEnv, so it's already an array
            obs = obs_dict
            agents = list(range(obs.shape[1])) if obs.ndim > 1 else [0]

        print(f"[DEBUG] Normalized OBS array: {obs.shape}")
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        base_env = env.envs[0]  # first env in the vector
        agents = base_env.agents
        nagents = len(agents)

        zero_agent_actions = [
            np.zeros(
                base_env.action_space(agent).n 
                if isinstance(base_env.action_space(agent), Discrete)
                else int(np.prod(base_env.action_space(agent).shape))
            )
            for agent in agents
            ]
        last_agent_actions = [zero_agent_actions for _ in range(delay_step)]

        

        # obs: list of shape (n_rollout_threads, nagents), each obs[t][a_i] is np.ndarray
        # inside your run() function, replace this part:
        for t in range(config.n_rollout_threads):
            for a_i in range(maddpg.nagents):
                # If obs is empty dict (agent not active yet), replace with zeros
                if isinstance(obs[t][a_i], dict):
                    obs[t][a_i] = np.zeros(
                        base_env.observation_space(base_env.agents[a_i]).shape[0],
                        dtype=np.float32
                    )
                else:
                    obs[t][a_i] = obs[t][a_i].astype(np.float32)
        
                print(f"Agent {a_i} obs shape BEFORE delay append: {obs[t][a_i].shape}")
        
                # Append only this agent's delayed actions as raw values
                for d in range(delay_step):
                    act = last_agent_actions[d][a_i]
                    obs[t][a_i] = np.append(obs[t][a_i], act.astype(np.float32))
        
                print(f"Agent {a_i} obs shape after delay append: {obs[t][a_i].shape}")





        print(f"[DEBUG] Flattened OBS ready for torch: {obs}")


        print(f"[DEBUG] Flattened OBS before appending last_agent_actions: {obs}")

       


        for et_i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions = [
                pad_or_clip_action(ac.data.numpy(), base_env.action_space(agent))
                for ac, agent in zip(torch_agent_actions, base_env.agents)
            ]
            for idx, (agent, ac) in enumerate(zip(base_env.agents, agent_actions)):
                    print(f"[DEBUG] Agent {agent} action shape: {ac.shape}, values: {ac}")



            if delay_step == 0:
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            else:
                #agent_actions_tmp = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)][0]
                agent_actions_tmp = [
                    pad_or_clip_action(ac.data.numpy().flatten(), base_env.action_space(agent))
                    for ac, agent in zip(torch_agent_actions, base_env.agents)
                ]

                for idx, (agent, ac) in enumerate(zip(base_env.agents, agent_actions_tmp)):
                    print(f"[DEBUG] Agent TMP {agent} action shape: {ac.shape}, values: {ac}")

                actions = last_agent_actions[0]
                last_agent_actions = last_agent_actions[1:]
                last_agent_actions.append(agent_actions_tmp)
            
           
             # agent_names = base_env.agents
            print(type(env))
            print(type(base_env))
            actions_dict = {
                agent_name: agent_action
                for agent_name, agent_action in zip(base_env.agents, agent_actions_tmp)
            }
            for agent_name, agent_action in actions_dict.items():
                print(f"[DEBUG] {agent_name} flattened action shape: {agent_action.shape}, values: {agent_action}")

            print(f"Env Step Input: {[actions_dict]}")
            next_obs, rewards, dones, infos = env.step([actions_dict])
            print(f"Env Step Output: {next_obs, rewards, dones, infos}")

            for a_i, agent_obs in enumerate(next_obs[0]):
                for d in range(delay_step):
                    agent_obs = np.append(agent_obs, last_agent_actions[d][a_i])
                next_obs[0][a_i] = agent_obs


            

            
            # Ensure each agent's action matches replay buffer shape
            # Build buffer push list that is (num_agents, 1, agent_action_dim)
            # Assume actions_dict is a dict: {agent_name: action for each agent}
            ac_dims = [
                base_env.action_space(agent).n if isinstance(base_env.action_space(agent), Discrete)
                else int(np.prod(base_env.action_space(agent).shape))
                for agent in base_env.agents
            ]
            
            actions_buffered = []
            for agent, adim in zip(base_env.agents, ac_dims):
                ac = np.array(actions_dict[agent], dtype=np.float32)
                # Always reshape to (1, action_dim)
                if ac.ndim == 1:
                    ac = ac.reshape(1, -1)
                # Pad or clip to correct shape if necessary
                if ac.shape[1] < adim:
                    pad = np.zeros((1, adim), dtype=ac.dtype)
                    pad[0, :ac.shape[1]] = ac
                    ac = pad
                elif ac.shape[1] > adim:
                    ac = ac[:, :adim]
                actions_buffered.append(ac)
            print("=== Debug: Actor buffer push shapes and values ===")
            for idx, (agent, ac) in enumerate(zip(base_env.agents, actions_buffered)):
                print(f"Agent {agent} (index {idx}):")
                print(f"  Expected action dim: {ac_dims[idx]}")
                print(f"  Action shape: {ac.shape}")
                print(f"  Action sample values: {ac.flatten()[:10]}")  # print first 10 elements
            print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
            print(f"Reward shape: {np.array(rewards).shape if hasattr(rewards, '__len__') else 'N/A'}")
            #print(f"Next obs shape: {np.array(next_obs).shape if hasattr(next_obs, '__len__') else 'N/A'}")
            print(f"Dones shape: {np.array(dones).shape if hasattr(dones, '__len__') else 'N/A'}")
            # Inspect structure of next_obs
            print("Type of next_obs:", type(next_obs))
            if isinstance(next_obs, (list, tuple)):
                print(f"Number of elements in next_obs: {len(next_obs)}")
                for i, env_obs in enumerate(next_obs):
                    print(f"  Element {i}: type={type(env_obs)}")
                    if hasattr(env_obs, '__len__'):
                        print(f"  Element {i} length: {len(env_obs)}")
                    if isinstance(env_obs, list):
                        for j, agent_obs in enumerate(env_obs):
                            print(f"    Agent {j}: type={type(agent_obs)}, shape={getattr(agent_obs, 'shape', None)}")
                            if isinstance(agent_obs, np.ndarray):
                                print(f"      shape={agent_obs.shape}")
            else:
                print("next_obs is not list or tuple, type:", type(next_obs))

            print("===============================================")
            # In 'run' just before replay_buffer.push
            rewards = np.array(rewards)
            obs = np.array(obs)
            # next_obs is List[envs] of List[agents] of np.ndarray(obs_dim)
            agent_next_obs = [[] for _ in base_env.agents]  # one list per agent
            for env_obs in next_obs:  # iterate over environments (usually 1 here)
                for i, agent_obs in enumerate(env_obs):
                    agent_next_obs[i].append(agent_obs)
            
            # Stack per agent to make np.arrays of shape (num_envs, obs_dim)
            next_obs_stacked = [np.stack(obs_list) for obs_list in agent_next_obs]

            dones = np.array(dones)

            replay_buffer.push(obs, actions_buffered, rewards, next_obs_stacked, dones)

            #replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            obs = next_obs
            t += config.n_rollout_threads

            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents - 1):  # do not update the runner
                        sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_adversaries()
                maddpg.prep_rollouts(device='cpu')

        ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalars('agent%i/mean_episode_rewards' % a_i, {'reward': a_ep_rew}, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name", help="Name of directory to store model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_true')

    config = parser.parse_args()
    run(config)
