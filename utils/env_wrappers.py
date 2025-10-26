"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait_old(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    def step_wait(self):
        results = []
        for a, env in zip(self.actions, self.envs):
            obs_dict, rew_dict, done_dict, info_dict = env.step(a)  # unpack here
            # convert dicts to arrays in consistent order
            obs = np.array([obs_dict[agent] for agent in env.agents], dtype=np.float32)
            rewards = np.array([rew_dict[agent] for agent in env.agents], dtype=np.float32)
            dones = np.array([done_dict[agent] for agent in env.agents], dtype=np.bool_)
            infos = info_dict  # keep as dict
            results.append((obs, rewards, dones, infos))
        obs, rews, dones, infos = map(np.array, zip(*results))
        return obs, rews, dones, infos


    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = []
        for a, env in zip(self.actions, self.envs):
        # Step the underlying env
        step_result = env.step(a)
        
        # Handle both 5-value (newer PettingZoo) or 4-value (older) outputs
        if isinstance(step_result, (list, tuple)):
            if len(step_result) == 5:
                # unpack 5 values
                obs_dict, rewards, terminations, truncations, infos = step_result
                # merge terminations and truncations into single 'done'
                done_dict = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
            elif len(step_result) == 4:
                obs_dict, rewards, done_dict, infos = step_result
            else:
                raise ValueError(f"Unexpected number of values returned from env.step(): {len(step_result)}")
        else:
            raise ValueError("Env.step() returned something unexpected")
        
        # convert dicts to arrays per agent
        obs = np.array([obs_dict[agent] for agent in env.agents], dtype=np.float32)
        rewards = np.array([rewards[agent] for agent in env.agents], dtype=np.float32)
        dones = np.array([done_dict[agent] for agent in env.agents], dtype=np.bool_)
        infos = info_dict = info_dict  # keep as is
        
        results.append((obs, rewards, dones, infos))
        
        # stack results from all envs
        obs_batch, rew_batch, done_batch, infos_batch = map(np.array, zip(*results))
        self.actions = None
        return obs_batch, rew_batch, done_batch, infos_batch


    




    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return
