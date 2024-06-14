import time
import numpy as np
import torch
from gym.spaces import Discrete
from inspect import getargspec
from smac.env import MultiAgentEnv, StarCraft2Env


class GymWrapper(object):
    '''
    for multi-agent
    '''
    def __init__(self, env, args):
        self.env = env
        self.device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')


    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''

        # tuple space
        if hasattr(self.env.observation_space, 'spaces'):
            total_obs_dim = 0
            for space in self.env.observation_space.spaces:
                if hasattr(self.env.action_space, 'shape'):
                    total_obs_dim += int(np.prod(space.shape))
                else: # Discrete
                    total_obs_dim += 1
            return total_obs_dim
        else:
            return int(np.prod(self.env.observation_space.shape))

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            # Discrete
            return self.env.action_space.n

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete, pp
            return self.env.action_space.shape[0]
            # return len(self.env.action_space.shape)
        elif hasattr(self.env.action_space, 'n'):
            # Discrete => only 1 action takes place at a time.
            # tj, grf
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, epoch):
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            obs = self.env.reset(epoch)
        else:
            obs = self.env.reset()

        obs = self._flatten_obs(obs)
        return obs
    

    def display(self):
        self.env.render()
        time.sleep(0.5)

    def end_display(self):
        self.env.exit_render()

    def step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this
        if self.dim_actions == 1:
            action = action[0]
        obs, r, done, info = self.env.step(action)
        obs = self._flatten_obs(obs)
        return (obs, r, done, info)

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)

        obs = obs.reshape(1, -1, self.observation_dim)
        obs = torch.from_numpy(obs).double()
        return obs.to(self.device)

    def get_stat(self):
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()



class SMACWrapper(object):
    '''
    for multi-agent
    '''
    def __init__(self, args):
        self.env = StarCraft2Env(map_name=args.map_name)
        self.env_info = self.env.get_env_info()
        self.device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')


    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''

        # tuple space
        return self.env_info.get('obs_shape')

    @property
    def num_actions(self):
        return self.env_info.get('n_actions')

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        return 1 # same as Discrete case

    @property
    def action_space(self):
        return Discrete(self.env.get_total_actions())

    def reset(self):
        obs, _ = self.env.reset()
        obs = self._flatten_obs(obs)
        return obs
    

    def step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this
        if self.dim_actions == 1:
            action = action[0]
        obs, r, done, info = self.env.step(action)
        obs = self._flatten_obs(obs)
        return (obs, r, done, info)
    
    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """

        actions = list(action_dict[0])

        reward, done, info = self.env.step(actions)
        obs_list = self.env.get_obs()
        obs = self._flatten_obs(obs_list)
        return obs, reward, done, info

    # def reward_terminal(self):
    #     # if hasattr(self.env, 'reward_terminal'):
    #     #     return self.env.reward_terminal()
    #     # else:
    #     #     return np.zeros(1)
        
    #     return np.zeros(self.env_info["n_agents"])

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)
        obs = torch.tensor(obs, dtype=torch.float64, device=self.device)
        obs = obs.reshape(1, -1, self.observation_dim)
        return obs

    def get_stat(self):
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()
        
    def get_avail_actions(self):
        return self.env.get_avail_actions()
