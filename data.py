import sys
import gym
import ic3net_envs

from env_wrappers import *

def init(env_name, args, final_init=True):
    if env_name == 'predator_prey':
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env, args)
    elif env_name == 'traffic_junction':
        env = gym.make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env, args)
    elif env_name == 'smac':
        env = SMACWrapper(args)
        
    else:
        raise RuntimeError("wrong env name")

    return env
