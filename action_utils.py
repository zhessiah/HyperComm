import numpy as np
import torch
from torch.autograd import Variable

def parse_action_args(args):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        args.continuous = False
        # assert args.dim_actions == 1
        # support multi action
        args.naction_heads = [int(args.num_actions[i]) for i in range(args.dim_actions)]
    else:
        # environment takes continuous action
        actions_heads = args.nactions.split(':')
        if len(actions_heads) == 1 and int(actions_heads[0]) == 1:
            args.continuous = True
        elif len(actions_heads) == 1 and int(actions_heads[0]) > 1:
            args.continuous = False
            args.naction_heads = [int(actions_heads[0]) for _ in range(args.dim_actions)]
        elif len(actions_heads) > 1:
            args.continuous = False
            args.naction_heads = [int(i) for i in actions_heads]
        else:
            raise RuntimeError("--nactions wrong format!")


def select_action(args, action_out, available_actions=None):
    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        if available_actions is None:
            log_p_a = action_out
            p_a = [[z.exp() for z in x] for x in log_p_a]
            ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
            return ret
        else:
            if len(action_out) < 2:
                log_p_a = action_out[0].clone()
                log_p_a[available_actions == 0] = float('-inf')  # set the log probability of unavailable actions to -inf
                p_a = torch.softmax(log_p_a, dim=-1)  # compute softmax only on available actions
                ret = torch.stack([torch.multinomial(x, 1).detach() for x in p_a])
                return ret
            else:
                log_p_a = action_out[0].clone()
                log_p_comm = action_out[1].clone()
                log_p_a[available_actions == 0] = float('-inf')  # set the log probability of unavailable actions to -inf
                p_a = [torch.softmax(log_p_a[0], dim=-1)]  # compute softmax only on available actions (the second list is communication actions)
                p_comm = [x.exp() for x in log_p_comm]
                p_concat = [p_a, p_comm]
                ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_concat])
                # ret_a = torch.stack([torch.multinomial(x, 1).detach() for x in p_a])
                
                # ret = torch.stack([torch.distributions.Categorical(x).sample().detach() for x in p_a[0]], [torch.multinomial(x, 1).detach() for x in p_a[1]])
                return ret

def translate_action(args, env, action):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        action = action.to('cpu')
        action = [x.squeeze().data.numpy() for x in action]
        actual = action
        return action, actual
    else:
        if args.continuous:
            action = action.data[0].numpy()
            cp_action = action.copy()
            # clip and scale action to correct range
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                cp_action[i] = cp_action[i] * args.action_scale
                cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            return action, cp_action
        else:
            actual = np.zeros(len(action))
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
            action = [x.squeeze().data[0] for x in action]
            return action, actual
