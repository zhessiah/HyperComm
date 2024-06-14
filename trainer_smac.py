from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
from torch.optim.lr_scheduler import StepLR

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))
# Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc', 'entropy_loss'))


class SMACTrainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.num_agents = args.nagents
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.997)

        self.params = [p for p in self.policy_net.parameters()]
        
        self.device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')


    def get_episode(self, epoch):
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        state = state.to(self.device)
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size, device=self.device)

        terminated = False
        t = 0
        while not terminated:
            misc = dict()

            if t == 0:
                prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

            x = [state, prev_hid]
            action_out, value, prev_hid = self.policy_net(x, info)

            if (t + 1) % self.args.detach_gap == 0:
                prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())



            # smac need to use available actions
            avail_actions = None
            if hasattr(self.env, 'get_avail_actions'):
                avail_actions = self.env.get_avail_actions()
                avail_actions = torch.tensor(avail_actions, device=self.device).unsqueeze(0)
            action = select_action(self.args, action_out, avail_actions)
            action, actual = translate_action(self.args, self.env, action)

            next_state, reward, terminated, info = self.env.step(actual)

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones(self.num_agents)
            stat['reward'] = stat.get('reward', 0) + reward
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


            episode_mask = np.ones(self.num_agents)
            episode_mini_mask = np.ones(self.num_agents)

            if terminated:
                episode_mask = np.zeros(self.num_agents)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if terminated:
                break
            
            t += 1
            
            
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)


    def compute_grad(self, batch):
        stat = dict()
        # num_actions: number of discrete actions in the action space
        num_actions = self.args.num_actions
        # dim_actions: number of action heads
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        # rewards: [batch_size * n]
        rewards = torch.tensor(batch.reward, device=self.device)
        # episode_mask: [batch_size * n]
        episode_masks = torch.tensor(batch.episode_mask, device=self.device)
        # episode_mini_mask: [batch_size * n]
        episode_mini_masks = torch.tensor(batch.episode_mini_mask, device=self.device)
        actions = torch.tensor(batch.action, device=self.device)
        # actions: [batch_size * n * dim_actions] have been detached
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        # alive_masks: [batch_size * n]
        alive_masks = torch.tensor(np.concatenate([item['alive_mask'] for item in batch.misc]), device=self.device).view(-1)

        coop_returns = torch.empty(batch_size, n, device=self.device)
        ncoop_returns = torch.empty(batch_size, n, device=self.device)
        returns = torch.empty(batch_size, n, device=self.device)
        deltas = torch.empty(batch_size, n, device=self.device)
        advantages = torch.empty(batch_size, n, device=self.device)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()
            
        # element of log_p_a: [(batch_size*n) * num_actions[i]]
        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        # actions: [(batch_size*n) * dim_actions]
        actions = actions.contiguous().view(-1, dim_actions)

        if self.args.advantages_per_action:
            # log_prob: [(batch_size*n) * dim_actions]
            log_prob = multinomials_log_densities(actions, log_p_a)
            # the log prob of each action head is multiplied by the advantage
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            # log_prob: [(batch_size*n) * 1]
            log_prob = multinomials_log_density(actions, log_p_a)
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        # entropy regularization term
        entropy = 0
        for i in range(len(log_p_a)):
            entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
        stat['entropy'] = entropy.item()
        if self.args.entr > 0:
            loss -= self.args.entr * entropy

        loss.backward()
        

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
#         for name, param in self.policy_net.named_parameters():
#             print(name)
#             print(param.grad)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()
        self.scheduler.step()
        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
