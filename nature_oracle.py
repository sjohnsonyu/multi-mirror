"""
implement nature oracle of double oracle

Lily Xu, 2020
"""

# import sys, os
# import pickle
import itertools
import random
import numpy as np
import torch

from park import Park, convert_to_a
from ddpg_nature import NatureDDPG

torch.autograd.set_detect_anomaly(True)


def sample_strategy(distrib):
    """ return the index of a strategy
    general util used by nature and agent oracle """
    strategy_i = random.choices(list(range(len(distrib))), weights=distrib)[0]
    return strategy_i

# NOTE: this is currently written so you need 2 diff nature oracle objects
# for poaching and logging but that's not really necessary; refactor
# to share a single object

class NatureOracle:
    def __init__(self,
                 park_params,
                 checkpoints,
                 n_train,
                 use_wake,
                 freeze_policy_step,
                 freeze_a_step
                 ):
        """
        use_wake: whether to use wake/sleep option
        freeze_policy_step: how often to freeze policy
        freeze_a_step: how often to unfreeze attractiveness
        """
        self.park_params = park_params
        self.n_train = n_train
        self.use_wake = use_wake
        self.freeze_policy_step = freeze_policy_step
        self.freeze_a_step = freeze_a_step

    def best_response(self,
                      agent_strategies,
                      agent_distrib,
                      agent_policy_modes,
                      reward_mode,
                      nature_strategies_secondary=None,
                      nature_strategies_secondary_distrib=None,
                      display=False):
        """
        agent_strategies: agent strategy set
        agent_distrib: mixed strategy distribution over the set

        returns: best response attractiveness
        """
        br = self.run_DDPG(agent_strategies,
                           agent_distrib,
                           agent_policy_modes,
                           reward_mode,
                           nature_strategies_secondary,
                           nature_strategies_secondary_distrib,
                           display=display)
        br = br.detach().numpy()

        return br

    def get_secondary_attractiveness(self, nature_strategies_secondary, nature_strategies_secondary_distrib):
        if nature_strategies_secondary is None:
            attractiveness_secondary = (np.random.rand(self.park_params['n_targets']) - 0.5) * 2
        else:
            assert len(nature_strategies_secondary) == len(nature_strategies_secondary_distrib)
            nature_strategy_secondary_i = sample_strategy(nature_strategies_secondary_distrib)
            attractiveness_secondary = nature_strategies_secondary[nature_strategy_secondary_i]
        return torch.Tensor(attractiveness_secondary)

    def run_DDPG(self,
                 agent_strategies,
                 agent_distrib,
                 agent_policy_modes,
                 reward_mode,
                 nature_strategies_secondary,
                 nature_strategies_secondary_distrib,
                 display=True):
        """ LEARNING ORACLE

        freeze_policy_step: for wake/sleep procedure, how often to freeze policy
        freeze_a_step: for wake/sleep procedure, how often to *unfreeze* attractiveness """

        # initialize with random attractiveness values in interval
        attractiveness_primary = (np.random.rand(self.park_params['n_targets']) - .5) * 2
        attractiveness_primary = attractiveness_primary.astype(float)
        print('attractiveness_primary (raw) before training', np.round(attractiveness_primary, 3))

        attractiveness_primary = torch.tensor(attractiveness_primary, requires_grad=True, dtype=torch.float32)

        ddpg = NatureDDPG(self.park_params['n_targets'], attractiveness_primary, actor_learning_rate=10, critic_learning_rate=10)

        batch_size = 4
        # batch_size  = 1
        rewards = []
        avg_rewards = []

        def get_agent_avg_reward(env, agent_strategy, agent_mode, n_iter=100):
            agent_total_rewards = torch.zeros(self.park_params['horizon'])
            for _ in range(n_iter):
                state = env.reset(agent_mode)
                for t in itertools.count():
                    action = agent_strategy.select_action(state)
                    action = torch.Tensor(action)
                    next_state, reward, done, info = env.step(action, agent_mode, use_torch=True)
                    agent_total_rewards[t] += reward
                    state = next_state

                    if done:
                        break

            agent_avg_rewards = agent_total_rewards / n_iter
            if display:
                print('agent avg rewards', agent_avg_rewards.detach().numpy())
            return agent_avg_rewards

        total_step = 0

        attractiveness_secondary = self.get_secondary_attractiveness(nature_strategies_secondary, nature_strategies_secondary_distrib)

        attractiveness_poaching = attractiveness_primary if reward_mode == 'poaching' else attractiveness_secondary
        attractiveness_logging = attractiveness_primary if reward_mode == 'logging' else attractiveness_secondary

        # TODO make sure gradients are still tracking
        env = Park(attractiveness_poaching,
                   attractiveness_logging,
                   self.park_params['initial_effort'],
                   self.park_params['initial_wildlife'],
                   self.park_params['initial_trees'],
                   self.park_params['initial_attack'],
                   self.park_params['height'],
                   self.park_params['width'],
                   self.park_params['n_targets'],
                   self.park_params['budget'],
                   self.park_params['horizon'],
                   self.park_params['psi'],
                   self.park_params['alpha'],
                   self.park_params['beta'],
                   self.park_params['eta'],
                   reward_mode,
                   param_int_poaching=self.park_params['param_int'],
                   param_int_logging=self.park_params['param_int_logging'])

        # memoize agent average reward for each policy
        agent_avg_rewards = []
        for i, agent_strategy in enumerate(agent_strategies):
            agent_avg_reward = get_agent_avg_reward(env, agent_strategy, agent_policy_modes[i])
            agent_avg_rewards.append(agent_avg_reward)

        print('agent strategies', len(agent_strategies))
        print('avg rewards', len(agent_avg_rewards), np.array([np.round(r.detach().numpy(), 2) for r in agent_avg_rewards]))

        # until convergence
        for i_episode in range(self.n_train):
            i_display = True if display and i_episode % 50 == 0 else False

            if self.use_wake:
                updating_a = i_episode % self.freeze_a_step == 0 # are we updating a?
                updating_policy = i_episode % self.freeze_policy_step > 0 # are we updating policy?

                if updating_a:
                    ddpg.unfreeze_attractiveness()
                else:
                    ddpg.freeze_attractiveness()

                if updating_policy and i_episode > 0:
                    ddpg.freeze_policy()
                else:
                    ddpg.unfreeze_policy()

            else:
                updating_a = True
                updating_policy = True

            # TODO I'm not totally sure whether these are copies or aliases to attractiveness_primary...
            # should we be detaching or not? I'm not totally sure
            # might need to use pytorch visualization
            if reward_mode == 'poaching':
                attractiveness_logging = self.get_secondary_attractiveness(nature_strategies_secondary, nature_strategies_secondary_distrib)
            else:
                attractiveness_poaching = self.get_secondary_attractiveness(nature_strategies_secondary, nature_strategies_secondary_distrib)

            env = Park(attractiveness_poaching,
                       attractiveness_logging,
                       self.park_params['initial_effort'],
                       self.park_params['initial_wildlife'],
                       self.park_params['initial_trees'],
                       self.park_params['initial_attack'],
                       self.park_params['height'],
                       self.park_params['width'],
                       self.park_params['n_targets'],
                       self.park_params['budget'],
                       self.park_params['horizon'],
                       self.park_params['psi'],
                       self.park_params['alpha'],
                       self.park_params['beta'],
                       self.park_params['eta'],
                       reward_mode,
                       param_int_poaching=self.park_params['param_int'],
                       param_int_logging=self.park_params['param_int_logging'])

            # get reward of sampled agent strategy
            agent_strategy_i  = sample_strategy(agent_distrib)
            agent_avg_reward = agent_avg_rewards[agent_strategy_i]
            state_mode = agent_policy_modes[agent_strategy_i]

            state = env.reset(reward_mode)  # NOTE we're using reward_mode now!
            episode_reward = 0

            a = convert_to_a(env.get_attractiveness(state_mode).detach().numpy(), env.get_param_int(state_mode))
            if i_display:
                # print('episode {} attractiveness {} raw {}'.format(i_episode, np.round(a, 3), np.round(raw_a, 3)))
                print('episode {} attractiveness {} raw {}'.format(i_episode, np.round(a, 3), np.round(a, 3)))
            state = torch.cat([state, attractiveness_primary])  # NOTE: not a? (maybe not because of gradient tracking)

            # for timesteps in one episode
            for t in itertools.count():
                # NOTE: I think since it's the same ddpg obj, it should always be reward mode?
                # but at the same time there's something that feels wrong about that...
                action = ddpg.select_action(state)

                next_state, reward, done, info = env.step(action, reward_mode, use_torch=True)
                # print(state, action, next_state, reward)
                next_state = torch.cat([next_state, attractiveness_primary])

                if i_display:
                    print('t {} action {} reward {:.3f}'. format(t, np.round(action.detach().numpy(), 3), reward.item()))

                reward = reward - agent_avg_reward[t]  # want to max agent regret
                reward = reward.unsqueeze(0)
                ddpg.memory.push(state, action, reward, next_state, done)

                if len(ddpg.memory) > batch_size:
                    ddpg.update(batch_size, display=i_display)

                state = next_state
                episode_reward += reward

                if done:
                    if updating_a:
                        state = env.reset(reward_mode)  # doesn't need to return anything, so state_mode doesn't matter
                        
                        # if we update attractiveness, update agent avg rewards
                        for i, agent_strategy in enumerate(agent_strategies):
                            agent_mode = agent_policy_modes[i]
                            agent_avg_rewards[i] = get_agent_avg_reward(env, agent_strategy, agent_mode)
                    break

            rewards.append(episode_reward)
            avg_rewards.append(torch.mean(torch.stack(rewards[-10:])))

        print('attractiveness_primary (raw) after training', np.round(attractiveness_primary.detach().numpy(), 3))
        return attractiveness_primary
