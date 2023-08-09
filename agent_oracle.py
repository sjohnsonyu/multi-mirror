"""
implement agent oracle of double oracle

Lily Xu, 2021
"""

# import sys, os
# import pickle
import itertools
# import random
import numpy as np
# import torch
# from collections.abc import Iterable

from park import Park
from ddpg import *

from nature_oracle import sample_strategy

N_DISPLAY = 100


class AgentOracle:
    def __init__(self, park_params, checkpoints, n_train, n_eval):
        self.park_params = park_params
        self.checkpoints = checkpoints
        self.n_train = n_train
        self.n_eval = n_eval
        self.budget = park_params['budget']

    def simulate_reward(self,
                        agent_strategies,
                        nature_strategies,
                        agent_policy_modes,
                        reward_mode,
                        agent_distrib=None,
                        nature_distrib=None,
                        display=True,
                        attractiveness_secondary=None
                        ):
        """ this is similar to evaluate_DDPG() from agentender_oracle_evaluation.py
        if agent_distrib=None, agent_strategies only a single strategy
        if nature_distrib=None, nature_strategies only a single strategy """
        assert agent_distrib is None, "simulate_reward hasn't been implemented for evaluating multiple strategies"
        # TODO need to make sure that nature_strategy matches up with the reward mode

        if agent_distrib is None:
            assert len(agent_strategies) == 1
            agent_strategy = agent_strategies[0]
            agent_mode = agent_policy_modes[0]
        else:
            assert len(agent_strategies) == len(agent_distrib)

        if nature_distrib is None:
            assert len(nature_strategies) == 1
            attractiveness_primary = nature_strategies[0]
        else:
            assert len(nature_strategies) == len(nature_distrib)

        rewards = np.zeros(self.n_eval)
        for i_episode in range(self.n_eval):
            if nature_distrib is not None:
                nature_strategy_i = sample_strategy(nature_distrib)
                attractiveness_primary = nature_strategies[nature_strategy_i]
            if agent_distrib is not None:
                agent_strategy_i = sample_strategy(agent_distrib)
                agent_strategy = agent_strategies[agent_strategy_i]

            if attractiveness_secondary is None:
                attractiveness_secondary = (np.random.rand(*attractiveness_primary.shape) - 0.5) * 2

            attractiveness_poaching = attractiveness_primary if reward_mode == 'poaching' else attractiveness_secondary
            attractiveness_logging = attractiveness_primary if reward_mode == 'logging' else attractiveness_secondary

            park_params = self.park_params
            env = Park(attractiveness_poaching,
                       attractiveness_logging,
                       park_params['initial_effort'],
                       park_params['initial_wildlife'],
                       park_params['initial_trees'],
                       park_params['initial_attack'],
                       park_params['height'],
                       park_params['width'],
                       park_params['n_targets'],
                       park_params['budget'],
                       park_params['horizon'],
                       park_params['psi'],
                       park_params['alpha'],
                       park_params['beta'],
                       park_params['eta'],
                       reward_mode,
                       param_int_poaching=self.park_params['param_int'],
                       param_int_logging=self.park_params['param_int_logging']
                       )

            # initialize the environment and state
            state = env.reset(agent_mode)
            for t in itertools.count():
                # select and perform an action
                action = agent_strategy.select_action(state)

                # if DDPG (which returns softmax): take action up to budget and then clip each location to be between 0 and 1
                if isinstance(agent_strategy, DDPG):
                    before_sum = action.sum()
                    action = (action / action.sum()) * self.budget
                    action[np.where(action > 1)] = 1 # so DDPG learns to not make actions greater than budget

                next_state, reward, done, _ = env.step(action, agent_mode, use_torch=False)

                if display and i_episode % 1000 == 0:
                    print('  ', i_episode, t, action)

                # move to the next state
                state = next_state

                # evaluate performance if terminal
                if done:
                    rewards[i_episode] = reward
                    break

        avg_reward = np.mean(rewards)
        return avg_reward


    def best_response(self, nature_strategies, nature_distrib, reward_mode, display=False):
        # NOTE: nature_strategies refers to the poaching strategies when poaching is the objective,
        # and logging strategies when logging is the objective
        assert len(nature_strategies) == len(nature_distrib), 'nature strategies {}, distrib {}'.format(len(nature_strategies), len(nature_distrib))
        br, checkpoint_rewards = run_DDPG(self.park_params,
                                          nature_strategies,
                                          nature_distrib,
                                          self.checkpoints,
                                          self.n_train,
                                          reward_mode,
                                          display=display)

        return br

    def simulate_policy(self, agent_strategy, attractiveness, reward_mode, display=True):
        param_int = self.park_params['param_int'] if reward_mode == 'poaching' else self.park_params['param_int_logging']
        park_params = self.park_params
        env = Park(attractiveness,
                    park_params['initial_effort'],
                    park_params['initial_wildlife'],
                    park_params['initial_trees'],
                    park_params['initial_attack'],
                    park_params['height'],
                    park_params['width'],
                    park_params['n_targets'],
                    park_params['budget'],
                    park_params['horizon'],
                    park_params['psi'],
                    park_params['alpha'],
                    park_params['beta'],
                    park_params['eta'],
                    reward_mode,
                    param_int=param_int)

        policy = []
        rewards = []
        states = []
        # initialize the environment and state
        state = env.reset()

        for t in itertools.count():
            # select and perform an action
            action = agent_strategy.select_action(state)

            # if DDPG (which returns softmax): take action up to budget and then clip each location to be between 0 and 1
            if isinstance(agent_strategy, DDPG):
                before_sum = action.sum()
                action = (action / action.sum()) * self.budget
                action[np.where(action > 1)] = 1
                
            next_state, reward, done, _ = env.step(action, use_torch=False)
            
            policy.append(action)
            states.append(next_state)
            rewards.append(reward)

            # move to the next state
            state = next_state

            # evaluate performance if terminal
            if done:
                break
        
        return np.array(policy), np.array(rewards), np.array(states)
        

def run_DDPG(park_params, nature_strategies, nature_distrib, checkpoints, n_train, reward_mode, display=True):
    state_dim  = 2*park_params['n_targets'] + 1
    action_dim = park_params['n_targets']

    ddpg = DDPG(park_params['n_targets'])

    batch_size = 128
    rewards = []
    avg_rewards = []
    checkpoint_rewards = []


    # if args.load: agent.load()
    total_step = 0
    for i_episode in range(n_train):
        episode_reward = 0

        nature_strategy_i = sample_strategy(nature_distrib)
        attractiveness = nature_strategies[nature_strategy_i]
        arbitrary_attractiveness = (np.random.rand(*attractiveness.shape) - 0.5) * 2

        attractiveness_poaching = attractiveness if reward_mode == 'poaching' else arbitrary_attractiveness
        attractiveness_logging = attractiveness if reward_mode == 'logging' else arbitrary_attractiveness

        env = Park(attractiveness_poaching,
                   attractiveness_logging,
                   park_params['initial_effort'],
                   park_params['initial_wildlife'],
                   park_params['initial_trees'],
                   park_params['initial_attack'],
                   park_params['height'],
                   park_params['width'],
                   park_params['n_targets'],
                   park_params['budget'],
                   park_params['horizon'],
                   park_params['psi'],
                   park_params['alpha'],
                   park_params['beta'],
                   park_params['eta'],
                   reward_mode
                   )

        # initialize the environment and state
        state = env.reset(state_mode=reward_mode)

        for t in itertools.count():
            action = ddpg.select_action(state)

            next_state, reward, done, info = env.step(action, state_mode=reward_mode)
            reward = info['expected_reward']  # use expected reward

            ddpg.memory.push(state, action, np.expand_dims(reward, axis=0), next_state, done)

            if len(ddpg.memory) > batch_size:
                ddpg.update(batch_size)

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))


        if display and i_episode % N_DISPLAY == 0:
            print('episode {:4d}   reward: {:.2f}   average reward: {:.2f}'.format(i_episode, np.round(episode_reward, 2), np.mean(rewards[-10:])))

    return ddpg, checkpoint_rewards


