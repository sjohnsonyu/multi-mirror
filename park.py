"""
implement a park with defined behavior for wildlife and poacher response

Lily Xu, 2021
"""

import numpy as np
import torch
import math

PAST_ILLEGAL = False  # use past illegal activity, not past effort
USE_NEIGHBOR = True   # use neighboring cells


def convert_to_a(raw_a, param_int, use_torch=False):
    """ convert raw value (e.g., from nature strategy) to attractiveness_poaching """
    if use_torch:
        a = torch.tanh(raw_a)
        a = (a + 1.) / 2.
        a = a * torch.Tensor(param_int[:, 1] - param_int[:, 0])
        a = a + torch.Tensor(param_int[:, 0])
    else:
        a = np.tanh(raw_a)
        if torch.is_tensor(a):
            a = a.detach().numpy()
        if torch.is_tensor(param_int):
            param_int = param_int.detach().numpy()

        a = (a + 1.) / 2.
        a = a * (param_int[:, 1] - param_int[:, 0])
        a = a + (param_int[:, 0])

    return a

class Park:
    def __init__(self,
                 attractiveness_poaching,
                 attractiveness_logging,
                 initial_effort,
                 initial_wildlife,
                 initial_trees,
                 initial_attack,
                 height,
                 width,
                 n_targets,
                 budget,
                 horizon,
                 psi,
                 alpha,
                 beta,
                 eta,
                 reward_mode,
                 verbose=False,
                 param_int_poaching=None,
                 param_int_logging=None):
        """
        attractiveness_poaching: will be torch.tensor (if tracking gradients for Nature oracle)
                        or np.ndarray (if agent oracle)
        param_int_poaching: optional parameter; if set for Nature oracle, then attractiveness_poaching
                   values will be computed through a sigmoid
        """

        # if true, use torch (instead of numpy) to track gradients
        # used by nature oracle
        self.use_tensor = torch.is_tensor(attractiveness_poaching) or torch.is_tensor(attractiveness_logging)
        # TODO this won't always be attractiveness_poaching that's a tensor! sometimes it's logging
        # is it harmful to treat everything as a tensor?
        
        self.reward_mode = reward_mode

        # store initial values for resetting
        if self.use_tensor:
            self.initial_wildlife = initial_wildlife
            self.initial_trees = initial_trees
            self.initial_effort = initial_effort
            self.initial_attack = initial_attack
            self.effort = torch.FloatTensor(initial_effort)
            self.wildlife = torch.FloatTensor(initial_wildlife)
            self.trees = torch.FloatTensor(initial_trees)
        else:
            self.initial_wildlife = np.array(initial_wildlife)
            self.initial_trees = np.array(initial_trees)
            self.initial_effort = np.array(initial_effort)
            self.initial_attack = np.array(initial_attack)
            self.effort = np.array(initial_effort)
            self.wildlife = np.array(initial_wildlife)
            self.trees = np.array(initial_trees)

        self.height = height
        self.width = width
        self.n_targets = n_targets
        self.state_dim = 2 * n_targets + 1
        self.action_dim = n_targets

        assert height * width == n_targets

        self.budget = budget

        # note that attractiveness_poaching here is not the final value; it is
        # the input to the sigmoid
        self.param_int_poaching = param_int_poaching
        self.attractiveness_poaching = attractiveness_poaching
        self.param_int_logging = param_int_logging
        self.attractiveness_logging = attractiveness_logging

        self.t = 0 # timestep
        self.horizon = horizon

        self.psi = psi    # wildlife growth ratio
        self.psi_tree = 1    # wildlife growth ratio
        self.alpha = alpha  # strength that poachers eliminate wildlife
        self.beta = beta   # coefficient on current effort - likelihood of finding snares
        self.eta = eta    # effect of neighbors

        def i_to_xy(i):
            x = math.floor(i / self.width)
            y = i % self.width
            return x, y

        def xy_to_i(x, y):
            return (x * self.width) + y

        # create neighbors dict for easy access later
        self.neighbors = {}
        for i in range(n_targets):
            neigh = []
            x, y = i_to_xy(i)

            for xx in range(x-1, x+1):
                for yy in range(y-1, y+1):
                    if xx < 0 or xx >= self.width: continue
                    if yy < 0 or yy >= self.height: continue
                    if xx == x and yy == y: continue
                    ii = xy_to_i(xx, yy)
                    neigh.append(ii)

            self.neighbors[i] = neigh

        self.verbose = verbose

    def get_curr_attack(self, p_attack, use_torch):
        if use_torch:
            return torch.bernoulli(p_attack)
        return np.random.binomial(1, p=p_attack)

    def get_torch_modes(self, use_torch, state_mode):
        poaching_use_torch = use_torch and (self.reward_mode == 'poaching' or state_mode == 'poaching')
        logging_use_torch = use_torch and (self.reward_mode == 'logging' or state_mode == 'logging')
        return poaching_use_torch, logging_use_torch

    def step(self, action, state_mode, use_torch=False):
        '''
        returns:
        - observation (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
        - reward (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
        - done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
        - info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). However, official evaluations of your agent are not allowed to use this for learning.
        '''
        assert state_mode in ('poaching', 'logging'), 'state_mode must be one of poaching and logging'
    
        if use_torch:
            assert torch.is_tensor(action)

        poaching_use_torch, logging_use_torch = self.get_torch_modes(use_torch, state_mode)
        # ensure action is legal
        assert action.sum() <= self.budget + 1e-5

        p_attack_poaching = self.adv_behavior(self.attractiveness_poaching, self.param_int_poaching, action, poaching_use_torch)
        p_attack_logging = self.adv_behavior(self.attractiveness_logging, self.param_int_logging, action, logging_use_torch)

        curr_attack_poaching =  self.get_curr_attack(p_attack_poaching, poaching_use_torch)
        curr_attack_logging =  self.get_curr_attack(p_attack_logging, logging_use_torch)

        self.effort = action
        curr_wildlife = self.resource_response(self.wildlife, curr_attack_poaching, action, self.psi, poaching_use_torch)
        curr_trees = self.resource_response(self.trees, curr_attack_logging, action, self.psi_tree, logging_use_torch)

        # instead of using stochastic draw of attack, compute expected attack
        expected_wildlife = self.resource_response(self.wildlife, p_attack_poaching, action, self.psi, poaching_use_torch)
        expected_trees = self.resource_response(self.trees, p_attack_logging, action, self.psi_tree, logging_use_torch)
        
        expected_reward = expected_wildlife.sum() if self.reward_mode == 'poaching' else expected_trees.sum()

        if self.verbose:
            print('pattack_poaching', np.around(p_attack_poaching, 5), '  attack', curr_attack_poaching, f'  wildlife', np.around(curr_wildlife, 2))
            print('pattack_logging', np.around(p_attack_logging, 5), '  attack', curr_attack_logging, f'  trees', np.around(curr_trees, 2))

        self.wildlife = curr_wildlife
        self.trees = curr_trees

        self.t += 1

        reward = self.wildlife.sum() if self.reward_mode == 'poaching' else self.trees.sum()

        info = {'expected_reward': expected_reward}  # dict for debugging
        return self.get_state(state_mode, use_torch), reward, self.is_terminal(), info

    def get_state(self, state_mode, use_torch):
        """ state is a tensor of dimension n_target * 3 + 1
        [curr_wildlife, curr_trees, curr_effort]
        """
        assert state_mode in ('poaching', 'logging'), 'state_mode must be one of poaching and logging'

        resource = self.wildlife if state_mode == 'poaching' else self.trees
        if use_torch:
            if state_mode == 'poaching':
                assert torch.is_tensor(self.wildlife), 'wildlife not tensor {}'.format(self.wildlife)
            else:
                assert torch.is_tensor(self.trees), 'trees not tensor {}'.format(self.trees)
            assert torch.is_tensor(self.effort), 'effort not tensor {}'.format(self.effort)
            return torch.cat((resource, self.effort, torch.FloatTensor([self.t])))

        return np.concatenate((resource, self.effort, np.array([self.t])))

    def is_terminal(self):
        assert self.t <= self.horizon, 'error: timestep {} is beyond horizon {}'.format(self.t, self.horizon)
        return self.t == self.horizon

    def reset(self, state_mode):
        assert state_mode in ('poaching', 'logging'), 'state_mode must be one of poaching and logging'
        poaching_use_torch, logging_use_torch = self.get_torch_modes(self.use_tensor, state_mode)

        self.t = 0
        if self.use_tensor:
            self.effort = torch.FloatTensor(self.initial_effort)
            if poaching_use_torch:  # determines what is an array and what is a tensor
                self.wildlife = torch.FloatTensor(self.initial_wildlife)
            if logging_use_torch:
                self.trees = torch.FloatTensor(self.initial_trees)
            if not poaching_use_torch:
                self.wildlife = np.array(self.initial_wildlife)
                # possibly convert attractiveness to an array??
            if not logging_use_torch:
                self.trees = np.array(self.initial_trees)
                # possibly convert attractiveness to an array??
                # why did we need it as a tensor anyways?
        else:
            self.effort = np.array(self.initial_effort)
            self.wildlife = np.array(self.initial_wildlife)
            self.trees = np.array(self.initial_trees)

        return self.get_state(state_mode, self.use_tensor)

    def adv_behavior(self, attractiveness, param_int, past_effort, use_torch=False):
        """ adversary response function
        a:       attractiveness_poaching
        beta:    responsiveness
        past_effort:  past effort
        eta:     neighbor effort response
        """
        assert self.beta < 0, self.beta
        assert self.eta >= 0, self.eta


        if param_int is not None:
            a = convert_to_a(attractiveness, param_int, use_torch=use_torch)
        else:
            a = attractiveness
        


        if not use_torch and isinstance(past_effort, torch.Tensor):
            past_effort = past_effort.detach().numpy()

        # whether to include displacement effect
        past_neigh = self.get_neighbor_effort(past_effort, use_torch)
        eta = self.eta if USE_NEIGHBOR else 0

        if use_torch:
            temp = torch.FloatTensor(self.beta * past_effort + eta * past_neigh)
            behavior = 1 / (1 + torch.exp(-(a + temp)))
        else:
            if torch.is_tensor(a):
                a = a.detach().numpy()
            behavior = 1 / (1 + np.exp(-(a + self.beta * past_effort + eta * past_neigh)))

        return behavior

    def resource_response(self, past_w, past_a, past_c, psi, use_torch=False):
        ''' wildlife response function
        psi:     wildlife growth ratio
        past_w:  past wildlife count
        past_a:  past poacher action
        past_c:  past ranger action
        alpha:   responsiveness to past poaching
        '''
        # assert self.psi >= 1, f'psi is {self.psi}'
        assert psi >= 1, f'psi is {psi}'
        self.validate_past_a(past_a)
        self.validate_past_c(past_c)

        # if rangers used full patrol, they stop all attacks
        effort_multiplier = 1. - past_c

        if use_torch:
            curr_w = torch.FloatTensor(past_w**psi) - (self.alpha * past_a * effort_multiplier)
            curr_w = torch.clamp(curr_w, 0, None)
        else:
            if torch.is_tensor(past_a):
                past_a = past_a.detach().numpy()
            if torch.is_tensor(past_w):
                past_w = past_w.detach().numpy()
            if torch.is_tensor(effort_multiplier):
                effort_multiplier = effort_multiplier.detach().numpy()
            curr_w = past_w**psi - (self.alpha * past_a * effort_multiplier)
            np.clip(curr_w, 0, None, out=curr_w)

        return curr_w

    def validate_past_a(self, past_a):
        if torch.is_tensor(past_a):
            assert torch.all(past_a <= 1.), 'past_a has val > 1 {}'.format(past_a)
            assert torch.all(past_a >= 0.), 'past_a has val < 0 {}'.format(past_a)
        else:
            assert np.all(past_a <= 1.), 'past_a has val > 1 {}'.format(past_a)
            assert np.all(past_a >= 0.), 'past_a has val < 0 {}'.format(past_a)
    
    def validate_past_c(self, past_c):
        # I think past_c is past patrol effort
        if torch.is_tensor(past_c):
            assert torch.all(past_c <= 1.), 'past_c has val > 1 {}'.format(past_c)
        else:
            assert np.all(past_c <= 1.), 'past_c has val > 1 {}'.format(past_c)

    def get_neighbor_effort(self, past_c, use_torch=False):
        if use_torch:
            neigh_effort = torch.zeros(self.n_targets)
            for i in range(self.n_targets):
                neigh_effort[i] = torch.sum(past_c[self.neighbors[i]])
        else:
            neigh_effort = np.zeros(self.n_targets)
            for i in range(self.n_targets):
                neigh_effort[i] = np.sum(past_c[self.neighbors[i]])

        return neigh_effort
    
    def get_attractiveness(self, state_mode):
        assert state_mode in ('poaching', 'logging'), 'state_mode must be one of poaching and logging'
        if state_mode == 'poaching':
            return self.attractiveness_poaching
        return self.attractiveness_logging
        
    def get_param_int(self, state_mode):
        assert state_mode in ('poaching', 'logging'), 'state_mode must be one of poaching and logging'
        if state_mode == 'poaching':
            return self.param_int_poaching
        return self.param_int_logging


    # def get_reward_wildlife(self):
    #     """ compute reward, defined as sum of wildlife """
    #     return self.wildlife.sum()

    # def get_reward_trees(self):
    #     """ compute reward, defined as sum of trees """
    #     return self.trees.sum()

    # def get_initial_state(self):
    #     assert self.use_tensor == False
    #     return np.concatenate((self.initial_wildlife, self.initial_trees, self.effort, np.array([0])))
