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
    """ convert raw value (e.g., from nature strategy) to attractiveness """
    if use_torch:
        a = torch.tanh(raw_a)
        a = (a + 1.) / 2.
        a = a * torch.Tensor(param_int[:, 1] - param_int[:, 0])
        a = a + torch.Tensor(param_int[:, 0])
    else:
        a = np.tanh(raw_a)
        a = (a + 1.) / 2.
        a = a * (param_int[:, 1] - param_int[:, 0])
        a = a + (param_int[:, 0])

    return a

class Park:
    def __init__(self,
                 attractiveness,  # TODO add in attractiveness of logging
                 initial_effort,
                 initial_wildlife,
                 initial_trees,  # TODO use in this class!
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
                 threat_mode,
                 verbose=False,
                 param_int=None):
        """
        attractiveness: will be torch.tensor (if tracking gradients for Nature oracle)
                        or np.ndarray (if agent oracle)
        param_int: optional parameter; if set for Nature oracle, then attractiveness
                   values will be computed through a sigmoid
        """

        # if true, use torch (instead of numpy) to track gradients
        # used by nature oracle
        self.use_tensor = torch.is_tensor(attractiveness)
        
        assert threat_mode in ('poaching', 'logging'), "threat_mode must be either 'poaching' or 'logging'"
        self.threat_mode = threat_mode

        # store initial values for resetting
        if self.use_tensor:
            self.initial_wildlife = initial_wildlife
            self.initial_trees = initial_trees
            self.initial_effort = initial_effort
            self.initial_attack = initial_attack
            self.effort = torch.FloatTensor(initial_effort)
            self.wildlife = torch.FloatTensor(initial_wildlife)
            self.trees = torch.FloatTensor(initial_trees)
            self.past_attack = torch.FloatTensor(initial_attack)
        else:
            self.initial_wildlife = np.array(initial_wildlife)
            self.initial_trees = np.array(initial_trees)
            self.initial_effort = np.array(initial_effort)
            self.initial_attack = np.array(initial_attack)
            self.effort = np.array(initial_effort)
            self.wildlife = np.array(initial_wildlife)
            self.trees = np.array(initial_trees)
            self.past_attack = np.array(initial_attack)

        if self.threat_mode == 'poaching':
            self.resource = self.wildlife
            self.initial_resource = self.initial_wildlife
            # self.get_reward_resource = self.get_reward_wildlife
        else:
            self.resource = self.trees
            self.initial_resource = self.initial_trees
            # self.get_reward_resource = self.get_reward_trees
            # print("Ignoring assumption that trees don't regrow")
            # self.psi = 1  # assume trees don't regrow

        self.height = height
        self.width = width
        self.n_targets = n_targets
        self.state_dim = 2 * n_targets + 1
        self.action_dim = n_targets

        assert height * width == n_targets

        self.budget = budget

        # note that attractiveness here is not the final value; it is
        # the input to the sigmoid
        self.param_int = param_int
        self.attractiveness = attractiveness

        self.t = 0 # timestep
        self.horizon = horizon

        self.psi = psi    # wildlife growth ratio
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


    def step(self, action, use_torch=False):
        '''
        returns:
        - observation (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
        - reward (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
        - done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
        - info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). However, official evaluations of your agent are not allowed to use this for learning.
        '''

        if use_torch:
            assert torch.is_tensor(action)

        # ensure action is legal
        assert action.sum() <= self.budget + 1e-5

        # p_attack = self.adv_behavior(self.wildlife, action, use_torch)
        p_attack = self.adv_behavior(self.resource, action, use_torch)

        if use_torch:
            curr_attack = torch.bernoulli(p_attack)
        else:
            curr_attack = np.random.binomial(1, p=p_attack)

        if PAST_ILLEGAL:
            self.effort = np.zeros(self.n_targets, dtype=int)
            self.effort[np.where(np.logical_and(action, curr_attack))] = 1
        else:
            self.effort = action

        curr_resource = self.resource_response(self.resource, curr_attack, action, use_torch)

        # instead of using stochastic draw of attack, compute expected attack
        expected_resource = self.resource_response(self.resource, p_attack, action, use_torch)
        expected_reward = expected_resource.sum()

        if self.verbose:
            population_str = 'wildlife' if self.threat_mode == 'poaching' else 'trees'
            print('pattack', np.around(p_attack, 5), '  attack', curr_attack, f'  {population_str}', np.around(curr_resource, 2))

        self.resource = curr_resource
        self.past_attack = curr_attack

        self.t += 1

        info = {'expected_reward': expected_reward}  # dict for debugging
        return self.get_state(use_torch), self.get_reward_resource(), self.is_terminal(), info


    def get_state(self, use_torch):
        """ state is a tensor of dimension n_target * 2
        [curr_resource, curr_effort]
        """
        if use_torch:
            assert torch.is_tensor(self.resource), 'resource not tensor {}'.format(self.resource)
            assert torch.is_tensor(self.effort), 'effort not tensor {}'.format(self.effort)
            return torch.cat((self.resource, self.effort, torch.FloatTensor([self.t])))
        else:
            return np.concatenate((self.resource, self.effort, np.array([self.t])))
        
        # NOTE broken for use_torch = True
        return state

    def get_effort(self):
        return self.effort

    # def get_wildlife(self):
    #     return self.wildlife

    # def get_trees(self):
    #     return self.trees

    def get_initial_state(self):
        assert self.use_tensor == False
        return np.concatenate((self.initial_resource, self.effort, np.array([0])))

    def get_attack(self):
        return self.past_attack

    def is_terminal(self):
        assert self.t <= self.horizon, 'uh oh! our timestep {} is beyond horizon {}'.format(self.t, self.horizon)

        return self.t == self.horizon

    # def get_reward_wildlife(self):
    #     """ compute reward, defined as sum of wildlife """
    #     return self.wildlife.sum()

    # def get_reward_trees(self):
    #     """ compute reward, defined as sum of trees """
    #     return self.trees.sum()
    def get_reward_resource(self):
        return self.resource.sum()

    def reset(self):
        self.t = 0
        if self.use_tensor:
            self.effort = torch.FloatTensor(self.initial_effort)
            self.wildlife = torch.FloatTensor(self.initial_wildlife)
            self.trees = torch.FloatTensor(self.initial_trees)
            self.past_attack = torch.FloatTensor(self.initial_attack)
        else:
            self.effort = np.array(self.initial_effort)
            self.wildlife = np.array(self.initial_wildlife)
            self.trees = np.array(self.initial_trees)
            self.past_attack = np.array(self.initial_attack)
        
        if self.threat_mode == 'poaching':
            self.resource = self.wildlife
        else:
            self.resource = self.trees

        return self.get_state(self.use_tensor)


    def adv_behavior(self, past_w, past_c, use_torch=False):
        ''' adversary response function
        a:       attractiveness
        beta:    responsiveness
        past_c:  past effort (this is the same thing as action, apparently)
        eta:     neighbor effort response
        '''
        assert self.beta < 0, self.beta
        assert self.eta >= 0, self.eta

        if self.param_int is not None:
            a = convert_to_a(self.attractiveness, self.param_int, use_torch=use_torch)
        else:
            a = self.attractiveness

        # whether to include displacement effect
        past_neigh = self.get_neighbor_effort(past_c, use_torch)
        eta = self.eta if USE_NEIGHBOR else 0

        if use_torch:
            temp = torch.FloatTensor(self.beta * past_c + eta * past_neigh)
            behavior = 1 / (1 + torch.exp(-(a + temp)))
        else:
            if torch.is_tensor(a):
                a = a.detach().numpy()
            behavior = 1 / (1 + np.exp(-(a + self.beta * past_c + eta * past_neigh)))

        return behavior


    def resource_response(self, past_w, past_a, past_c, use_torch=False):
        ''' wildlife response function
        psi:     wildlife growth ratio
        past_w:  past wildlife count
        alpha:   responsiveness to past poaching
        past_a:  past poacher action
        '''
        assert self.psi >= 1, f'psi is {self.psi}'
        self.validate_past_a(past_a)
        self.validate_past_c(past_c)

        # if rangers used full patrol, they stop all attacks
        effort_multiplier = 1. - past_c

        if use_torch:
            curr_w = torch.FloatTensor(past_w**self.psi) - (self.alpha * past_a * effort_multiplier)
            curr_w = torch.clamp(curr_w, 0, None)
        else:
            if torch.is_tensor(past_a):
                past_a = past_a.detach().numpy()
            if torch.is_tensor(past_w):
                past_w = past_w.detach().numpy()
            curr_w = past_w**self.psi - (self.alpha * past_a * effort_multiplier)
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
