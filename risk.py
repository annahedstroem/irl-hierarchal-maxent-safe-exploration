#!/usr/bin/env python3

import sys
import numpy as np
import gym
import time
from optparse import OptionParser
import ipdb
import random
import itertools
import inverse_agent as inverse_agent
import hirl as inverse_agent

import matplotlib.pyplot as plt

# -----------------------------
# ---MaxEnt Inverse RL agent---
# -----------------------------


class RiskClass():
    def __init__(self, env, test_env, TAU_S):

        self.gridSize = env.gridSize
        self.num_states = self.gridSize * self.gridSize  # number of states
        self.num_actions = env.action_space.n  # number of actions

        self.TAU_S = TAU_S

    # Check if training and test environments are the same. If they differ; say state 16 is no longer
    # "empty" but an "unknown" object, we need to alter our policy for risk.
    def test_MDPs(self, env, test_env):
        env_states = self.sensor_uncertainty(env)
        test_states = self.sensor_uncertainty(test_env)

        if env_states[1] == test_states[1]:
            # print("train and test env are identical")
            return 1
        else:
            # print("train and test env differ")
            return 0

    # Sensor to get the object type or other properties for our states that was included in the expert trajectories.
    # This information allow us to interpret the expert trajectories as a "proxy" rather than taking them at face value.
    def sensor_uncertainty(self, env):

        known = ['goal', 'wall', 'flag']
        self.risk_types = []

        # Check object types of expert trajectories.
        for traj in self.TAU_S.T:
            for s in traj:
                states = (int(s % self.gridSize), int(s / self.gridSize))

                check = env.grid.get(*states)
                empty = env.grid.get(*states) == None
                object = env.grid.get(*states) != None

                # Classify risk based on uncertainty level.
                if empty:
                    self.risk_types.append((s, 0))                      # empty grid
                elif object and any(i in check.type for i in known):
                    self.risk_types.append((s, 1))                      # known object
                else:
                    self.risk_types.append((s, "?"))                    # uncertainty identified

        return self.risk_types

    # Determine how neighbouring states should behave according to some identified uncertain state.
    # E.g. if we want a risk-averse agent we can set the probability to zero of going to that risky state.
    def alter_policy_for_risk(self, pi):

        # Identify risky states.
        risky_s = np.unique([s for (s, risk) in self.risk_types if risk == "?"])

        # Identify concerned neighbours with respect to our risky states.
        for i in range(len(risky_s)):
            turn = 0
            idy, idx = (int(risky_s[i] % self.gridSize), int(risky_s[i] / self.gridSize))
            neighbour_idx = np.vstack((idx - 1, idx + 1, idx, idx)); neighbour_idy = np.vstack((idy, idy, idy - 1, idy + 1))
            num_neighbour = len(neighbour_idy)

            for j in range(num_neighbour):
                x = []; y = []
                x.append(neighbour_idx[j]); y.append(neighbour_idy[j])
                x = np.ndarray.flatten(np.array(x)); y = np.ndarray.flatten(np.array(y))
                nei_coor = (y + self.gridSize * x)

                # KEY: Policy implementation.
                # Determine how to alter our policy, pi(a|s;theta), e.g. as below with forbidden actions.
                if turn == 0 and x[0] >= 0 and y[0] >= 0:
                    self.pi[nei_coor, 3] = 0                # Down action is forbidden for the cell above our risky state.
                if turn == 1 and x[0] >= 0 and y[0] >= 0:
                    self.pi[nei_coor, 2] = 0                # Up action is forbidden for the cell below our risky state.
                if turn == 2 and x[0] >= 0 and y[0] >= 0:
                    self.pi[nei_coor, 1] = 0                # Right action is forbidden for the cell left to our risky state.
                if turn == 3 and x[0] >= 0 and y[0] >= 0:
                    self.pi[nei_coor, 0] = 0                # Left action is forbidden for the cell right to our risky state.
                turn += 1

        return self.pi
