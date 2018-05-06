#!/usr/bin/env python3

import sys
import numpy as np
import gym
import time
from optparse import OptionParser
import ipdb
import random

import matplotlib.pyplot as plt

# -----------------------------
# ---MaxEnt Inverse RL agent---
# -----------------------------
class InverseAgentClass():
    
    def __init__(self, env, test_env, tau_num, tau_len, risk_mode):

        ## risk mode setting
        self.risk_mode = risk_mode
        self.env = env
        self.test_env = test_env        

        ## gradient converge check
        self.gradient_history =  [];

        # rl meta-parameters
        self.tau_num = tau_num; # number of trajectories
        self.tau_len = tau_len; # length of each trajectory
        self.gamma = 0.9; # discount factor
        self.alpha = 0.1; # learning rate
        
        self.gridSize = env.gridSize
        self.num_states = self.gridSize*self.gridSize # number of states
        self.num_actions = env.action_space.n # number of actions

        ## POLICY / value-FUNCTIONS
        #self.theta = np.round(np.random.rand(self.num_states,self.num_actions),2) # policy parameter theta # should we parametrize pi??
        self.pi = np.round(np.random.rand(self.num_states,self.num_actions),2)
        self.pi = (self.pi.T/np.sum(self.pi,1)).T
        self.value = np.zeros((self.num_states))

        ## REWARD
        self.psi = np.random.rand(self.num_states); # reward function parameter

        ## plot reward function
        self.init_reward_plot()

    # STEP:0 store all trajectories data (states and actions seperately)
    def store_trajectories(self, TAU):
        self.TAU_S = TAU[0];
        self.TAU_A = TAU[1];

    ## STEP:1 value-iteration
    ## perform value iteration with the current R(s;psi) and update pi(a|s;theta)
    def value_iteration(self,env):

        value_threshold = 0.001;
        
        while True:
            update_difference = -999;
            for s in range(self.num_states):
                old_value = self.value[s]
                self.value[s] = np.max( [np.sum([ env.T_sas(s,a,s_prime)*(self.reward(s_prime)+self.gamma*self.value[s_prime])
                                                  for s_prime in range(self.num_states)])
                                         for a in range(self.num_actions)])
                update_difference = max(update_difference, abs(old_value-self.value[s]))
            if(update_difference<value_threshold):
                break;

        #print("Value iteration converged!: update_difference=",update_difference)

        # get pi(a|s) = argmax_a sum_s' (r(s')+gamma*v(s'))
        for s in range(self.num_states):
            #greedy_action = np.argmax([np.sum([ env.T_sas(s,a,s_prime)*(self.reward(s_prime)+self.gamma*self.value[s_prime])
            #                                    for s_prime in range(self.num_states)])
            #                           for a in range(self.num_actions)])
            for a in range(self.num_actions):
                #self.pi[s,a] = 1.0 if a==greedy_action else 0.0;
                self.pi[s,a] = np.exp(np.sum([ env.T_sas(s,a,s_prime)*(self.reward(s_prime)+self.gamma*self.value[s_prime]) for s_prime in range(self.num_states)]))
                
            self.pi[s,:] /= np.sum(self.pi[s,:])

        if self.risk_mode==True and self.risk_taker.test_MDPs(self.env,self.test_env)==0:
            self.pi = self.risk_taker.alter_policy_for_risk(self.pi)

    ## get reward: r(s;psi)
    ## reward function is linear r(s;pi)= psi(i) phi(i)
    def reward(self,s):
        return self.psi[s];
                    
    # get policy: pi(a|s,theta)
    def policy(self,env,s,a):
        #return np.exp(self.theta[s,a])/ np.sum([np.exp(self.theta[s,b]) for b in range(self.num_actions)])
        return self.pi[s,a]

    ## STEP:2.1 compute P(s | TAU, T)
    ## find the state-visition frequency for the provided trajectories
    def get_state_visitation_frequency_under_TAU(self,env):

        # mu_tau[state, time] is the prob of visiting state s at time t FROM our trajectories         
        mu_tau = np.zeros([self.num_states])        
        
        for tau_i in self.TAU_S.T:
            for tau_it in tau_i:
                if(tau_it>=0):
                    mu_tau[int(tau_it)] += 1.0

        return mu_tau
    
    ## STEP:2.2 compute P(s | pi_theta, T)
    ## find the state-visitation frequency for all states
    def get_state_visitation_frequency(self,env):

        # mu[state, time] is the prob of visiting state s at time t
        mu = np.zeros([self.num_states, self.tau_len]) 

        # TODO: WHY MU_0 comes from trajectories?
        for tau_t0 in self.TAU_S[0,:]: # look at t=0 for each trajectory
            if int(tau_t0)>=0:
                mu[int(tau_t0),0] += 1.0 # initialize mu(.,t=0)

        mu[:,0] = mu[:,0]/float(self.tau_num)

        for time in range(self.tau_len-1):
            for state_next in range(self.num_states):
                mu[state_next, time+1] +=  np.sum([np.sum([mu[state, time] * self.policy(env,state,action) * env.T_sas(state,action,state_next)
                                                           for action in range(self.num_actions)])
                                                   for state in range(self.num_states)])
        return np.sum(mu, 1) # squeeze throughout time and return

    #### plotters
    def init_reward_plot(self):

        fig = plt.figure(figsize=(5,5))
        self.axes = fig.add_subplot(111)
        self.axes.set_autoscale_on(True)

        r = np.reshape(self.psi,(self.gridSize,self.gridSize));

        self.r_plotter = plt.imshow(r,interpolation='none', cmap='viridis', vmin=r.min(), vmax=r.max());
        plt.colorbar(); plt.xticks([]); plt.yticks([]); self.axes.grid(False);
        plt.title('inferred reward'); plt.ion(); plt.show();
        
    def see_reward_plot(self):
        r = np.reshape(self.psi,(self.gridSize,self.gridSize))        ;
        self.r_plotter.set_data(r)
        plt.clim(r.min(),r.max()) 
        plt.draw(); plt.show()
        plt.pause(0.0001)

    #####################################################
    ############### sub-goal discovery ##################
    #####################################################

    def get_subgoal(self):

        self.env.reset(False) # reset, but not random reset
        start_s = env.agentPos[0] + self.gridSize*env.agentPos[1]
        self.reward[start_s] = 0
        ipdb.set_trace()
        return np.argmax(self.reward)
        
    def update(self,env,PRINT):

        mu_tau = self.get_state_visitation_frequency_under_TAU(env)

        for step in range(1000):

            # [STEP:1] solve for optimal policy: do policy iteration on r(s;psi)
            self.value_iteration(env);
            
            # [STEP:2] compute state-visitation frequencies under tau / otherwise
            mu = self.get_state_visitation_frequency(env)

            # [STEP:3] find gradient
            grad = mu_tau/self.tau_num - mu;            
            
            # [STEP:4] update psi of r(s;psi)
            self.psi = self.psi + self.alpha*grad;

            # print, plot, and debug
            self.see_reward_plot()
            print("f_tau=",np.sum(mu_tau/self.tau_num)," mu=",np.sum(mu)," gradient=",np.sum(grad))
            self.gradient_history.append(np.sum(np.abs(grad)))

            # convergence check
            if step>4:
                if all(np.isclose(self.gradient_history[-2:],self.gradient_history[-4:-2],0.01)):
                    break

        print(self.gradient_history)
        return self.get_subgoal()
            
        print("updating..")    
