#!/usr/bin/env python3

import sys
import numpy as np
import gym
import time
from optparse import OptionParser
import ipdb
import random
import risk as risk

import matplotlib.pyplot as plt

# -----------------------------
# ---MaxEnt Inverse RL agent---
# -----------------------------
class HInverseAgentClass():
    
    def __init__(self, env, test_env, tau_num, tau_len, risk_mode):

        # risk mode setting
        self.risk_mode = risk_mode
        self.env = env
        self.test_env = test_env

        ## gradient converge check
        self.gradient_history =  [];

        # rl meta-parameters
        self.tau_num = tau_num; # number of trajectories
        self.tau_len = tau_len; # length of each trajectory
        self.gamma = 0.9; # discount factor
        self.alpha = 1.0; # learning rate
        
        self.gridSize = env.gridSize
        self.num_states = self.gridSize*self.gridSize # number of states
        self.num_actions = env.action_space.n # number of actions

        ## POLICY / value-FUNCTIONS
        self.pi = np.round(np.random.rand(self.num_states,self.num_actions),2)
        self.pi = (self.pi.T/np.sum(self.pi,1)).T
        self.value = np.zeros((self.num_states))

        ## REWARD r(s;psi) = softmax(psi.s)
        self.psi = np.random.rand(self.num_states); # reward function parameter
        self.expo = np.random.random(self.num_states) # (optimizing) store exp(psi[s]) from psi
        self.reward = np.random.rand(self.num_states) # (optimizing) store reward function from psi

        self.compute_reward_from_psi()

        ## plot reward function
        self.init_reward_plot()

        ## ENTROPY REG
        self.entropy_under_tau = 0

    ######## 4-step MaxEnt #########
    
    # [STEP:0] store all trajectories data (states and actions seperately)
    def store_trajectories(self, TAU):
        self.TAU_S = TAU[0];
        self.TAU_A = TAU[1];

        if self.risk_mode:
            self.risk_taker = risk.RiskClass(self.env,self.test_env,self.TAU_S)


    ## [STEP:1] do value iteration with the current r(s;psi) and update pi
    def value_iteration(self,env):

        value_threshold = 0.001;
        
        while True:
            update_difference = -999;
            for s in range(self.num_states):
                old_value = self.value[s]
                self.value[s] = np.max( [np.sum([ env.T_sas(s,a,s_prime)*(self.reward[s_prime]+self.gamma*self.value[s_prime])
                                                  for s_prime in range(self.num_states)])
                                         for a in range(self.num_actions)])
                update_difference = max(update_difference, abs(old_value-self.value[s]))
            if(update_difference<value_threshold):
                break;

        #print("Value iteration converged!: update_difference=",update_difference)

        # get pi(a|s) = argmax_a sum_s' (r(s')+gamma*v(s'))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.pi[s,a] = np.exp(np.sum([ env.T_sas(s,a,s_prime)*(self.reward[s_prime]+self.gamma*self.value[s_prime]) for s_prime in range(self.num_states)]))
                                    
            self.pi[s,:] /= np.sum(self.pi[s,:])

        if self.risk_mode==True and self.risk_taker.test_MDPs(self.env,self.test_env)==0:
            self.pi = self.risk_taker.alter_policy_for_risk(self.pi)

    ## get SOFTMAX reward: r(s;psi) = softmax(psi.phi) = softmax(psi[s])
    def compute_reward_from_psi(self):
        for s in range(self.num_states):
            self.expo[s] = np.exp(self.psi[s])
        self.reward = self.expo/np.sum(self.expo)

    # get policy: pi(a|s,theta)
    def policy(self,env,s,a):
        return self.pi[s,a]

    ## STEP:2.1 compute P(s | TAU, T)
    ## find the state-visition frequency for the provided trajectories
    def get_feature_count_under_TAU(self,env):

        r_trail = np.zeros([self.num_states])        

        for tau_i in self.TAU_S.T:
            for tau_it in tau_i:
                if(tau_it>=0):
                    r_trail[int(tau_it)] += self.reward[int(tau_it)]

        r_trail = r_trail - self.reward*np.sum(r_trail)

        return r_trail  
    
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
    
        mu = np.sum(mu, 1); # state-visitation frequency squeeze throughout time        
        
        term2 = np.multiply(mu,self.reward)
        term2 = term2 - np.sum(term2)*self.reward            
        return term2

    #### plotters
    def init_reward_plot(self):

        fig = plt.figure(figsize=(5,5))
        self.axes = fig.add_subplot(111)
        self.axes.set_autoscale_on(True)

        r = np.reshape(self.reward,(self.gridSize,self.gridSize));

        self.r_plotter = plt.imshow(r,interpolation='none', cmap='viridis', vmin=r.min(), vmax=r.max());
        plt.colorbar(); plt.xticks([]); plt.yticks([]); self.axes.grid(False);
        plt.title('inferred reward'); plt.ion(); plt.show();
        
    def see_reward_plot(self):
        r = np.reshape(self.reward,(self.gridSize,self.gridSize))        ;
        self.r_plotter.set_data(r)
        plt.clim(r.min(),r.max()) 
        plt.draw(); plt.show()
        plt.pause(0.0001)

    #####################################################
    ############### sub-goal discovery ##################
    #####################################################

    def get_subgoal(self):
        ipdb.set_trace()
        return np.argmax(self.reward)

    #######################################
    ############### hirl ##################
    #######################################
    
    # compute H(r(s)) for s in TAU
    def compute_entropy_under_tau(self,env):
        #fig = plt.figure(figsize=(5,5))
        rprob = np.zeros([self.num_states])
        for tau_i in self.TAU_S.T:
            for tau_it in tau_i:
                if(tau_it>=0):
                    rprob[int(tau_it)] += self.reward[int(tau_it)]
        rprob = rprob/np.sum(self.TAU_S>=0)

        log_rprob = np.log(rprob);
        log_rprob[log_rprob==np.inf]=0; log_rprob[log_rprob==-np.inf]=0;

        self.entropy_under_tau = -np.multiply(rprob,log_rprob)
        
        #plt.imshow(np.reshape(self.entropy_under_tau, (env.gridSize,env.gridSize)),interpolation='none', cmap='viridis')
        #plt.colorbar(); plt.xticks([]); plt.yticks([]);        
        #plt.title('H(r) for s in TAU'); plt.ioff(); plt.show();    

    # compute H(r(s)) for all s
    def compute_entropy(self,env):

        #fig = plt.figure(figsize=(5,5))
        rprob = self.reward
        log_rprob = np.log(rprob);
        log_rprob[log_rprob==np.inf]=0;log_rprob[log_rprob==-np.inf]=0;

        self.entropy = -np.multiply(rprob,log_rprob)
        
        #plt.imshow(np.reshape(self.entropy, (env.gridSize,env.gridSize)),interpolation='none', cmap='viridis')
        #plt.colorbar(); plt.xticks([]); plt.yticks([]);        
        #plt.title('H(r) for all S'); plt.ioff(); plt.show();    

    def grad_entropy(self,env):

        r_under_tau = np.zeros([self.num_states])        

        for tau_i in self.TAU_S.T:
            for tau_it in tau_i:
                if(tau_it>=0):
                    r_under_tau[int(tau_it)] += self.reward[int(tau_it)]

        return self.entropy_under_tau - r_under_tau*np.sum(self.entropy)
        
        #return self.entropy_under_tau - self.reward*np.sum(self.entropy_under_tau)
        
        #plt.imshow(np.reshape(entropy_under_tau, (env.gridSize,env.gridSize)),interpolation='none', cmap='viridis')
        #plt.colorbar(); plt.xticks([]); plt.yticks([]);        
        #plt.title('state prob'); plt.ioff(); plt.show();    

    def update(self,env,PRINT):

        term1 = self.get_feature_count_under_TAU(env)

        for step in range(10):

            print("computing gradient.. ",end='')
            
            # [STEP:1] solve for optimal policy: do policy iteration on r(s;psi)
            self.value_iteration(env);
            
            # [STEP:2] compute state-visitation frequencies under tau / otherwise
            term2 = self.get_state_visitation_frequency(env)
            
            # [STEP:3] find gradient
            grad = term1/self.tau_num - term2
            
            # [STEP:3.5] alter to entropy gradient
            #self.compute_entropy_under_tau(env)
            #self.compute_entropy(env)
            #grad = grad - self.grad_entropy(env)/self.tau_num

            # [STEP:4] update psi of r(s;psi)
            self.psi = self.psi + self.alpha*grad;
            self.compute_reward_from_psi()

            ## print and convergence check grads
            print("step:",step,"t1=",np.sum(np.abs(term1/self.tau_num))," t2=",np.sum(np.abs(term2))," gradient=",np.sum(np.abs(grad)))
            
            self.see_reward_plot()
            self.gradient_history.append(np.sum(np.abs(grad)))
            
            if step>4:
                if all(np.isclose(self.gradient_history[-2:],self.gradient_history[-4:-2],0.00001)):
                    break

        print(self.gradient_history)
        return self.get_subgoal()
