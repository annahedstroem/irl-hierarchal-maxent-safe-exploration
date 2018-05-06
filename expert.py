#!/usr/bin/env python3

import sys
import numpy as np
import gym
import time
from optparse import OptionParser
import ipdb
import random
import matplotlib.pyplot as plt

class ExpertClass():
    def __init__(self,env,tau_num,tau_len):

        self.HEIRARCHICAL = True;
        self.PLOT_V = True
        self.EGREEDY = False
        
        # states
        self.gridSize = env.gridSize
        if(self.HEIRARCHICAL):
            self.num_states = self.gridSize*self.gridSize*2
        else:
            self.num_states = self.gridSize*self.gridSize

        # meta-parameters
        self.epsilon = 1.0; # e-greedy
        self.temp = 1.0; # soft-max temp        
        self.alpha = 0.01; # learning rate
        self.gamma = 0.7; # discount factor
        self.carried_flag = None;
        
        self.q = np.zeros((self.num_states, env.action_space.n)); 

        if(self.PLOT_V):
            self.init_value_plot()

        ## initialize trajectory details ##
        self.tau_num = tau_num;
        self.tau_len = tau_len;

        self.TAU_S = np.zeros((self.tau_len, self.tau_num))-1 # matrix of states with all trajectories
        self.TAU_A = np.zeros((self.tau_len, self.tau_num))-1 # matrix of actions with all trajectories

        self.tau_s = np.zeros((self.tau_len))-1
        self.tau_a = np.zeros((self.tau_len))-1
        self.tau_time=0;
        self.tau_episode=0
        
    def reset(self,env,RANDOM_RESET):
        #print("done!"+str(env.stepCount)+" steps")
        env.reset(RANDOM_RESET)
        self.tau_s = np.zeros((self.tau_len))-1
        self.tau_a = np.zeros((self.tau_len))-1
        self.tau_time = 0

        
    ###################
    ##### core RL #####
    ###################

    def get_action(self, env):
        if(self.HEIRARCHICAL):
            s = env.agentPos[0] + self.gridSize*env.agentPos[1] + int((env.carrying!=None)*self.num_states/2);
        else:
            s = env.agentPos[0] + self.gridSize*env.agentPos[1];
            
        if(self.EGREEDY):
            if random.uniform(0, 1) <= self.epsilon:
                return env.action_space.sample()
            else:
                return np.argmax(self.q[s,:])
        else:
            pi = np.exp(10.0*(self.q[s,:]-np.max(self.q)))
            pi = pi/np.sum(pi)
            return np.random.choice(range(env.action_space.n), p=pi)

    def update_q(self,s,a,r,s_prime):
        self.q[s,a] = self.q[s,a] + self.alpha*(r + self.gamma*np.max(self.q[s_prime,:]) - self.q[s,a])

    ########################
    ##### Trajectories #####
    ########################
    
    def record_tau(self,state,action):
        if(self.tau_time>self.tau_len):
            print("Time exceeded for storing trajectories")
        if(self.HEIRARCHICAL):
            self.tau_s[self.tau_time] = int(state%(self.gridSize*self.gridSize));
        else:
            self.tau_s[self.tau_time] = int(state);            
        self.tau_a[self.tau_time] = int(action);
        self.tau_time += 1
        
    def store_tau(self,episode):
        self.TAU_S[:,self.tau_episode] = self.tau_s
        self.TAU_A[:,self.tau_episode] = self.tau_a

        print("\ntau(",self.tau_episode,"): ",sep='',end='')
        for s in self.tau_s:
            print(int(s),',',sep='',end='')
        self.tau_episode += 1

    def get_tau(self):
        return (self.TAU_S,self.TAU_A)
        
    ####################
    ##### plotters #####
    ####################
    
    def init_value_plot(self):

        fig = plt.figure(figsize=(5,10))
        
        # get initial plot config
        self.ax1 = fig.add_subplot(111);
        self.ax1.set_autoscale_on(True);
        
        # get value from q-function
        q_max = np.max(self.q,1)
        v = np.reshape(q_max,(self.gridSize*2,self.gridSize))
        
        # plot value function
        self.v1_plotter = plt.imshow(v,interpolation='none', cmap='viridis', vmin=v.min(), vmax=v.max());
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.colorbar();

        plt.title('true value function'); plt.ion(); plt.show();        
        
    def see_value_plot(self):
        q_max = np.max(self.q,1)                
        v1 = np.reshape(q_max,(self.gridSize*2,self.gridSize))        
        self.v1_plotter.set_data(v1)

        plt.clim(np.min(v1),np.max(v1)) 
        plt.draw(); plt.show()
        plt.pause(0.0001)
        
    ###################
    ##### update ######
    ###################

    # returns (episode done?, main goal reached?)
    def update(self,env,episode,STORE):
        
        if(STORE):
            self.temp = 10.0;
        else:
            self.temp = 1.0;        

        if(self.HEIRARCHICAL):
            s = env.agentPos[0] + self.gridSize*env.agentPos[1] + int((env.carrying!=None)*self.num_states/2);
        else:
            s = env.agentPos[0] + self.gridSize*env.agentPos[1]

        a = self.get_action(env)

        if (env.agentPos[0]<0 and env.agentPos[0]>=env.gridSize and env.agentPos[1]<0 and env.agentPos[1]>env.gridSize):
            print("What's wrong with state values?!")

        obs, r, done, info = env.step(a)

        main_task_done = r

        # sub-goal
        if(self.HEIRARCHICAL):
            if(env.carrying!=None and self.carried_flag==None): #if reached sub-goal
                r = 1 # sub-goal reached
            self.carried_flag = env.carrying                        
        
        if(self.HEIRARCHICAL):
            s_prime = env.agentPos[0] + self.gridSize*env.agentPos[1] + int((env.carrying!=None)*self.num_states/2);
        else:
            s_prime = env.agentPos[0] + self.gridSize*env.agentPos[1];

        self.update_q(s,a,r,s_prime)

        if done and self.PLOT_V and episode%100==0:
            self.see_value_plot()
            
        if(STORE):
            self.record_tau(s,a); 
            if(main_task_done):                
                self.record_tau(s_prime,env.action_space.sample());

                self.store_tau(episode);
                return done,True 
                
        return done,False
        
            

