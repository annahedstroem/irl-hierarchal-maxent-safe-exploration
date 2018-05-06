#!/usr/bin/env python3

from __future__ import division, print_function

import sys,time
import numpy as np,gym
from optparse import OptionParser
import gym_minigrid,expert,inverse_agent,hirl
from tempfile import TemporaryFile
import ipdb

def main():
    
    MODE = "inverse"
    risk_mode = False
    
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-CaptureTheFlag-Static-v0',
        #default='MiniGrid-LockedRoom-v0',
    )
    (options, args) = parser.parse_args()

    # trajectory data parameters
    tau_num = 1000; # number of trajectories
    tau_len = 100; # length of each trajectories
    
    # Load the gym environment
    test_env_name = 'MiniGrid-CaptureTheFlag-Test-v0'
    test_env = gym.make(test_env_name)
    
    env = gym.make(options.env_name)
    env.maxSteps = tau_len; # maximum time for an episode = length of our trajectory
        
    if(MODE=="expert"):
        
        # Load expert agent
        q_expert = expert.ExpertClass(env,tau_num,tau_len)

        # training
        for episode in range(25000):
            q_expert.reset(env,True)
            for t in range(tau_len-1):                
                done, r = q_expert.update(env,episode,False)
                if done:
                    q_expert.reset(env,True)
                    break
                #if(episode%1000==0):
                #    env.render('human')
                #    time.sleep(0.05)

            if(episode%1000==0):
                print('Training expert episode:',episode)
                        
        q_expert.reset(env,False)
        
        # testing (store successful expert trajectories)
        success_episode = 0
        while success_episode<tau_num:
            q_expert.reset(env,False)
            for t in range(tau_len-1):
                done, r = q_expert.update(env,episode,True)
                if r: #if main goal reached
                    success_episode += 1                
                if done:  # if episode done
                    q_expert.reset(env,False)
                    break
                #env.render('human')
                #time.sleep(0.05)
                        
        ## get traj    
        TAU = q_expert.get_tau();
        np.save('expert_traj.npy', TAU)
        
    elif(MODE=="inverse"):

        print("inverse mode")
        
        # load traj
        TAU = np.load('expert_traj.npy')
        TAU = TAU[:,:,0:100]            
            
        # load inverse rl agent
        maxent_learner = hirl.HInverseAgentClass(env, test_env, tau_num, tau_len, risk_mode=risk_mode)
        #maxent_learner = inverse_agent.InverseAgentClass(env, test_env, tau_num, tau_len, risk_mode=risk_mode)
        
        ## inverse RL mode: learn MaxEnt IRL from trajectories        
        maxent_learner.store_trajectories(TAU);
        subgoal = maxent_learner.update(env,PRINT=True)

        ############

        TAU_SPLIT = (np.split(TAU[0],[int(subgoal)])[1], np.split(TAU[1],[int(subgoal)])[1]) ## split traj at subgoal and get latter

        # load inverse rl agent
        #maxent_learner = inverse_agent.InverseAgentClass(env, test_env, tau_num, tau_len, risk_mode=risk_mode)        
        maxent_learner = hirl.HInverseAgentClass(env, test_env, tau_num, tau_len, risk_mode=risk_mode)

        ## inverse RL mode: learn MaxEnt IRL from trajectories        
        maxent_learner.store_trajectories(TAU_SPLIT);
        subgoal = maxent_learner.update(env,PRINT=True)

if __name__ == "__main__":
    main()
