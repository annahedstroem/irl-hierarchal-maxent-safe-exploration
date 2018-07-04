# irl-hierarchal-maxent-safe-exploration

Implementation of a updated softmax Maxmium Entropy algo for 2D capture the flag (CTP) set up for inverse reinforcement learning. 

The agent successfully learned to solve hierarchal tasks, that is learned all its subgoals, while ensuring safe exploration (avoiding risky states in the state space). 

We used the interface by Minimalistic gridworld environment for OpenAI Gym (https://github.com/maximecb/gym-minigrid) but modified it for three main scenarios for CTP.

An example configuration for testing :

![Screenshot1](plots/corridor/corridor.png)
