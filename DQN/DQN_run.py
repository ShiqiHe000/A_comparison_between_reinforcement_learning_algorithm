import numpy as np
import gym
from DQN_brain import DQN_network
import matplotlib.pyplot as plt

# create the environment
env=gym.make('FrozenLake-v0')
env = env.unwrapped

# number of episode
num_episode=10000

if __name__ == "__main__":
    DQN = DQN_network(env.action_space.n,
                      env.observation_space.n,
                      learning_rate=0.01,
                      discount=0.9,
                      e_greedy=0.9,
                      replace_iteration=300,
                      memory_size=1000,
                      # output_graph=True
                      )

# store the reward
reward_all_episodes=[]

# define step
step=0

# Q_learning algorithm
for episode in range(num_episode):
    # initialize the observation
    observation=env.reset()
    done=False
    reward_current_episode=0

    for step in range(100): # the maximum step in an episode is 100
        # visualize the env
        env.render()

        # choose an action
        action = DQN.choose_action(observation)

        # get reward and next observation
        observation_, reward, done, info = env.step(action)

        # transition to the next state
        DQN.store_transition(observation,reward,action,observation_)

        # after 300 step the agent begins to learn
        # and after that the agent learns every 10 steps
        if (step>300) and (step%10==0):
            DQN.learn()

        # step to the next state
        observation=observation_

        # break the loop if when it gets to the terminal state
        if done==True:
            break

        step += 1

print('game over')
'''
if __name__ == "__main__":
    DQN = DQN_network(env.actions, env.states,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_iteration=300,
                      memory_size=1000,
                      # output_graph=True
                      )
                      '''

