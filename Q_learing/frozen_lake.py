import numpy as np
import gym
from Q_table import QLearingTable
import matplotlib.pyplot as plt


# create the environment
env=gym.make('FrozenLake-v0')
env = env.unwrapped

# set the parameters in Qlearingtable
QT = QLearingTable(actions=env.action_space.n,
                  states=env.observation_space.n,
                learing_rate=0.01, e_greedy=0.9,
                  discount=0.9,)

# show the action space
print('The action space of the env:')
print(env.action_space.n)
print('The state space size:')
print(env.observation_space.n)

# number of episode
num_episode=2000

# store the reward
reward_all_episodes=[]

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
        action = QT.choose_action(observation)

        # get reward and next observation
        observation_, reward, done, info= env.step(action)

        # learn from this action
        QT.learn(observation,reward,action,observation_)

        # transition to the next state
        observation=observation_
        reward_current_episode += reward

        if done==True:
            break

    reward_all_episodes.append(reward_current_episode)

print('game over!')
print(reward_all_episodes)

# calculate and print the average reward per 100 episode
# split the reward array
reward_per_hundred_episode=np.split(np.array(reward_all_episodes),num_episode/100)

y=np.zeros(int(num_episode/100))
print('reward per hundred episode:')
for i in range(int(num_episode/100)):
    y[i]=sum(reward_per_hundred_episode[i])/100
    print(i+1,':',str(y[i]))

# plot the trend
x=np.linspace(0,num_episode,num_episode/100)
plt.plot(x,y)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Q_learning')
plt.show()