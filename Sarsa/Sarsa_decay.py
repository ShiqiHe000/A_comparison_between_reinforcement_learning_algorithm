import numpy as np
import gym
import random
import matplotlib.pyplot as plt


# create the environment
env=gym.make('FrozenLake-v0')
env = env.unwrapped

# build q_table
s_table = np.zeros((env.observation_space.n, env.action_space.n))

# number of episode
num_episode = 5000

learning_rate=0.1
gamma=0.95

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01


# store the reward
reward_all_episodes=[]


# choose action function
def choose_action(state):
    # randomly create a number
    exploration_rate_threshold = random.uniform(0, 1)
    if exploration_rate_threshold > exploration_rate:
        action = np.argmax(s_table[state, :])
    else:
        action = env.action_space.sample()
    return  action

# Sarsa algorithm
for episode in range(num_episode):
    # initialize the observation
    state=env.reset()
    done=False
    reward_current_episode=0

    # the first action
    action = choose_action(state)

    for step in range(100): # the maximum step in an episode is 100
        # visualize the env
        #env.render()

        # get reward and next observation
        state_, reward, done, info= env.step(action)

        # choose action for the next state
        action_ = choose_action(state_)

        # learn from this action
        s_table[state, action] += learning_rate *\
                                 (reward +gamma * np.max(s_table[state_, action_])-\
                                  s_table[state, action])

        # transition to the next state
        state = state_
        action = action_
        reward_current_episode += reward

        if done==True:
            break

    # store the rewards
    reward_all_episodes.append(reward_current_episode)
    # epsilon decays
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) *\
                       np.exp(-exploration_decay_rate * episode)

print('game over!')
# print the percentage of the succesful episodes
print('Precent of successful episode: ' + str(sum(100*reward_all_episodes)/num_episode) + '%')

# calculate and print the average reward per 100 episode
# split the reward array
reward_per_hundred_episode=np.split(np.array(reward_all_episodes),num_episode/100)

y=np.zeros(int(num_episode/100))
#print('reward per hundred episode:')
for i in range(int(num_episode/100)):
    y[i]=sum(reward_per_hundred_episode[i])/100
    #print(i+1,':',str(y[i]))

# plot the trend
plt.figure(1)
x=np.linspace(0,num_episode,num_episode/100)
plt.plot(x,y)
plt.xlabel('Episode')
plt.ylabel('Reward per 100 episodes')
plt.title('Sarsa')

plt.figure(2)
plt.plot(reward_all_episodes)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
