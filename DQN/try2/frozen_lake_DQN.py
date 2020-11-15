 import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load the env
env=gym.make('FrozenLake-v0')

# implement the network
tf.reset_default_graph()
# the feed-forward part for the network used to choose actions
inputs1=tf.placeholder(shape=[1,16], dtype=tf.float32)
W=tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout=tf.matmul(inputs1,W) # matrice product
predict=tf.argmax(Qout,1)

# obtain the loss value
nextQ=tf.placeholder(shape=[1,4], dtype=tf.float32)
loss=tf.reduce_sum(tf.square(nextQ-Qout))
trainer=tf.train.GradientDescentOptimizer(learning_rate=0.1) # gradient decent
updateModel=trainer.minimize(loss) # minimize the loss function

# training the network
init=tf.global_variables_initializer() # creat a node for initializing all the variables

# set learning parameters
gamma=0.95
num_episode=2000
epsilon=0.1
# create a list to contain total rewards and steps per episode
rlist=[]
jlist=[]

# run the game
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episode):
        # reset environment and get first observation
        s=env.reset()
        rAll=0
        done = False
        j=0

        # the Q network
        for j in range(100):
            j += 1
            # choose an action with greedy with e chance of random action
            a, allQ = \
                sess.run([predict,Qout], feed_dict={inputs1:np.identity(16)[s:s+1]})

            if np.random.rand(1)<epsilon:
                a[0]=env.action_space.sample()


            # get a new state and reward
            s_, reward, done,info=env.step(a[0])

            # obtain the Q_ value by feeding the new state to thte network
            Q_=sess.run(Qout,feed_dict={inputs1:np.identity(16)[s_:s_+1]})

            # obtain maxQ_ and set target value for chosen action
            maxQ_=np.max(Q_)
            targetQ=allQ
            targetQ[0,a[0]]=reward+gamma*maxQ_

            # train the network using target and predicted Q values
            _,W1 = \
                sess.run([updateModel, W], \
                         feed_dict={inputs1:np.identity(16)[s:s+1], nextQ:targetQ})
            rAll += reward
            s=s_
            if done==True:
                break

        # reward list
        rlist.append(rAll)
        jlist.append(j)

# print the percentage of the succesful episodes
print('Precent of successful episode: ' + str(sum(rlist)/num_episode) + '%')

# calculate and print the average reward per 100 episode
# split the reward array
reward_per_hundred_episode=np.split(np.array(rlist),num_episode/100)

y=np.zeros(int(num_episode/100))
#print('reward per hundred episode:')
for i in range(int(num_episode/100)):
    y[i]=sum(reward_per_hundred_episode[i])/100
    #print(i+1,':',str(y[i]))

# plot for the reward and
plt.figure(1)
plt.plot(rlist)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('DQN reward result')

plt.figure(2)
plt.plot(jlist)
plt.xlabel('Episodes')
plt.ylabel('Steps per episode')
plt.title('DQN steps per episode result')

plt.figure(3)
x=np.linspace(0,num_episode,num_episode/100)
plt.plot(x,y)
plt.xlabel('Episode')
plt.ylabel('Reward per 100 episodes')
plt.title('DQN learning')
plt.show()










