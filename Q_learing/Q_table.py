# Q_learing table
import numpy as np

class QLearingTable:
    # initialize the parametere
    def __init__(self, actions, states, learing_rate=0.01, e_greedy=0.1, discount=0.9):
        self.actions=actions # a list
        self.states=states
        self.learning_rate=learing_rate
        self.epsilon=e_greedy
        self.gamma=discount
        # build a q_table
        self.q_table=np.zeros((self.states,self.actions))

    # choose an action
    def choose_action(self,observation):
        # randomly create a number
        threshold=np.random.uniform(0,1)
        if threshold>self.epsilon:
            # exploitation
            action=np.argmax(self.q_table[observation,:])
        else:
            # exploration
            action=np.random.choice(self.actions)
        return action


    # update Q_table and learn
    def learn(self, observation, reward, action, observation_):
        self.q_table[observation,action] += self.learning_rate*\
                                            (reward+\
                                             self.gamma*(np.max(self.q_table[observation_,:]))-\
                                             self.q_table[observation,action])




