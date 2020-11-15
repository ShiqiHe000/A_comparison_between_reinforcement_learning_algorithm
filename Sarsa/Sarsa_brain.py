# Q_learing table
import numpy as np

class SarsaTable:
    # initialize the parametere
    def __init__(self, actions, states, learing_rate=0.01, e_greedy=0.9, discount=0.9):
        self.actions=actions # a list
        self.states=states
        self.learning_rate=learing_rate
        self.epsilon=e_greedy
        self.gamma=discount
        # build a s_table
        self.s_table=np.zeros((self.states,self.actions))

    # choose an action
    def choose_action(self,observation):
        # randomly create a number
        threshold=np.random.uniform(0,1)
        if threshold>self.epsilon:
            # exploitation
            action=np.argmax(self.s_table[observation,:])
        else:
            # exploration
            action=np.random.choice(self.actions)
        return action


    # update s_table and learn
    def learn(self, observation, reward, action, observation_, action_):
        self.s_table[observation,action] += self.learning_rate*\
                                            (reward+\
                                             self.gamma*(self.s_table[observation_,action_])-\
                                             self.s_table[observation,action])




