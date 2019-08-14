#Import packages and dependencies
import random

import numpy as np
np.set_printoptions(precision=3)

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from collections import deque

class DDQN:
    def __init__(self, num_states, num_actions, gamma, alpha):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha

        self.curr_epsilon = 1
        self.decay = 0.999
        self.epsilon_min = 0.005

        self.minibatch = 32
        self.epochs = 1
        self.memory = deque(maxlen=7000)

        self.target_model = self.create_model()
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()

        #Hidden Layers
        model.add(Dense(64, input_dim = self.num_states, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))

        #Output Layer
        model.add(Dense(self.num_actions, activation = 'linear'))
        
        model.compile(loss = 'mse', optimizer = Adam(lr=  self.alpha))
        
        return model

    def experience_replay(self):
        minibatch = np.array(random.sample(self.memory, self.minibatch))
        
        output = np.copy(minibatch[:, 2])
        noncomplete = np.where(minibatch[:, 4] == False)

        #Separate Update for non-termianl
        if len(noncomplete[0]) > 0:
            predict_state_prime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
            predict_state_prime = self.model.predict(np.vstack(minibatch[:, 3]))
            
            output[noncomplete] += np.multiply(self.gamma, predict_state_prime_target[noncomplete, np.argmax(predict_state_prime[noncomplete, :][0], axis=1)][0])

        output_target = self.model.predict(np.vstack(minibatch[:, 0]))
        output_target[range(self.minibatch), np.array(minibatch[:, 1], dtype=int)] = output

        #self.model.fit(np.vstack(minibatch[:, 0]), output_target, epochs = self.epochs, verbose = 0)
        self.model.fit(np.vstack(minibatch[:, 0]), output_target, batch_size = self.minibatch, verbose = 0)

    def actionChoice(self, s):
        #Exploration vs Exploitation
        if self.curr_epsilon > np.random.rand():
            return np.random.choice(self.num_actions)

        return np.argmax(self.model.predict(s)[0])
