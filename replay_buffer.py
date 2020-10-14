import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self,size):
        self.memory = deque(maxlen=size)
        
    def sample(self,batch_size):
        idxs = np.random.choice(np.arange(len(self.memory)),
                                size = min(len(self.memory), batch_size))
        minibatch = [self.memory[idx] for idx in idxs]
        states, actions, rewards, next_states, terminals = map(tuple, zip(*minibatch))
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        return states,actions,rewards,next_states,terminals

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))