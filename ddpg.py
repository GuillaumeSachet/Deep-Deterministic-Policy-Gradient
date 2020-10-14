from tensorflow.keras import layers, Model, models, optimizers,losses
import tensorflow as tf
import numpy as np
from replay_buffer import ReplayBuffer
import pickle

class DDPGAgent:
    def __init__(self, state_shape, action_shape, neurons_actor, neurons_critic, act_limit_high, act_limit_low, buffer_size,
                 gamma, lr_actor, lr_critic, act_noise = 0.1, polyak = 0.995, input_is_picture = False):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.input_is_picture = input_is_picture
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.polyak = polyak
        
        self.act_noise = act_noise
        self.act_limit_low = act_limit_low
        self.act_limit_high = act_limit_high
        
        self.actor_optimizer = optimizers.Adam(learning_rate = self.lr_actor)
        
        self.critic = self.create_critic(neurons_critic)
        self.actor = self.create_actor(neurons_actor)
        self.critic_target = self.create_critic(neurons_critic)
        self.actor_target = self.create_actor(neurons_actor)
        self.critic_target.set_weights(self.critic.get_weights())
        self.actor_target.set_weights(self.actor.get_weights())
        
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        print(self.critic.summary())
        print(self.actor.summary())
        self.total_replay = 0

        self.last_state = None
        self.last_action = None

    def create_critic(self, neurons):
        state_input = layers.Input(shape = self.state_shape)
        action_input = layers.Input(shape = self.action_shape)
        if self.input_is_picture:
            x = layers.Conv2D(8, (3, 3), activation='relu', padding = "same")(state_input)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(16, (3, 3), activation='relu', padding = "same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(32, (3, 3), activation='relu', padding = "same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Flatten()(x)
            for i in range(1, len(neurons)):
                x = layers.Dense(neurons[i], activation='relu')(x)
            x = layers.Concatenate()([x,action_input])
        else:
            x = layers.Concatenate()([state_input,action_input])
            x = layers.Dense(neurons[0], activation='relu')(x)
            for i in range(1, len(neurons)):
                x = layers.Dense(neurons[i], activation='relu')(x)
                
        output = layers.Dense(1)(x)

        critic = Model([state_input,action_input],output)
        critic.compile(loss = 'mse', optimizer = optimizers.Adam(learning_rate = self.lr_critic))
        return critic
    
    def create_actor(self, neurons):
        
        state_input = layers.Input(shape = self.state_shape)
        if self.input_is_picture:
            x = layers.Conv2D(8, (3, 3), activation='relu', padding = "same")(state_input)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(16, (3, 3), activation='relu', padding = "same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(32, (3, 3), activation='relu', padding = "same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Flatten()(x)
            for i in range(1, len(neurons)):
                x = layers.Dense(neurons[i], activation='relu')(x)
        else:
        
            x = layers.Dense(neurons[0], activation='relu')(state_input)
            for i in range(1, len(neurons)):
                x = layers.Dense(neurons[i], activation='relu')(x)
        
        output = layers.Dense(self.action_shape[0], activation='tanh')(x)

        actor = Model(state_input,output)
        return actor
    
    def policy(self, state):
        return self.actor.predict(state).squeeze()
    
    def get_action(self, state):
        action = self.policy(state)
        action += self.act_noise * np.random.randn(self.action_shape)[0]
        return np.clip(action, self.act_limit_low, self.act_limit_high)
        
    def step(self, reward, state, done):
        self.replay_buffer.add(self.last_state, self.last_action, reward, state, done)
        self.last_state = state
        self.last_action = self.get_action(state)
        return self.last_action
    
    def reset(self, state):
        self.last_state = state
        self.last_action = self.get_action(state)
        return self.last_action
    
    def experience_replay(self, batch_size, num_replay):
        self.total_replay += num_replay
        if len(self.replay_buffer.memory) > 1000:
            for replay in range(num_replay):
                states,actions,rewards,next_states,terminals = self.replay_buffer.sample(batch_size)
                states = tf.convert_to_tensor(states)
                next_states = tf.convert_to_tensor(next_states)
                critic_input = tf.concat([states,actions], axis=-1)
                critic_target_input = tf.concat([next_states,self.actor_target.predict(next_states)], axis=-1)
                targets = rewards + self.gamma * (1 - terminals) * self.critic_target.predict(critic_target_input).reshape(-1)
                self.critic.fit(critic_input, targets, batch_size = batch_size, epochs = 1, verbose = 0)
                
                with tf.GradientTape() as tape:
                    actions_pred = self.actor(states)
                    q_values = self.critic(tf.concat([states,actions_pred], axis=-1))
                    loss = -tf.reduce_mean(q_values)
                    
                actor_grads = tape.gradient(loss, self.actor.trainable_variables)
                grads_and_vars = zip(actor_grads,self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(grads_and_vars)
                
                for i in range(len(self.actor_target.weights)):
                    self.actor_target.weights[i].assign(self.polyak*self.actor_target.weights[i] + (1-self.polyak)*self.actor.weights[i])
                    self.critic_target.weights[i].assign(self.polyak*self.critic_target.weights[i] + (1-self.polyak)*self.critic.weights[i])
                    
            return 'Experience replay done {} times'.format(num_replay)
        else :
            return 'Not enough memory in replay buffer'
                
    def save_agent(self, file):
        filehandler = open(file+'_replay_buffer.pickle', 'wb') 
        pickle.dump(self.replay_buffer, filehandler)
        filehandler = open(file+'_total_replay.pickle', 'wb') 
        pickle.dump(self.total_replay, filehandler)
        self.critic.save_weights(file+'_critic.h5')
        self.critic_target.save_weights(file+'_critic_target.h5')
        self.actor.save_weights(file+'_actor.h5')
        self.actor_target.save_weights(file+'_actor_target.h5')
        print('Saved')
    
    def load_agent(self,file):
        filehandler = open(file+'_replay_buffer.pickle', 'rb') 
        self.replay_buffer = pickle.load(filehandler)
        filehandler = open(file+'_total_replay.pickle', 'rb') 
        self.total_replay = pickle.load(filehandler) 
        self.critic.load_weights(file+'_critic.h5')
        self.critic_target.load_weights(file+'_critic_target.h5')
        self.actor.load_weights(file+'_actor.h5')
        self.actor_target.load_weights(file+'_actor_target.h5')
        print('Loaded')