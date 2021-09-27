####Deep deterministic policy gradient

import numpy as np
import tensorflow as tf
from collections import deque
import trfl
import os
import shutil
from Continuum_Cellspace_Moving_Source import *

Number_Agent = 1
Exit.append( np.array([0.5, 0.5, 0.5]) )  ##Source

output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
            
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)
    
output_dir = output_dir + '/Moving_Source_DDPG_local'
model_saved_path = model_saved_path + '/Moving_Source_DDPG_local'    
name_main = 'main_DDPG_local_Moving_Source'    
name_target = 'target_DDPG_local_Moving_Source'

class Network:
    def __init__(self, name, learning_rate_actor=0.0001, learning_rate_critic=0.0001,
                 action_size=1, value_size = 1):
        
        self.name = name
        
        self.actor_name = self.name + '/actor'
        self.critic_name = self.name + '/critic'
        
        # state inputs to the Q-network
        with tf.variable_scope(self.actor_name):
            
            self.actor_inputs = tf.placeholder(tf.float32, [None, 4], name='inputs')  
            
            self.a1 = tf.contrib.layers.fully_connected(self.actor_inputs, 32, activation_fn=tf.nn.relu)
            self.a2 = tf.contrib.layers.fully_connected(self.a1, 32, activation_fn=tf.nn.relu)
            self.a3 = tf.contrib.layers.fully_connected(self.a2, 32, activation_fn=tf.nn.relu)

            self.action = tf.contrib.layers.fully_connected(self.a3, action_size, activation_fn=tf.nn.tanh)

        with tf.variable_scope(self.critic_name):
            
            
            self.critic_inputs = tf.concat([self.actor_inputs, self.action], axis =1) 

            self.c1 = tf.contrib.layers.fully_connected(self.critic_inputs, 64, activation_fn=tf.nn.relu)
            self.c2 = tf.contrib.layers.fully_connected(self.c1, 128, activation_fn=tf.nn.relu)
            self.c3 = tf.contrib.layers.fully_connected(self.c2, 64, activation_fn=tf.nn.relu)

            self.Q = tf.contrib.layers.fully_connected(self.c3, value_size, activation_fn=None)            
            self.target_Q = tf.placeholder(tf.float32, [None, value_size], name='targets')

        self.actor_param = self.get_actor_network_variables()  
        self.critic_param = self.get_critic_network_variables()
        
        self.actor_loss = tf.reduce_mean(-self.Q)
        self.critic_loss = tf.losses.mean_squared_error(self.Q, self.target_Q)
        
        self.actor_opt = tf.train.AdamOptimizer(learning_rate=learning_rate_actor).minimize(self.actor_loss, 
                                               var_list= self.actor_param)
        self.critic_opt = tf.train.AdamOptimizer(learning_rate=learning_rate_critic).minimize(self.critic_loss, 
                                               var_list= self.critic_param)
            
    def get_actor_network_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.actor_name)] 
  
    def get_critic_network_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.critic_name)] 


class OUNoise:
    def __init__(self, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.01, decay_period=5000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = 1
        self.low          = -1
        self.high         = 1
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


####Memory replay 
class Memory():
    def __init__(self, max_size = 500):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        
        if len(self.buffer) < batch_size:
            return self.buffer
        
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]


if __name__ == '__main__':
    
    train_episodes = 100000        # max number of episodes to learn from
    max_steps = 1000                # max steps in an episode
    source_steps = 1000
    gamma = 0.                   # future reward discount
    decay_period = 50000

    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.1            # minimum exploration probability 
    decay_percentage = 0.5          
    decay_rate = 4/decay_percentage ####exploration decay rate
            
    # Network parameters
    learning_rate_actor = 1e-4         # Q-network learning rate 
    learning_rate_critic = 1e-3         # Q-network learning rate
    
    # Memory parameters
    memory_size = 10000          # memory capacity
    batch_size = 50                # experience mini-batch size
    pretrain_length = batch_size   # number experiences to pretrain the memory
    
    #target QN
    update_target_every = 1   ###target update frequency
    tau = 0.01                 ###target update factor
    save_step = 1000          ###steps to save the model
    train_step = 50            ###steps to train the model
    
    Cfg_save_freq = 1000       ###Cfg save frequency (episode)
    
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut= 2.0, dt=delta_t, Number=Number_Agent,
                     source_total_steps = source_steps)
    noise = OUNoise(decay_period = decay_period)
    memory = Memory(max_size=memory_size)
        
    tf.reset_default_graph()
    
    ####set mainQN for training and targetQN for updating
    mainNetwork = Network(name=name_main, learning_rate_actor=learning_rate_actor,
                          learning_rate_critic = learning_rate_critic)
    targetNetwork = Network(name=name_target, learning_rate_actor=learning_rate_actor,
                            learning_rate_critic = learning_rate_critic)

    #TRFL way to update the target network
    target_actor_network_update_ops = trfl.update_target_variables(targetNetwork.actor_param,
                                                                   mainNetwork.actor_param,
                                                                   tau=tau)

    target_critic_network_update_ops = trfl.update_target_variables(targetNetwork.critic_param,
                                                                   mainNetwork.critic_param,
                                                                   tau=tau)
    
    copy_actor_network_update_ops = trfl.update_target_variables(targetNetwork.actor_param,
                                                                   mainNetwork.actor_param,
                                                                   tau=1.0)

    copy_critic_network_update_ops = trfl.update_target_variables(targetNetwork.critic_param,
                                                                   mainNetwork.critic_param,
                                                                   tau=1.0)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep = 10) 
    
    ######GPU usage fraction
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    with tf.Session(config = config) as sess:
        
        ####check saved model to continue or start from initialiation
        if not os.path.isdir(model_saved_path):
            os.mkdir(model_saved_path)
        
        checkpoint = tf.train.get_checkpoint_state(model_saved_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            
            print("Removing check point and files")
            for filename in os.listdir(model_saved_path):
                filepath = os.path.join(model_saved_path, filename)
                
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)
                
            print("Done")
            
        else:
            print("Could not find old network weights. Run with the initialization")
            sess.run(init)
            sess.run(copy_actor_network_update_ops)
            sess.run(copy_critic_network_update_ops)
        ####
        
        step = 0    
        loss = 1.
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        for ep in range(1, train_episodes+1):
            total_reward = 0
            t = 0
            state = env.reset()
            noise.reset()
            
            if ep % Cfg_save_freq ==0:
                
                pathdir = os.path.join(output_dir, 'case_' + str(ep) )             
                if not os.path.isdir(pathdir):
                    os.mkdir(pathdir)
                    
                env.save_output(pathdir + '/s.' + str(t))
            
            while t < max_steps:
                
                env.Exit[0][0] += env.vl
                env.Exit[0][1] += env.vl
                
                feed_state = np.array(state)
                feed_state[:2] =  env.Normalization_XY(feed_state[:2])  
                    
                ###### deterministic policy    
                feed = {mainNetwork.actor_inputs: feed_state[np.newaxis, :]}
                
                #####with noise
                action = sess.run(mainNetwork.action, feed_dict=feed)[0] 
                action = noise.get_action(action, t = ep)[0]    
                    
                t = min(t, source_steps)
                next_state, reward, done, ext = env.step_continuous_moving_source(action, t)
            
                total_reward += reward
                step += 1
                t += 1
                
                feed_next_state = np.array(next_state)
                feed_next_state[:2] = env.Normalization_XY(feed_next_state[:2])               
                
                memory.add((feed_state, action, reward, feed_next_state, done))
                
                if done:
                    # Start new episode
                    if ep % Cfg_save_freq ==0:
                        env.save_output(pathdir + '/s.' + str(t))
                    break

                else:

                    state = next_state
                    
                    if ep % Cfg_save_freq ==0:
                        if t%cfg_save_step ==0:
                            env.save_output(pathdir + '/s.' + str(t))
            
                if len(memory.buffer) == memory_size and t%train_step==0:
                    # Sample mini-batch from memory
                    batch = memory.sample(batch_size)
                    states = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch])
                    next_states = np.array([each[3] for each in batch])
                    finish = np.array([each[4] for each in batch])
                    
                    # Train network
                    target_Q = sess.run(targetNetwork.Q, feed_dict={targetNetwork.actor_inputs: next_states})
                    ####End state has 0 action values
                    target_Q[finish == True] = 0.
                    target_Q = rewards[:,np.newaxis] + gamma * target_Q
                    
                    #update critic network
                    loss, _ = sess.run([mainNetwork.critic_loss, mainNetwork.critic_opt],
                                        feed_dict={mainNetwork.actor_inputs: states,
                                                   mainNetwork.target_Q: target_Q})
                    
                    #update actor network                    
                    sess.run([mainNetwork.actor_opt],
                                 feed_dict={mainNetwork.actor_inputs: states})
                    
                    ####update targetnetwork
                    sess.run(target_actor_network_update_ops)
                    sess.run(target_critic_network_update_ops)
                    
            if len(memory.buffer) == memory_size:
                print("Episode: {}, Loss: {}, steps per episode: {}, exit : {}".format(ep,loss, t, ext))
                
            if ep % save_step ==0:
                saver.save(sess, os.path.join(model_saved_path, "Evacuation_Continuum_model.ckpt"), global_step = ep)
            
            
        saver.save(sess, os.path.join(model_saved_path, "Evacuation_Continuum_model.ckpt"), global_step= train_episodes)
 
