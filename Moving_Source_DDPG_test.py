####Test of trained model for evacuation

import numpy as np
import tensorflow as tf
import os
import shutil
from Continuum_Cellspace_Moving_Source import *

Number_Agent = 40
delta_t = 0.1

######4Exits
Exit.append( np.array([0.5, 0.5, 0.5]) )  
#Exit.append( np.array([0.5, 0.0, 0.5]) )  ##Down
#Exit.append( np.array([0, 0.5, 0.5]) )  ##Add Left exit
#Exit.append( np.array([1.0, 0.5, 0.5]) )     ##Add Right Exit

output_dir = './Test'
model_saved_path = './model'


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
            
    def get_actor_network_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.actor_name)] 
  
    def get_critic_network_variables(self):
      return [t for t in tf.trainable_variables() if t.name.startswith(self.critic_name)] 



if __name__ == '__main__':
    
    test_episodes = 1        # max number of episodes to test
    max_steps = 1000                # max steps in an episode
    gamma = 0.0                  # future reward discount
    source_steps = 1000

    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.1            # minimum exploration probability 
    decay_percentage = 0.5        
    decay_rate = 4/decay_percentage
            
    # Network parameters
    learning_rate_actor = 1e-4 
    learning_rate_critic = 1e-4         # Q-network learning rate 
    
    # Memory parameters
    memory_size = 10000          # memory capacity
    batch_size = 50                # experience mini-batch size
    pretrain_length = batch_size   # number experiences to pretrain the memory    
    
    Cfg_save_freq = 1
    cfg_save_step = 2
    
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut= 2.0, dt=delta_t, Number=Number_Agent,
                     source_total_steps = source_steps)
        
    tf.reset_default_graph()
    mainNetwork = Network(name=name_main, learning_rate_actor=learning_rate_actor,
                          learning_rate_critic = learning_rate_critic)

    Actor_list = mainNetwork.get_actor_network_variables()
    Critic_list = mainNetwork.get_critic_network_variables()
    
    init = tf.global_variables_initializer()
    
    saver_actor = tf.train.Saver(Actor_list)
    saver_critic = tf.train.Saver(Critic_list)

    ######GPU usage fraction
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    with tf.Session(config = config) as sess:   
        
        sess.run(init)
        ####check saved model
        if not os.path.isdir(model_saved_path):
            os.mkdir(model_saved_path)
            
            
        checkpoint = tf.train.get_checkpoint_state(model_saved_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver_actor.restore(sess, checkpoint.model_checkpoint_path)
            saver_critic.restore(sess, checkpoint.model_checkpoint_path)
            
#            saver_actor.restore(sess, checkpoint.all_model_checkpoint_paths[9])
#            saver_critic.restore(sess, checkpoint.all_model_checkpoint_paths[9])
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            

        ##########test run
        step = 0     
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)  
        
        for ep in range(0, test_episodes):
            total_reward = 0
            t = 0
            state = env.reset()
            
            print("Testing episode: {}".format(ep))
            
            if ep % Cfg_save_freq ==0:
                
                pathdir = os.path.join(output_dir, 'case_' + str(ep) )             
                if not os.path.isdir(pathdir):
                    os.mkdir(pathdir)
                    
                else:            
                    for filename in os.listdir(output_dir):
                        filepath = os.path.join(output_dir, filename)
                        
                    try:
                        shutil.rmtree(filepath)
                    except OSError:
                        os.remove(filepath)  
                    
                env.save_output(pathdir + '/s.' + str(t))
            
            while t < max_steps:

                # Get action from Q-network    
                
                env.Exit[0][0] += env.vl
                env.Exit[0][1] += env.vl
                
                t = min(t, source_steps)
                reward, done, ext = env.step_continuous_moving_source_all(sess, mainNetwork , t)
                
                step += 1
                t += 1
                
                if done:
                    # Start new episode
                    if ep % Cfg_save_freq ==0:
                        env.save_output(pathdir + '/s.' + str(t))
        
                    break

                else:
                    if ep % Cfg_save_freq ==0:
                        if t%cfg_save_step ==0:
                            env.save_output(pathdir + '/s.' + str(t))
            
            print("Total steps in episode {} is : {}".format(ep, t))