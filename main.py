
# coding: utf-8

# In[1]:


import numpy as np
import os
import random
import sys
import cv2
import time

import pickle
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
from DQN_N import DeepQNetwork


# In[2]:


def pltin(x, y, x_name, y_name, file_path):
    plt.figure()
    plt.plot(x, y,'--*b')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(file_path)


# In[3]:


def skip_step(frames_to_skip, env, step, action):
    reward = 0
    for i in range(frames_to_skip):
        state, r, done, info = step(action)
        reward += r 
        if done:
            break
    return state, reward, done, info


# In[4]:


def preprocess(img, width, height):
    #img = cv2.cvtColor(cv2.resize(img, (width, length)), cv2.COLOR_RGB2GRAY)
    img = np.array(img, dtype = np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (width, height))
#     img = np.vstack(img)
    img = img.astype(np.float)
    img /= 255
    
    return img


# In[5]:


env = gym.make('MsPacman-v0')# Breakout-ram-v0

width = 84
height = 84
batch_size = 32
memory_size = 80000
replace_iter = 4000
frame_repeat = 4
n_a = env.action_space.n
push = False
        
step = 0    
e_step = []
e_score = []
e_loss = []
epsilon = []
episodes = 100
num_game = 0

RL = DeepQNetwork(n_a,
                  st_shape = [width, height],
                  learning_rate= 0.0001,
                  reward_decay =0.99,
                  epsilon_start= 1,
                  epsilon_end = 0.1,
                  explore = 500000,
                  replace_target_iter= replace_iter,
                  memory_size= memory_size,
                  batch_size = batch_size,
                  training = True,
                  output_graph = False,)


# In[6]:


while(True):
    total_loss = 0
    total_score = 0
    with tqdm(total=episodes) as t:
        for episode in range(episodes):


            action_tracker = np.zeros(n_a)
            loss = 0
            i=0
            score = 0
            done = False
            lives = -1
            info = {}
#             info['ale.lives'] = 5
            info['ale.lives'] = 3
            
            s = env.reset()
#             s = np.array(s,dtype='float')
            s = preprocess(s, width, height)
            s = np.stack((s, s, s, s), axis=2)
#                 print(s.shape)

            while not done:
                action = RL.make_policy(s)
                
                if info['ale.lives'] != lives:
                    lives = info['ale.lives']
#                     action=1
                    #s = True
                    
                action_tracker[action] += 1
                n_s, reward, done, info = skip_step(frame_repeat, env, env.step, action)
#                 n_s = np.array(n_s,dtype='float')
                n_s = preprocess(n_s, width, height)
                n_s = np.append(s[:, :, 1:], np.expand_dims(n_s, 2), axis=2)
                score += reward
                    
                if info['ale.lives'] != lives:
                    reward = -10
                    #s = True
                    
                reward /= 10
                reward = np.max([-1 , np.min([1, reward])])
                RL.store(s, action, reward, n_s, not done)

                s = n_s
                if (len(RL.replay_memory) > RL.memory_size/4 and step%4 == 0):
                    los = RL.learn()
#                     print(los)
#                     los,TD,p_loss = RL.learn()
#                     print("TD: %f, p_loss: %f"%(np.reducemean(TD), p_loss))
                    loss += los
                    i+=1
                step += 1
#                     if step %1000 == 0:
#                          print('total_step: %d, step: %d, current_score: %f' % (step, i*4, score))

            loss = loss/i if (i != 0 ) else 0
            total_score += score
            total_loss += loss
#                 print('episode: %d, epsilon: %f, score: %f, loss: %f'%(num_game+episode+1, RL.epsilon, score, loss))
#                 print('action'+'action'.join(str(i)+': '+str(action_tracker[i])[:-2]+'  ' for i in range(len(action_tracker))))


            t.set_postfix(score=score, loss = loss, step=step)
            t.update()
#         if RL.epsilon == RL.epsilon_end and RL.epsilon_end != 0.001:
#             RL.learning_rate= 0.0001
#             RL.epsilon_start /= 10
#             RL.explore = 2000000
#             RL.epsilon_end /= 10

        num_game += episodes

    if push:
        e_step.append(RL.learn_step_counter/10000)
        e_score.append(total_score/episodes)
        epsilon.append(RL.epsilon)
        e_loss.append(total_loss/episodes)
        path = './picture/'+str(RL.learn_step_counter)
        file = open('./saved_networks/train_history.pickle', 'wb')
        pickle.dump([e_step, e_score, epsilon, e_loss], file)
        file.close()
#         pltin(e_step, e_score, 'step(10000)' , 'score', path + '_score.png')
#         pltin(e_step, e_loss, 'step(10000)' , 'loss', path + '_loss.png')
#         plt.close('all')
    if total_loss != 0:
        push = True
#         if not os.path.exists('./picture'):
#             os.makedirs('picture')

