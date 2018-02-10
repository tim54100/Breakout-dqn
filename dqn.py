# coding: utf-8


import numpy as np
from DQN_N import DeepQNetwork
import sys
# sys.path.append("/home/eastgeno/workspace/gym")
# sys.path.append("/home/eastgeno/anaconda2/lib/python2.7/site-packages")
import cv2
import pyglet
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from gym.wrappers import SkipWrapper
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box
from gym import wrappers

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        img = np.squeeze(img)
        return img

def make_env():
    env_spec = gym.spec('Breakout-v0')
    env_spec.id = 'Breakout-v0'
    env = env_spec.make()
    e = PreprocessImage(SkipWrapper(4)(env),
                                 width=84, height=84, grayscale=True)
    return e

if __name__ == "__main__":
    #creat a env,the example is Breakout-v0

    env = make_env()
    env = wrappers.Monitor(env, './experiment', force=True)
    RL = DeepQNetwork(env.action_space.n,
            str(env.observation_space)[4:-1].split(','),
            learning_rate= 0.00025,
            reward_decay =0.99,
            epsilon_start= 1,
            epsilon_end = 0.1,
            explore = 1000000,
            replace_target_iter=10000,
            memory_size= 1000000,
            batch_size = 32,
            training = True,
            output_graph = True,
            )
    #env = wrappers.Monitor(env, './experiment', force=True)
    step = 0
    loss_step =0
    RL.epsilon_decrease=False
    num_game = 0
    e_step = []
    e_score = []
    epsilon = []
    while(True):
        loss = 0
        loss_m =0
        loss_step /=100
        for episode in range(1000):
            state = env.reset()
            
        #state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        #ret, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        #state = np.reshape(state, (80, 80))
            state = np.stack((state, state, state, state), axis=2)
        #print(state.shape)
        
            action_tracker=np.zeros(env.action_space.n)
            total_reward=0
            done = False
            lives = -1
            i = 0
            info = {}
            info['ale.lives'] = 5

            while not done:
                #env.render()
                action = RL.make_policy(state)
                #action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                #action = RL.make_policy(state)
                #print(action_probs)
                #s =False
                if info['ale.lives'] != lives:
                    lives = info['ale.lives']
                    action=1
                    #s = True
                action_tracker[action]+=1
                
                n_state, reward, done, info = env.step(action)
            #n_state = cv2.cvtColor(cv2.resize(n_state, (80, 80)), cv2.COLOR_BGR2GRAY)
            #ret, n_state = cv2.threshold(n_state, 1, 255, cv2.THRESH_BINARY)
            #n_state = np.reshape(n_state, (80, 80, 1))
            #print(info['ale.lives'])
                total_reward+=reward
                #if reward == 0:
                #    reward=-0.01
                #else:
                #    reward/=1000
                #if info['ale.lives'] != lives or done:
                #    lives = info['ale.lives']
                #    reward-=9
                #    action_tracker[action]-=1
                #    action = 1
                #reward/=4
                #if lives == -1:
                #    lives = info['ale.lives']
                '''if s:
                    state = n_state
                    state = np.stack((state, state, state, state), axis=2)
                else:
                    n_state = np.append(state[:, :, 1:], np.expand_dims(n_state, 2), axis=2)
                   # n_done = 1 if done else 0
                    if info['ale.lives'] != lives:
                        RL.store_transition(state, action, -1, n_state, True)
                    else:
                        RL.store_transition(state, action, reward, n_state, done)
                    step+=1
                   
                    state=n_state
                    if (step > 50000 and step%4 == 0):
                        loss += RL.learn()
                        RL.epsilon_decrease=True
                        i+=1
                    if step %100 == 0:
                        print('total_step: %d, step: %d' % (step, i*4))'''
                n_state = np.append(state[:, :, 1:], np.expand_dims(n_state, 2), axis=2)
                   # n_done = 1 if done else 0
                if info['ale.lives'] != lives:
                    RL.store_transition(state, action, -1, n_state, True)
                    #lives = info['ale.lives']
                else:
                    RL.store_transition(state, action, reward, n_state, done)
                step+=1
               
                state=n_state
                if (step > RL.memory_size and step%4 == 0):
                    loss += RL.learn()
                    i+=1
                if step > RL.memory_size*0.8:
                    RL.epsilon_decrease=True
                if step %100 == 0:
                     print('total_step: %d, step: %d' % (step, i*4))
                
            loss = loss/i if (i != 0 ) else 0

            '''if episode % 10 == 0:
                loss_m =loss
            if loss_m != 0 and loss != 0 and RL.lr > 1e-06 and loss < 100 and loss >= loss_m and RL.lr != 1e-08:
                loss_step += 1
                if loss_step >=10:
                    RL.lr /= 10
                    loss_step = -10
            elif loss*1.1 < loss_m and RL.lr != 1e-08:
                loss_step -= 1
                if loss_step <=-10 and RL.lr <0.00001:
                    #RL.lr *= 10
                    loss_step = 0'''

            #print('episode: %d, epsilon: %f, total_reward: %d, loss: %f, lr: %s'%(num_game+episode+1, RL.epsilon, \
            #      total_reward, loss, str(RL.lr)))
            print('episode: %d, epsilon: %f, total_reward: %d, loss: %f'%(num_game+episode+1, RL.epsilon, total_reward, loss))
            #print('episode: %d, epsilon: %f, total_reward: %d, lr: %s'%(num_game+episode+1, RL.epsilon, total_reward, str(RL.lr)))
            print('action'+'action'.join(str(i)+': '+str(action_tracker[i])[:-2]+'  ' for i in range(len(action_tracker))))
            if loss != 0:
                e_step.append(RL.learn_step_counter)
                e_score.append(total_reward)
                epsilon.append(RL.epsilon)
                
        if RL.epsilon == RL.epsilon_end and RL.epsilon_end == 0.1:
            RL.explore = 5000000
            RL.epsilon_end = 0.01
        
        num_game+=1000
        if RL.learn_step_counter != 0:

            plt.plot(e_step, e_score)
            plt.ion()#本次运行请注释，全局运行不要注释
            plt.savefig('./picture/'+str(RL.learn_step_counter)+'.png')
            #plt.show()
# RL.plot_cost()
