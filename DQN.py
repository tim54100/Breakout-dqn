
# coding: utf-8

# In[38]:

import tensorflow as tf
import numpy as np
from DQN_N import DeepQNetwork
import sys
sys.path.append("/home/eastgeno/workspace/gym")
sys.path.append("/home/eastgeno/anaconda2/lib/python2.7/site-packages")
import cv2
import gym
from gym.wrappers import SkipWrapper
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box


# In[39]:
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
                                 width=80, height=80, grayscale=True)
    return e

if __name__ == "__main__":
    #creat a env,the example is Breakout-v0
    env = make_env()
    RL = DeepQNetwork(env.action_space.n-1,
            str(env.observation_space)[4:-1].split(','),
            learning_rate=0.01,
            reward_decay=0.9,
            epsilon_start=0.9,
            replace_target_iter=2000,
            memory_size=20000,
            )
    step = 0
    for episode in range(3000):
        state = env.reset()
        
        #state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        #ret, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        #state = np.reshape(state, (80, 80))
        state = np.stack((state, state, state, state), axis=2)
        #print(state.shape)
        
        action_tracker=np.zeros(env.action_space.n)
        total_reward=0
        done = False
        health=5 
        start=False

        while not done:
            #env.render()
            action_probs = RL.make_policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)+1
            #print(action_probs)
    
            if start and health!=info['ale.lives']:
                health=info['ale.lives']
                action=1
            else:
                start=True
            action_tracker[action]+=1
            
            n_state, reward, done, info = env.step(action)
            #n_state = cv2.cvtColor(cv2.resize(n_state, (80, 80)), cv2.COLOR_BGR2GRAY)
            #ret, n_state = cv2.threshold(n_state, 1, 255, cv2.THRESH_BINARY)
            #n_state = np.reshape(n_state, (80, 80, 1))
            n_state = np.append(state[:, :, 1:], np.expand_dims(n_state, 2), axis=2)
            total_reward+=reward
            RL.store_transition(state, action-1, reward, done, n_state)
            
            if (step > 2000):
                RL.learn()
                
            state=n_state
            
            step+=1
        print('episode: %d/3000, epsilon: %f, total_reward: %d' % (episode, RL.epsilon, total_reward))
        print('action'+'action'.join(str(i)+': '+str(action_tracker[i])[:-2]+'  ' for i in range(len(action_tracker))))
    #RL.plot_cost()


# In[27]:


# In[ ]:




# In[ ]:



