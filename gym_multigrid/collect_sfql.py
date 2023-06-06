import gymnasium as gym
import time
from gymnasium.envs.registration import register
from matplotlib import animation
import matplotlib.pyplot as plt
from gym_multigrid.agent import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class SFQLAgent:
    def __init__(self, env, state_dim, action_dim, feats, learning_rate, discount_factor):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.psi = np.random.uniform(-0.01, 0.01, size=(state_dim, action_dim, feats))
        self.n_actions = action_dim
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.2
        self.w = np.array([-1, 1.25, 1.25])
        self.env = env

    def update_psi(self, transitions):
        running_err = 0.0
        for state, action, phi, next_state, next_action in transitions:
            #print(state.shape)
            big_phi = self.env.gaussian(next_state).flatten()
            psi_temp = np.einsum('i,ijk->jk', big_phi, self.psi)
            targets = phi.flatten() + self.discount_factor * psi_temp[next_action,:]
            big_phi = self.env.gaussian(state).flatten()
            cur = np.einsum('i,ijk->jk', big_phi, self.psi)
            errors = targets - cur[action,:]
            running_err += np.mean(errors)
            #print(np.reshape(errors, (-1, 1)).shape, np.reshape(self.env.gaussian(state).flatten(), (1, -1)).shape, self.psi[:,action,:].shape)
            self.psi[:,action,:] += (self.learning_rate * np.reshape(errors, (-1, 1)) * np.reshape(self.env.gaussian(state).flatten(), (1, -1))).T
        return running_err / len(transitions)
    
    def get_psi(self, state):
        return self.psi[state]
    
    def get_w(self):
        return self.w
    
    def phi(self):
        return self.env.phi()
    
    def factored_rew(self, w_p):
        w_a = np.ones_like(w_p) - w_p
        return w_a

    def train_agent(self, s, a, s1):
        q1 = self.GPE(s1)
        next_action = np.argmax(q1)
        phi = self.phi()[a]
        transitions = [(s, a, phi, s1, next_action)]
        return self.update_psi(transitions)
    
    def get_action(self, state):        
        if random.random() <= self.epsilon:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(self.GPE(state))
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return a
    
    def GPE(self, state):
        big_phi = self.env.gaussian(state).flatten()
        q = np.einsum('i,ijk->jk', big_phi, self.psi) @ self.w
        #print(q.shape)
        return q

def save_frames_as_gif(frames, path='./', filename='collect-fixed-', ep=0):
    filename = filename + str(ep) + '.gif'
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    plt.close()

def main():
    register(
        id='multigrid-collect-more-v0',
        entry_point='gym_multigrid.envs:CollectGame3Obj2Agent'
    )
    env = gym.make('multigrid-collect-more-v0')
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = SFQLAgent(env, state_dim=100, action_dim=env.ac_dim, feats=env.num_ball_types, 
                      learning_rate=0.05, discount_factor=0.95)
    rewards = []
    for ep in tqdm(range(5000), desc="SFQL-training"):
        frames = []
        s, _ = env.reset()
        agent_pos = env.agents[0].pos
        obs = env.grid.width * agent_pos[0] + agent_pos[1]
        done = False
        ep_rew = 0
        running_loss = 0
        for t in range(100):
            #env.render(mode='human', highlight=True if env.partial_obs else False)
            #time.sleep(0.1)
            #frames.append(env.render())
            actions = []
            action = agent.get_action(obs)
            actions.append(action)
            action_p = env.action_space.sample()
            actions.append(action_p)
            s_next, rew, done, truncated, info = env.step(actions)
            agent_pos = env.agents[0].pos
            obs_next = env.grid.width * agent_pos[0] + agent_pos[1]
            ep_rew += np.sum(rew)
            running_loss += agent.train_agent(obs, action, obs_next)
            obs = obs_next
            s = s_next

            if done:
                break
        env.close()
        #save_frames_as_gif(frames, ep=ep)
        writer.add_scalar('training loss', running_loss/t, ep)
        writer.add_scalar('reward', ep_rew, ep)
        writer.add_scalar('ep_length', t, ep)
        writer.add_scalar('num_balls_collected', env.collected_balls, ep)
        writer.add_scalar('num_agent1_ball1', info['agent1ball1'], ep)
        writer.add_scalar('num_agent1_ball2', info['agent1ball2'], ep)
        writer.add_scalar('num_agent1_ball3', info['agent1ball3'], ep)
        writer.add_scalar('num_agent2_ball1', info['agent2ball1'], ep)
        writer.add_scalar('num_agent2_ball2', info['agent2ball2'], ep)
        writer.add_scalar('num_agent2_ball3', info['agent2ball3'], ep)
        rewards.append(ep_rew)
        print("ep: ", ep, "rew: ", ep_rew)

if __name__ == "__main__":
    main()