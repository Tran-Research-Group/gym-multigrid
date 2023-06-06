import gymnasium as gym
from gymnasium.envs.registration import register
from matplotlib import animation
import matplotlib.pyplot as plt
from gym_multigrid.agent import *
import torch
import numpy as np
from tqdm import tqdm
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class QLAgent:
    def __init__(self, env, state_dim, action_dim, learning_rate, discount_factor):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = np.random.uniform(-0.01, 0.01, size=(state_dim, action_dim))
        self.n_actions = action_dim
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.3
        self.env = env
        self.transitions = deque(maxlen=10000)

    def update_q(self):
        running_err = 0.0
        for (state, action, r, next_state, next_action) in self.transitions:
            #print(state.shape)
            big_phi = self.env.gaussian(next_state).flatten('F')
            q_temp = np.matmul(np.reshape(big_phi, (1, -1)), self.q).reshape(-1,)
            targets = r + self.discount_factor * q_temp[next_action]
            big_phi = self.env.gaussian(state).flatten('F')
            cur = np.matmul(np.reshape(big_phi, (1, -1)), self.q).reshape(-1,)
            errors = targets - cur[action]
            running_err += errors
            self.q[:,action] += self.learning_rate * errors * self.env.gaussian(state).flatten('F')
        return running_err / len(self.transitions)
    
    def get_q(self, state):
        return self.q[state]

    def remember(self, s, a, r, s1):
        q1 = self.q[s1]
        next_action = np.argmax(q1)
        self.transitions.append([s, a, r, s1, next_action])
        #return self.update_q(transitions)
    
    def get_action(self, state):        
        if random.random() <= self.epsilon:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(self.q[state])
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return a, self.epsilon

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
    agent = QLAgent(env, state_dim=100, action_dim=env.ac_dim, learning_rate=0.05, discount_factor=0.95)
    rewards = []
    episodes = 5000
    for ep in tqdm(range(episodes), desc="QL-training"):
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
            if ep == episodes - 1:
                frames.append(env.render())
            actions = []
            action, epsilon = agent.get_action(obs)
            actions.append(action)
            action_p = env.action_space.sample()
            actions.append(action_p)
            s_next, rew, done, truncated, info = env.step(actions)
            agent_pos = env.agents[0].pos
            obs_next = env.grid.width * agent_pos[0] + agent_pos[1]
            ep_rew += rew[0]
            agent.remember(obs, action, ep_rew, obs_next)
            obs = obs_next
            s = s_next

            if done:
                break
        running_loss = agent.update_q()
        #plt.imshow(np.reshape(env.gaussian(obs).flatten('F'), (-1, 1)))
        #plt.savefig("gauss.png")
        #print(str(env))
        #print(obs, env.agents[0].pos)
        #env.render(mode='human', highlight=True if env.partial_obs else False)
        env.close()
        writer.add_scalar('training loss', running_loss, ep)
        writer.add_scalar('reward', ep_rew, ep)
        writer.add_scalar('ep_length', t, ep)
        writer.add_scalar('epsilon', epsilon, ep)
        writer.add_scalar('num_balls_collected', env.collected_balls, ep)
        writer.add_scalar('num_agent1_ball1', info['agent1ball1'], ep)
        writer.add_scalar('num_agent1_ball2', info['agent1ball2'], ep)
        writer.add_scalar('num_agent1_ball3', info['agent1ball3'], ep)
        writer.add_scalar('num_agent2_ball1', info['agent2ball1'], ep)
        writer.add_scalar('num_agent2_ball2', info['agent2ball2'], ep)
        writer.add_scalar('num_agent2_ball3', info['agent2ball3'], ep)
        rewards.append(ep_rew)
        print("ep: ", ep, "rew: ", ep_rew)
    save_frames_as_gif(frames, ep='ql-random')

if __name__ == "__main__":
    main()