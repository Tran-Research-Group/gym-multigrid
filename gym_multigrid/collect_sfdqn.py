import gymnasium as gym
from gymnasium.envs.registration import register
from matplotlib import animation
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class NNet(nn.Module):
    def __init__(self, input_size, action_dim, feature_dim):
        super(NNet, self).__init__()
        self.input_size = input_size
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        layers = []
        layers.append(nn.Linear(input_size, 64))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(64, 128))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(128, action_dim * feature_dim))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        #print('input: ', x.shape)
        x = x.view(-1, self.input_size).float()
        #print('reshaped input: ', x.shape)
        output = self.model(x)
        #print('output: ', output.shape)
        return output.view([output.shape[0], self.action_dim, self.feature_dim])

class ReplayBuffer:
    def __init__(self, n_samples=10000, n_batch=32):
        self.n_samples = n_samples
        self.n_batch = n_batch
        self.buffer = np.empty(self.n_samples, dtype=object)
        self.index = 0
        self.size = 0
    
    def reset(self):
        self.buffer = np.empty(self.n_samples, dtype=object)
        self.index = 0
        self.size = 0
    
    def replay(self):
        indices = np.random.randint(low=0, high=self.size, size=(self.n_batch,))
        states, actions, rewards, next_states = zip(*self.buffer[indices])
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.vstack(rewards)
        next_states = np.vstack(next_states)
        return states, actions, rewards, next_states
    
    def append(self, state, action, reward, next_state):
        self.buffer[self.index] = (state, action, reward, next_state)
        self.size = min(self.size + 1, self.n_samples)
        self.index = (self.index + 1) % self.n_samples

class IndSFDQNAgent:
    def __init__(self, state_dim, action_dim, feat_dim, w, lr, gamma, epsilon) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.psi1 = NNet(state_dim, action_dim, 1).to(self.device)
        self.psi2 = NNet(state_dim, action_dim, 1).to(self.device)
        self.psi3 = NNet(state_dim, action_dim, 1).to(self.device)
        self.optim1 = optim.Adam(self.psi1.parameters(), lr=lr)
        self.optim2 = optim.Adam(self.psi2.parameters(), lr=lr)
        self.optim3 = optim.Adam(self.psi3.parameters(), lr=lr)

        self.w = torch.from_numpy(w)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.feat_dim = feat_dim
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                q_values = self.psi1(state) * self.w[0] + self.psi2(state) * self.w[1] + self.psi3(state) * self.w[2]
                action = torch.argmax(q_values).item()
        return action
    
    def phi(self, state, next_state):
        ball1 = np.sum(state[:,:,1]) - np.sum(next_state[:,:,1])
        ball2 = np.sum(state[:,:,2]) - np.sum(next_state[:,:,2])
        ball3 = np.sum(state[:,:,3]) - np.sum(next_state[:,:,3])
        return np.array([ball1, ball2, ball3])

    def update(self, state, next_state):
        phi = self.phi(state, next_state)
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        cur_psi1 = torch.max(self.psi1(state))
        cur_psi2 = torch.max(self.psi2(state))
        cur_psi3 = torch.max(self.psi3(state))

        with torch.no_grad():
            next_psi1 = torch.max(self.psi1(next_state))
            next_psi2 = torch.max(self.psi2(next_state))
            next_psi3 = torch.max(self.psi3(next_state))
        target1 = phi[0] + self.gamma * next_psi1
        target2 = phi[1] + self.gamma * next_psi2
        target3 = phi[2] + self.gamma * next_psi3

        loss1 = nn.MSELoss(reduction='sum')(cur_psi1, target1)
        loss2 = nn.MSELoss(reduction='sum')(cur_psi2, target2)
        loss3 = nn.MSELoss(reduction='sum')(cur_psi3, target3)

        self.optim1.zero_grad()
        self.optim2.zero_grad()
        self.optim3.zero_grad()

        loss1.backward()
        loss2.backward()
        loss3.backward()

        self.optim1.step()
        self.optim2.step()
        self.optim3.step()

        return np.array([loss1.item(), loss2.item(), loss3.item()])

class SFDQNAgent:
    def __init__(self, state_dim, action_dim, feature_dim, w, learning_rate, discount_factor, epsilon, buffer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.psi_network = NNet(state_dim, action_dim, feature_dim).to(self.device)
        self.target_network = NNet(state_dim, action_dim, feature_dim).to(self.device)
        self.target_network.load_state_dict(self.psi_network.state_dict())
        self.optimizer = optim.Adam(self.psi_network.parameters(), lr=learning_rate)

        self.w = torch.from_numpy(w)
        self.discount_factor = discount_factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = epsilon
        self.action_dim = action_dim
        self.buffer = buffer
    
    def remember(self, state, action, reward, next_state):
        self.buffer.append(state, action, reward, next_state)

    def update_network(self, state, action, reward, next_state):
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        action = torch.Tensor([action]).to(self.device).view(-1)
        reward = torch.Tensor([reward]).to(self.device).view(-1)
        q_values = torch.matmul(self.psi_network(state), self.w)
        with torch.no_grad():
            next_q_values = torch.matmul(self.target_network(next_state), self.w)
            #print('next_q_values shape: ', next_q_values.shape)
            #print('reward shape: ', reward.shape)
            max_next_q_values = torch.max(next_q_values, dim=1)[0].unsqueeze(0).view(-1)
            #print('max_next_q_values: ', max_next_q_values.shape)
        target_q_values = reward + self.discount_factor * max_next_q_values
        #print('q_values shape: ', q_values.squeeze(-1).shape, 'action shape: ', action.unsqueeze(0).view(-1, 1).shape)
        q_values = q_values.squeeze(-1).gather(1, action.unsqueeze(0).view(-1, 1)).flatten()
        #print(target_q_values.shape, q_values.shape)
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.psi_network.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                q_values = torch.matmul(self.psi_network(state), self.w)
                action = torch.argmax(q_values).item()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return action

def save_frames_as_gif(frames, path='./', filename='collect-', ep=0):
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
    w = np.array([-2., 2., 0.])
    agent = IndSFDQNAgent(env.grid.width * env.grid.height * 4, env.ac_dim, env.phi_dim(), w, 0.001, 0.9, 0.2)
    frames = []
    episodes = 10000
    for ep in tqdm(range(episodes), desc="Ind-SFDQN-training"):
        obs, _ = env.reset()
        agent_pos = env.agents[0].pos
        idx = env.grid.width * agent_pos[0] + agent_pos[1]
        obs = env.toroid(idx)
        #plt.imshow(20*obs[:,:,0] + 50*obs[:,:,1] + 100*obs[:,:,2] + 70*obs[:,:,3])
        #plt.savefig('toroid.png')
        #print(str(env))
        done = False
        ep_rew = 0
        rew_a = 0
        running_loss = 0
        for t in range(50):
            #env.render(mode='human', highlight=True if env.partial_obs else False)
            #time.sleep(0.1)
            #frames.append(env.render(mode="rgb_array"))
            actions = []
            action = agent.select_action(obs.flatten())
            actions.append(action)
            action_p = env.action_space.sample()
            actions.append(action_p)

            obs_next, rew, done, truncated, info = env.step(actions)
            agent_pos = env.agents[0].pos
            idx = env.grid.width * agent_pos[0] + agent_pos[1]
            obs_next = env.toroid(idx)
            rew_a += np.dot(w, agent.phi(obs, obs_next))
            ep_rew += rew

            loss = agent.update(obs, obs_next)
            running_loss += np.sum(loss)
            if ep == episodes - 1:
                frames.append(env.render())
            if done:
                break
            obs = obs_next

        writer.add_scalar('training loss', running_loss/t, ep)
        writer.add_scalar('reward', rew_a, ep)
        writer.add_scalar('ep_length', t, ep)
        writer.add_scalar('num_balls_collected', env.collected_balls, ep)
        writer.add_scalar('num_agent1_ball1', info['agent1ball1'], ep)
        writer.add_scalar('num_agent1_ball2', info['agent1ball2'], ep)
        writer.add_scalar('num_agent1_ball3', info['agent1ball3'], ep)
        writer.add_scalar('num_agent2_ball1', info['agent2ball1'], ep)
        writer.add_scalar('num_agent2_ball2', info['agent2ball2'], ep)
        writer.add_scalar('num_agent2_ball3', info['agent2ball3'], ep)
    # for name, weight in model.named_parameters():
    # tb.add_histogram(name,weight, epoch)
    # tb.add_histogram(f'{name}.grad',weight.grad, epoch)
    writer.close()
    save_frames_as_gif(frames, ep='ind-sfdqn-random')
    #torch.save(agent.q_network, 'ind-sfdqn.torch')

if __name__ == "__main__":
    main()