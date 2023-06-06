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
    def __init__(self, input_size, action_dim):
        super(NNet, self).__init__()
        self.input_size = input_size
        self.action_dim = action_dim
        layers = []
        layers.append(nn.Linear(input_size, 64))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(64, 64))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(64, action_dim))

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

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate, discount_factor, epsilon, buffer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = NNet(state_dim, action_dim).to(self.device)
        self.target_network = NNet(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = epsilon
        self.action_dim = action_dim
        self.buffer = buffer
    
    def remember(self, state, action, reward, next_state):
        self.buffer.append(state, action, reward, next_state)

    def update_q_network(self, state, action, reward, next_state):
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        action = torch.LongTensor([action]).to(self.device).view(-1)
        reward = torch.Tensor([reward]).to(self.device).view(-1)
        q_values = torch.matmul(self.q_network(state), self.w)
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
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                q_values = torch.matmul(self.q_network(state), self.w)
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
    agent = DQNAgent(env.grid.width * env.grid.height, env.ac_dim, 0.005, 0.99, 0.2, ReplayBuffer())
    frames = []
    episodes = 5000
    for ep in tqdm(range(episodes), desc="DQN-training"):
        obs, _ = env.reset()
        obs = env.gaussian()
        done = False
        ep_rew = 0
        running_loss = 0
        for t in range(100):
            #env.render(mode='human', highlight=True if env.partial_obs else False)
            #time.sleep(0.1)
            #frames.append(env.render(mode="rgb_array"))
            action = agent.select_action(obs)
            obs_next, rew, done, truncated, info = env.step(action)
            obs_next = env.gaussian()
            ep_rew += rew
            agent.remember(obs, action, rew, obs_next)
            if agent.buffer.size >= 500:
                states, actions, rews, next_states = agent.buffer.replay()
                loss = agent.update_q_network(states, actions, rews, next_states)
                running_loss += loss
            if ep == episodes - 1:
                frames.append(env.render())
            if done:
                break
            obs = obs_next
        if ep % 100 == 0:
            agent.update_target_network()
        #save_frames_as_gif(frames, 'sfql-random')
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
    # for name, weight in model.named_parameters():
    # tb.add_histogram(name,weight, epoch)
    # tb.add_histogram(f'{name}.grad',weight.grad, epoch)
    writer.close()
    save_frames_as_gif(frames, ep='dqn-random')
    torch.save(agent.q_network, 'dqn.torch')

'''
def hyperparameters():
comment = f' batch_size = {batch_size} lr = {lr} shuffle = {shuffle}'
tb = SummaryWriter(comment=comment)
tb.add_hparams(
            {"lr": lr, "bsize": batch_size, "shuffle":shuffle},
            {
                "accuracy": total_correct/ len(train_set),
                "loss": total_loss,
            },
        )
'''

if __name__ == "__main__":
    main()