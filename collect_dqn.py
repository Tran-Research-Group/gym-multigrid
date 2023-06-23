import gymnasium as gym
from gymnasium.envs.registration import register
from gym_multigrid.utils.misc import set_seed, save_frames_as_gif
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class NNet(nn.Module):
    def __init__(self, input_size, action_dim, feature_dim):
        super(NNet, self).__init__()
        self.input_size = input_size
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        layers = []
        layers.append(nn.Linear(input_size, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, action_dim * feature_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size).float()
        output = self.model(x)
        return output.view([output.shape[0], self.action_dim, self.feature_dim])


class SimpleDQNAgent:
    def __init__(self, state_dim, action_dim, feat_dim, lr, gamma, epsilon) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = NNet(state_dim, action_dim, feat_dim).to(self.device)
        self.target_network = NNet(state_dim, action_dim, feat_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optim = optim.Adam(self.q_network.parameters(), lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 20
        self.buffer = np.empty(self.batch_size, dtype=object)
        self.buffer_size = 0

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        # epsilon greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                q_values = self.q_network(state)
                action = torch.argmax(q_values).item()
        return action

    def update(self, state, action, rew, next_state):
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        action = torch.from_numpy(np.array([action])).to(self.device).view(-1)
        rew = torch.from_numpy(np.array([rew])).to(self.device).view(-1)

        self.buffer[self.buffer_size] = (state, action, rew, next_state)
        self.buffer_size += 1

        if self.buffer_size == self.batch_size:
            # randomize batch
            indices = np.random.randint(
                low=0, high=self.buffer_size, size=(self.batch_size,)
            )
            states, actions, rewards, next_states = zip(*self.buffer[indices])
            states = torch.from_numpy(
                np.vstack(states).reshape((self.batch_size, self.state_dim))
            ).to(self.device)
            actions = torch.from_numpy(np.vstack(actions)).to(self.device)
            rewards = torch.from_numpy(np.vstack(rewards)).to(self.device)
            next_states = torch.from_numpy(
                np.vstack(next_states).reshape((self.batch_size, self.state_dim))
            ).to(self.device)
            # compute q values
            qvals = (
                self.q_network(states)
                .squeeze(-1)
                .gather(1, actions.unsqueeze(0).view(-1, 1))
                .flatten()
            )
            # compute target values
            with torch.no_grad():
                targets = (
                    rew
                    + self.gamma
                    * torch.max(self.target_network(next_states), dim=1).values
                )
            # clear buffer
            self.buffer = np.empty(self.batch_size, dtype=object)
            self.buffer_size = 0
            # compute loss
            loss = nn.MSELoss(reduction="sum")(qvals.view(-1, 1), targets.float()) / 2
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            return loss.item()
        return 0


def phi(state, next_state):
    # how many of each type of object was picked up between s and s'
    ball1 = np.sum(state[:, :, 0]) - np.sum(next_state[:, :, 0])
    ball2 = np.sum(state[:, :, 1]) - np.sum(next_state[:, :, 1])
    ball3 = np.sum(state[:, :, 2]) - np.sum(next_state[:, :, 2])
    return np.array([ball1, ball2, ball3])


def main():
    types = ["red-random", "orange-random", "yellow-random"]
    weights = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for idx, t in enumerate(types):
        train_str = "dqn-" + t
        w = weights[idx]
        seed = 42
        set_seed(seed=seed)
        writer = SummaryWriter(comment=train_str)
        register(
            id="multigrid-collect-more-v0",
            entry_point="gym_multigrid.envs:CollectGame3Obj2Agent",
        )
        env = gym.make("multigrid-collect-more-v0")
        agent = SimpleDQNAgent(
            state_dim=env.grid.width * env.grid.height * 4,
            action_dim=env.ac_dim,
            feat_dim=1,
            lr=0.00001,
            gamma=0.9,
            epsilon=0.1,
        )
        frames = []
        episodes = 75000
        for ep in tqdm(range(episodes), desc="Simple-DQN-training"):
            obs, _ = env.reset(seed=seed)
            agent_pos = env.agents[0].pos
            idx = env.grid.width * agent_pos[0] + agent_pos[1]
            obs = env.toroid(idx)
            done = False
            ep_rew = 0
            ep_rew_a = 0
            ep_rew_p = 0
            s_rew = 0
            running_loss = 0
            for t in range(100):
                # get agent action
                actions = []
                action = agent.select_action(obs.flatten())
                actions.append(action)
                action_p = env.action_space.sample()
                actions.append(action_p)
                # step env with selected action
                obs_next, rew, done, truncated, info = env.step(actions)
                agent_pos = env.agents[0].pos
                idx = env.grid.width * agent_pos[0] + agent_pos[1]
                obs_next = env.toroid(idx)
                ep_rew += np.sum(rew)
                s_rew += np.dot(w, phi(obs, obs_next))
                ep_rew_a += rew[0]
                ep_rew_p += rew[1]
                loss = agent.update(
                    obs, action, np.dot(w, phi(obs, obs_next)), obs_next
                )
                running_loss += loss
                if ep == episodes - 1:
                    frames.append(env.render())
                if done:
                    break
                obs = obs_next
            if ep % 100 == 0:
                agent.update_target_network()
            writer.add_scalar("training loss", running_loss / t, ep)
            writer.add_scalar("reward", ep_rew, ep)
            writer.add_scalar("ep_length", t, ep)
            writer.add_scalar("learner_reward", ep_rew_a, ep)
            writer.add_scalar("partner_reward", ep_rew_p, ep)
            writer.add_scalar("total_shaped_reward", s_rew, ep)
            obj_arr = np.array(
                [info["agent1ball1"], info["agent1ball2"], info["agent1ball3"]]
            )
            agent_shaped_rew = np.dot(w, obj_arr)
            writer.add_scalar("learner_shaped_reward", agent_shaped_rew, ep)
            writer.add_scalar("num_balls_collected", env.collected_balls, ep)
            writer.add_scalar("num_agent1_ball1", info["agent1ball1"], ep)
            writer.add_scalar("num_agent1_ball2", info["agent1ball2"], ep)
            writer.add_scalar("num_agent1_ball3", info["agent1ball3"], ep)
            writer.add_scalar("num_agent2_ball1", info["agent2ball1"], ep)
            writer.add_scalar("num_agent2_ball2", info["agent2ball2"], ep)
            writer.add_scalar("num_agent2_ball3", info["agent2ball3"], ep)

        writer.close()
        save_frames_as_gif(frames, ep=train_str)
        model_str = train_str + ".torch"
        torch.save(agent.q_network, model_str)


if __name__ == "__main__":
    main()
