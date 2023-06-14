import gymnasium as gym
from gymnasium.envs.registration import register
from gym_multigrid.utils import set_seed, save_frames_as_gif
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
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, action_dim * feature_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_size).float()
        output = self.model(x)
        return output.view([output.shape[0], self.action_dim, self.feature_dim])


class IndSFDQNAgent:
    def __init__(self, state_dim, action_dim, feat_dim, w, lr, gamma, epsilon) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1 psi network per feature dimension. probably a cleaner way to set this up
        self.psi1 = NNet(state_dim, action_dim, 1).to(self.device)
        self.psi2 = NNet(state_dim, action_dim, 1).to(self.device)
        self.psi3 = NNet(state_dim, action_dim, 1).to(self.device)
        self.optim1 = optim.Adam(self.psi1.parameters(), lr=lr)
        self.optim2 = optim.Adam(self.psi2.parameters(), lr=lr)
        self.optim3 = optim.Adam(self.psi3.parameters(), lr=lr)

        self.w = torch.from_numpy(w)
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.batch_size = 20
        self.buffer = np.empty(self.batch_size, dtype=object)
        self.buffer_size = 0

    def select_action(self, state):
        # epsilon greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                q_values = (
                    self.psi1(state) * self.w[0]
                    + self.psi2(state) * self.w[1]
                    + self.psi3(state) * self.w[2]
                )
                action = torch.argmax(q_values).item()
        return action

    def phi(self, state, next_state):
        # how many of each type of object was picked up between s and s'
        ball1 = np.sum(state[:, :, 0]) - np.sum(next_state[:, :, 0])
        ball2 = np.sum(state[:, :, 1]) - np.sum(next_state[:, :, 1])
        ball3 = np.sum(state[:, :, 2]) - np.sum(next_state[:, :, 2])
        return np.array([ball1, ball2, ball3])

    def update(self, state, action, next_state):
        phi = self.phi(state, next_state)
        state = torch.from_numpy(state).to(self.device)
        action = torch.from_numpy(np.array([action])).to(self.device).view(-1)
        next_state = torch.from_numpy(next_state).to(self.device)
        phi = torch.from_numpy(phi).to(self.device)
        # add this transition to buffer
        self.buffer[self.buffer_size] = (state, action, phi, next_state)
        self.buffer_size += 1
        # once buffer is big enough do batch update
        if self.buffer_size == self.batch_size:
            indices = np.random.randint(
                low=0, high=self.buffer_size, size=(self.batch_size,)
            )
            states, actions, phis, next_states = zip(*self.buffer[indices])
            states = torch.from_numpy(
                np.vstack(states).reshape((self.batch_size, self.state_dim))
            ).to(self.device)
            actions = torch.from_numpy(np.vstack(actions)).to(self.device)
            phis = torch.from_numpy(
                np.vstack(phis).reshape((self.batch_size, self.feat_dim))
            ).to(self.device)
            next_states = torch.from_numpy(
                np.vstack(next_states).reshape((self.batch_size, self.state_dim))
            ).to(self.device)
            # compute current values
            cur_psi1 = (
                self.psi1(states)
                .squeeze(-1)
                .gather(1, actions.unsqueeze(0).view(-1, 1))
                .flatten()
            ).to(self.device)
            cur_psi2 = (
                self.psi2(states)
                .squeeze(-1)
                .gather(1, actions.unsqueeze(0).view(-1, 1))
                .flatten()
            ).to(self.device)
            cur_psi3 = (
                self.psi3(states)
                .squeeze(-1)
                .gather(1, actions.unsqueeze(0).view(-1, 1))
                .flatten()
            ).to(self.device)
            # compute target values
            with torch.no_grad():
                next_psi1 = torch.max(self.psi1(next_states), dim=1).values.squeeze(1)
                next_psi2 = torch.max(self.psi2(next_states), dim=1).values.squeeze(1)
                next_psi3 = torch.max(self.psi3(next_states), dim=1).values.squeeze(1)
            target1 = phis[:, 0] + self.gamma * next_psi1
            target2 = phis[:, 1] + self.gamma * next_psi2
            target3 = phis[:, 2] + self.gamma * next_psi3
            # reset buffer
            self.buffer = np.empty(self.batch_size, dtype=object)
            self.buffer_size = 0
            # compute losses
            loss1 = nn.MSELoss(reduction="sum")(cur_psi1, target1) / 2
            loss2 = nn.MSELoss(reduction="sum")(cur_psi2, target2) / 2
            loss3 = nn.MSELoss(reduction="sum")(cur_psi3, target3) / 2

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
        return 0

def load_partner(filename):
    partner = torch.load(filename)
    return partner

def main():
    seed = 42
    set_seed(seed=seed)
    register(
        id="multigrid-collect-more-v0",
        entry_point="gym_multigrid.envs:CollectGame3Obj2Agent",
    )
    env = gym.make("multigrid-collect-more-v0")
    w = np.array([-1.0, 1.0, 0.0])  # red, orange, yellow
    agent = IndSFDQNAgent(
        state_dim=env.grid.width * env.grid.height * 4,
        action_dim=env.ac_dim,
        feat_dim=env.phi_dim(),
        w=w,
        lr=0.00003,
        gamma=0.9,
        epsilon=0.1,
    )
    partner = load_partner(filename='')
    partner.eval()
    frames = []
    episodes = 50000
    for ep in tqdm(range(episodes), desc="Ind-SFDQN-training"):
        obs, _ = env.reset(seed=seed)
        agent_pos = env.agents[0].pos
        idx = env.grid.width * agent_pos[0] + agent_pos[1]
        obs = env.toroid(idx)
        # use this code to compare toroidal obs with standard env obs
        # plt.imshow(20*obs[:,:,0] + 50*obs[:,:,1] + 100*obs[:,:,2] + 70*obs[:,:,3])
        # plt.savefig('toroid.png')
        # print(str(env))
        done = False
        ep_rew = 0
        rew_a = 0
        running_loss = 0
        for t in range(100):
            # use this code for live rendering
            # env.render(mode='human', highlight=True if env.partial_obs else False)
            # time.sleep(0.1)
            # frames.append(env.render(mode="rgb_array"))

            # get agent action
            actions = []
            action = agent.select_action(obs.flatten())
            actions.append(action)

            # use this for random partner
            # action_p = env.action_space.sample()
            # actions.append(action_p)

            action_p = torch.argmax(partner(torch.from_numpy(obs.flatten()).to(agent.device))).item()
            actions.append(action_p)

            # step env with selected action
            obs_next, rew, done, truncated, info = env.step(actions)

            agent_pos = env.agents[0].pos
            idx = env.grid.width * agent_pos[0] + agent_pos[1]
            obs_next = env.toroid(idx)

            # shaped reward
            rew_a += np.dot(w, agent.phi(obs, obs_next))
            # standard env reward
            ep_rew += np.sum(rew)

            loss = agent.update(obs, action, obs_next)
            loss = np.sum(loss)
            running_loss += loss

            # save gif of last episode for fun
            if ep == episodes - 1:
                frames.append(env.render())
            if done:
                break
            obs = obs_next

        # tensorboard logging
        writer.add_scalar("training loss", running_loss / t, ep)
        writer.add_scalar("total_reward", ep_rew, ep)
        writer.add_scalar("agent_shaped_reward", rew_a, ep)
        writer.add_scalar("ep_length", t, ep)
        writer.add_scalar("num_balls_collected", env.collected_balls, ep)
        writer.add_scalar("num_agent1_ball1", info["agent1ball1"], ep)
        writer.add_scalar("num_agent1_ball2", info["agent1ball2"], ep)
        writer.add_scalar("num_agent1_ball3", info["agent1ball3"], ep)
        writer.add_scalar('num_agent2_ball1', info['agent2ball1'], ep)
        writer.add_scalar('num_agent2_ball2', info['agent2ball2'], ep)
        writer.add_scalar('num_agent2_ball3', info['agent2ball3'], ep)

    writer.close()
    save_frames_as_gif(frames, ep="ind-sfdqn-random")
    torch.save(agent.psi1, "psi1.torch")
    torch.save(agent.psi2, "psi2.torch")
    torch.save(agent.psi3, "psi3.torch")


if __name__ == "__main__":
    main()