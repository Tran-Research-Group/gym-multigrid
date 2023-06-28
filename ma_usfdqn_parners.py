from typing import Literal
import gymnasium as gym
from gymnasium.envs.registration import register
from gym_multigrid.utils.misc import set_seed, save_frames_as_gif
import torch
import torch.nn as nn
import torch.optim as optim
from torch.types import Number
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter


class NNet(nn.Module):
    def __init__(self, input_size: int, action_dim: int, feature_dim: int):
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

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.input_size).float()
        output: Tensor = self.model(x)
        return output.view([output.shape[0], self.action_dim, self.feature_dim])


class MaUsfDqnAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feat_dim: int,
        w: NDArray,
        lr: float,
        gamma: float,
        epsilon: float,
        nz: int,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1 psi network per feature dimension. probably a cleaner way to set this up
        self.input_size: int = state_dim + w.shape[0]
        self.psi1 = NNet(self.input_size, action_dim, 1).to(self.device)
        self.psi2 = NNet(self.input_size, action_dim, 1).to(self.device)
        self.psi3 = NNet(self.input_size, action_dim, 1).to(self.device)
        self.optim1 = optim.Adam(self.psi1.parameters(), lr=lr)
        self.optim2 = optim.Adam(self.psi2.parameters(), lr=lr)
        self.optim3 = optim.Adam(self.psi3.parameters(), lr=lr)

        joint_action_dim: int = action_dim * action_dim

        # the output of the joint psi network is a action_dim x action_dim vector of q values
        # e.g. if action_dim = 4, then the output is 16 q values for the joint action pairs.
        # the first 4 q values are the q values for the first action of the learner.
        # the second 4 q values are the q values for the second action of the learner, and so on.
        self.psi_joint1 = NNet(self.input_size, joint_action_dim, 1).to(self.device)
        self.psi_joint2 = NNet(self.input_size, joint_action_dim, 1).to(self.device)
        self.psi_joint3 = NNet(self.input_size, joint_action_dim, 1).to(self.device)
        self.optim_joint1 = optim.Adam(self.psi_joint1.parameters(), lr=lr)
        self.optim_joint2 = optim.Adam(self.psi_joint2.parameters(), lr=lr)
        self.optim_joint3 = optim.Adam(self.psi_joint3.parameters(), lr=lr)

        self.w = torch.from_numpy(w)
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.nz = nz
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.feat_dim: int = feat_dim
        self.batch_size: int = 20
        self.buffer: NDArray = np.empty(self.batch_size, dtype=object)
        self.buffer_size: int = 0

    def select_action(self, state) -> Number:
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

    def gpi_partner_action(
        self, state, zs: list[NDArray], learner_action: Number
    ) -> Number:
        if np.random.rand() < self.epsilon:
            action: Number = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                Q: list[Tensor] = []
                for z_i in zs:
                    z_i = torch.from_numpy(z_i).to(self.device)
                    nn_input = torch.cat([state, z_i])
                    q_values: Tensor = (
                        self.psi_joint1(nn_input).flatten()[
                            learner_action * 4 : learner_action * 4 + 4
                        ]
                        * self.w[0]
                        + self.psi_joint2(nn_input).flatten()[
                            learner_action * 4 : learner_action * 4 + 4
                        ]
                        * self.w[1]
                        + self.psi_joint3(nn_input).flatten()[
                            learner_action * 4 : learner_action * 4 + 4
                        ]
                        * self.w[2]
                    )
                    Q.append(q_values)

                Q_tensor: Tensor = torch.stack(Q)
                action: Number = torch.argmax(Q_tensor).item() % self.action_dim

        return action

    def phi(self, state, next_state):
        # how many of each type of object was picked up between s and s'
        ball1 = np.sum(state[:, :, 0]) - np.sum(next_state[:, :, 0])
        ball2 = np.sum(state[:, :, 1]) - np.sum(next_state[:, :, 1])
        ball3 = np.sum(state[:, :, 2]) - np.sum(next_state[:, :, 2])
        return np.array([ball1, ball2, ball3])

    def update(
        self, state, joint_action, next_state, z_i: NDArray
    ) -> NDArray | Literal[0]:
        phi = self.phi(state, next_state)
        state = torch.from_numpy(state).to(self.device)
        learner_action = torch.from_numpy(np.array([joint_action[0]])).to(self.device)
        partner_action = torch.from_numpy(np.array([joint_action[1]])).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        phi = torch.from_numpy(phi).to(self.device)
        # add this transition to buffer
        self.buffer[self.buffer_size] = (
            state,
            learner_action,
            partner_action,
            phi,
            next_state,
        )
        self.buffer_size += 1
        # once buffer is big enough do batch update
        if self.buffer_size == self.batch_size:
            indices = np.random.randint(
                low=0, high=self.buffer_size, size=(self.batch_size,)
            )
            states, learner_actions, partner_actions, phis, next_states = zip(
                *self.buffer[indices]
            )

            states = [
                torch.concat([s.cpu().flatten(), torch.from_numpy(z_i)]) for s in states
            ]
            learner_actions = [a.cpu() for a in learner_actions]
            partner_actions = [a.cpu() for a in partner_actions]
            phis = [p.cpu() for p in phis]
            next_states = [
                torch.concat([n.cpu().flatten(), torch.from_numpy(z_i)])
                for n in next_states
            ]

            states = torch.from_numpy(
                np.vstack(states).reshape((self.batch_size, self.input_size))
            ).to(self.device)

            learner_actions = torch.from_numpy(np.vstack(learner_actions)).to(
                self.device
            )
            partner_actions = torch.from_numpy(np.vstack(partner_actions)).to(
                self.device
            )
            phis = torch.from_numpy(
                np.vstack(phis).reshape((self.batch_size, self.feat_dim))
            ).to(self.device)
            next_states = torch.from_numpy(
                np.vstack(next_states).reshape((self.batch_size, self.input_size))
            ).to(self.device)
            # compute current values
            cur_psi1 = (
                self.psi1(states)
                .squeeze(-1)
                .gather(1, learner_actions.unsqueeze(0).view(-1, 1))
                .flatten()
            ).to(self.device)
            cur_psi2 = (
                self.psi2(states)
                .squeeze(-1)
                .gather(1, learner_actions.unsqueeze(0).view(-1, 1))
                .flatten()
            )
            cur_psi3 = (
                self.psi3(states)
                .squeeze(-1)
                .gather(1, learner_actions.unsqueeze(0).view(-1, 1))
                .flatten()
            )

            joint_actions: Tensor = learner_actions.unsqueeze(0).view(
                -1, 1
            ) * 4 + partner_actions.unsqueeze(0).view(-1, 1)

            cur_psi_joint1 = (
                self.psi_joint1(states).squeeze(-1).gather(1, joint_actions).flatten()
            ).to(self.device)
            cur_psi_joint2 = (
                self.psi_joint2(states).squeeze(-1).gather(1, joint_actions).flatten()
            ).to(self.device)
            cur_psi_joint3 = (
                self.psi_joint3(states).squeeze(-1).gather(1, joint_actions).flatten()
            ).to(self.device)

            # compute target values
            with torch.no_grad():
                next_psi1 = torch.max(self.psi1(next_states), dim=1).values.squeeze(1)
                next_psi2 = torch.max(self.psi2(next_states), dim=1).values.squeeze(1)
                next_psi3 = torch.max(self.psi3(next_states), dim=1).values.squeeze(1)
                next_psi_joint1 = torch.max(
                    self.psi_joint1(next_states), dim=1
                ).values.squeeze(1)
                next_psi_joint2 = torch.max(
                    self.psi_joint2(next_states), dim=1
                ).values.squeeze(1)
                next_psi_joint3 = torch.max(
                    self.psi_joint3(next_states), dim=1
                ).values.squeeze(1)

            target1 = phis[:, 0] + self.gamma * next_psi_joint1
            target2 = phis[:, 1] + self.gamma * next_psi_joint2
            target3 = phis[:, 2] + self.gamma * next_psi_joint3
            # reset buffer
            self.buffer = np.empty(self.batch_size, dtype=object)
            self.buffer_size = 0
            # compute losses
            loss1: Tensor = nn.MSELoss(reduction="sum")(cur_psi1, target1) / 2
            loss2: Tensor = nn.MSELoss(reduction="sum")(cur_psi2, target2) / 2
            loss3: Tensor = nn.MSELoss(reduction="sum")(cur_psi3, target3) / 2
            loss_joint1: Tensor = (
                nn.MSELoss(reduction="sum")(cur_psi_joint1, target1) / 2
            )
            loss_joint2: Tensor = (
                nn.MSELoss(reduction="sum")(cur_psi_joint2, target2) / 2
            )
            loss_joint3: Tensor = (
                nn.MSELoss(reduction="sum")(cur_psi_joint3, target3) / 2
            )

            self.optim1.zero_grad()
            self.optim2.zero_grad()
            self.optim3.zero_grad()
            self.optim_joint1.zero_grad()
            self.optim_joint2.zero_grad()
            self.optim_joint3.zero_grad()

            loss1.backward()
            loss2.backward()
            loss3.backward()
            loss_joint1.backward()
            loss_joint2.backward()
            loss_joint3.backward()

            self.optim1.step()
            self.optim2.step()
            self.optim3.step()
            self.optim_joint1.step()
            self.optim_joint2.step()
            self.optim_joint3.step()

            return np.array([loss1.item(), loss2.item(), loss3.item()])
        return 0

    def distribution(self, type: Literal["uniform"] = "uniform") -> NDArray:
        """
        Returns a sample from a distribution over the task given task vector w.

        Parameters
        ----------
        type: Literal['uniform'] = 'uniform'
            Type of distribution to sample from

        Returns
        -------
        z: NDArray
            Task from the distribution over tasks. The shape of z is (w.shape[0],).
        """

        match type:
            case "uniform":
                z = np.random.uniform(low=-1, high=1, size=(self.w.shape[0],))
            case _:
                raise ValueError(f"Unknown distribution type {type}")

        return z


class SfLearnerAgent:
    def __init__(self, dirname: str, filename: str, type: str) -> None:
        self.psi1: NNet = torch.load(dirname + filename + "_psi1.torch")
        self.psi1.eval()
        self.psi2: NNet = torch.load(dirname + filename + "_psi2.torch")
        self.psi2.eval()
        self.psi3: NNet = torch.load(dirname + filename + "_psi3.torch")
        self.psi3.eval()
        if filename == "red":
            self.w = np.array([1.0, 0.0, 0.0])
        elif filename == "orange":
            self.w = np.array([0.0, 1.0, 0.0])
        elif filename == "yellow":
            self.w = np.array([0.0, 0.0, 1.0])
        elif type == "twohot-red":
            self.w = np.array([0.0, 1.0, 1.0])
        elif type == "twohot-orange":
            self.w = np.array([1.0, 0.0, 1.0])
        elif type == "twohot-yellow":
            self.w = np.array([1.0, 1.0, 0.0])
        else:
            assert False, "that partner policy does not exist"

    def get_action(self, state):
        state = torch.from_numpy(state)
        q_values = (
            self.psi1(state) * self.w[0]
            + self.psi2(state) * self.w[1]
            + self.psi3(state) * self.w[2]
        )
        return torch.argmax(q_values).item()


def main():
    colors = ["red"]
    for i in range(len(colors)):
        seed = 42
        set_seed(seed=seed)
        register(
            id="multigrid-collect-rooms-v0",
            entry_point="gym_multigrid.envs:CollectGameRooms",
        )
        env = gym.make("multigrid-collect-rooms-v0")
        w = np.array([1.0, 1.0, 1.0])  # red, orange, yellow
        lr = 3e-5
        agent = MaUsfDqnAgent(
            state_dim=env.grid.width * env.grid.height * 5,
            action_dim=env.ac_dim,
            feat_dim=env.phi_dim(),
            w=w,
            lr=lr,
            gamma=0.9,
            epsilon=0.1,
            nz=5,
        )
        learner = SfLearnerAgent(
            dirname="sf-learner-twohot-models/",
            filename=f"{colors[i]}partner",
            type=f"twohot-{colors[i]}",
        )
        writer = SummaryWriter(comment=f"lr_{lr}_sflearner_{colors[i]}partner")
        frames = []
        episodes = 100_000
        for ep in tqdm(range(episodes), desc="SF-learner-training"):
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
                zs: list[NDArray] = [agent.distribution() for _ in range(agent.nz)]
                # get agent action
                learner_action = learner.get_action(obs.flatten())
                partner_action = agent.gpi_partner_action(
                    obs.flatten(), zs, learner_action
                )
                joint_action: list[Number] = [learner_action, partner_action]

                # step env with selected action
                obs_next, rew, done, truncated, info = env.step(joint_action)

                agent_pos = env.agents[0].pos
                idx = env.grid.width * agent_pos[0] + agent_pos[1]
                obs_next = env.toroid(idx)

                # shaped reward
                s_rew += np.dot(w, agent.phi(obs, obs_next))
                # standard env reward
                ep_rew += np.sum(rew)
                ep_rew_a += rew[0]
                ep_rew_p += rew[1]

                loss_sum = 0

                for z_i in zs:
                    losses = agent.update(obs, joint_action, obs_next, z_i)
                    loss_sum += np.sum(losses)

                running_loss += loss_sum / agent.nz

                # save gif of last episode for fun
                if ep == episodes - 1:
                    frames.append(env.render())
                if done:
                    break
                obs = obs_next

            # tensorboard logging
            writer.add_scalar("training loss", running_loss / t, ep)
            writer.add_scalar("total_reward", ep_rew, ep)
            writer.add_scalar("learner_reward", ep_rew_a, ep)
            writer.add_scalar("partner_reward", ep_rew_p, ep)
            writer.add_scalar("total_shaped_reward", s_rew, ep)
            writer.add_scalar(
                "learner_shaped_reward",
                info["agent1ball3"] + info["agent1ball2"] + info["agent1ball1"],
                ep,
            )
            writer.add_scalar("partner_shaped_reward", info["agent2ball2"], ep)
            writer.add_scalar("ep_length", t, ep)
            writer.add_scalar("num_balls_collected", env.collected_balls, ep)
            writer.add_scalar("num_agent1_ball1", info["agent1ball1"], ep)
            writer.add_scalar("num_agent1_ball2", info["agent1ball2"], ep)
            writer.add_scalar("num_agent1_ball3", info["agent1ball3"], ep)
            writer.add_scalar("num_agent2_ball1", info["agent2ball1"], ep)
            writer.add_scalar("num_agent2_ball2", info["agent2ball2"], ep)
            writer.add_scalar("num_agent2_ball3", info["agent2ball3"], ep)

        writer.close()
        learner_type = f"{colors[i]}partner"
        save_frames_as_gif(frames, path="./plots/", ep=learner_type)
        torch.save(agent.psi1, f"models/mausf/{learner_type}_psi1.torch")
        torch.save(agent.psi2, f"models/mausf/{learner_type}_psi2.torch")
        torch.save(agent.psi3, f"models/mausf/{learner_type}_psi3.torch")


if __name__ == "__main__":
    main()
