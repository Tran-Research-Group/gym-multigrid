import json
import multiprocessing as mp
import random
from typing import Literal, TypeVar

import gymnasium as gym
from gymnasium.envs.registration import register
import torch
from torch import Tensor
from torch.types import Number
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from numpy.typing import NDArray

from gym_multigrid.utils import set_seed, save_frames_as_gif

ModuleT = TypeVar("ModuleT", bound=nn.Module)


class NNet(nn.Module):
    def __init__(self, input_size: int, action_dim: int, feature_dim: int) -> None:
        super(NNet, self).__init__()
        self.input_size = input_size
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        layers: list[ModuleT] = []
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


class IndUSFDQNAgent:
    def __init__(
        self, state_dim, action_dim, feat_dim, w: NDArray, lr, gamma, epsilon, nz: int
    ) -> None:
        self.device = torch.device("cuda")
        # 1 psi network per feature dimension. probably a cleaner way to set this up
        self.input_size: int = state_dim + w.shape[0]
        self.psi1 = NNet(self.input_size, action_dim, 1).to(self.device)
        self.psi2 = NNet(self.input_size, action_dim, 1).to(self.device)
        self.psi3 = NNet(self.input_size, action_dim, 1).to(self.device)
        self.optim1 = optim.Adam(self.psi1.parameters(), lr=lr)
        self.optim2 = optim.Adam(self.psi2.parameters(), lr=lr)
        self.optim3 = optim.Adam(self.psi3.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.nz = nz
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.batch_size = 20
        self.buffer = np.empty(self.batch_size, dtype=object)
        self.buffer_size = 0

    def gpi_action(self, state, zs: list[NDArray], w_np: NDArray) -> Number:
        if np.random.rand() < self.epsilon:
            action: Number = np.random.randint(self.action_dim)
        else:
            w: Tensor = torch.from_numpy(w_np).to(self.device)
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device)
                Q: list[Tensor] = []
                for z_i in zs:
                    z_i = torch.from_numpy(z_i).to(self.device)
                    nn_input = torch.cat([state, z_i])
                    q_values: Tensor = (
                        self.psi1(nn_input) * w[0]
                        + self.psi2(nn_input) * w[1]
                        + self.psi3(nn_input) * w[2]
                    )
                    Q.append(q_values.flatten())

                Q_tensor: Tensor = torch.stack(Q)
                action: Number = torch.argmax(Q_tensor).item() % self.action_dim

        return action

    def phi(self, state, next_state) -> NDArray:
        # how many of each type of object was picked up between s and s'
        ball1 = np.sum(state[:, :, 0]) - np.sum(next_state[:, :, 0])
        ball2 = np.sum(state[:, :, 1]) - np.sum(next_state[:, :, 1])
        ball3 = np.sum(state[:, :, 2]) - np.sum(next_state[:, :, 2])
        return np.array([ball1, ball2, ball3])

    def update(self, state, action, next_state, z_i: NDArray) -> NDArray | Literal[0]:
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

            states = [
                torch.concat([s.cpu().flatten(), torch.from_numpy(z_i)]) for s in states
            ]
            actions = [a.cpu() for a in actions]
            phis = [p.cpu() for p in phis]
            next_states = [
                torch.concat([n.cpu().flatten(), torch.from_numpy(z_i)])
                for n in next_states
            ]

            states = torch.from_numpy(
                np.vstack(states).reshape((self.batch_size, self.input_size))
            ).to(self.device)

            actions = torch.from_numpy(np.vstack(actions)).to(self.device)
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
                .gather(1, actions.unsqueeze(0).view(-1, 1))
                .flatten()
            ).to(self.device)
            cur_psi2 = (
                self.psi2(states)
                .squeeze(-1)
                .gather(1, actions.unsqueeze(0).view(-1, 1))
                .flatten()
            )
            cur_psi3 = (
                self.psi3(states)
                .squeeze(-1)
                .gather(1, actions.unsqueeze(0).view(-1, 1))
                .flatten()
            )
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
            loss1: Tensor = nn.MSELoss(reduction="sum")(cur_psi1, target1) / 2
            loss2: Tensor = nn.MSELoss(reduction="sum")(cur_psi2, target2) / 2
            loss3: Tensor = nn.MSELoss(reduction="sum")(cur_psi3, target3) / 2

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

    def distribution(
        self,
        w_np: NDArray,
        type: Literal["uniform"] = "uniform",
    ) -> NDArray:
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
                z = np.random.uniform(low=-1, high=1, size=(w_np.shape[0],))
            case _:
                raise ValueError(f"Unknown distribution type {type}")

        return z


def main():
    lrs: list[float] = [1e-5]
    alg: str = "usfdqn"
    num_replicates: int = 3

    tensor_board_dir: str = f"runs/{alg}/{alg}_"
    seed_log_path: str = f"logs/seed_{alg}_.json"
    model_dir: str = f"models/{alg}/"
    start_replicate: int = 1

    mp.set_start_method("spawn")

    with mp.Pool(processes=len(lrs)) as pool:
        pool.starmap(
            run_replicates,
            [
                (
                    num_replicates,
                    lr,
                    tensor_board_dir,
                    seed_log_path,
                    model_dir,
                    start_replicate,
                )
                for lr in lrs
            ],
        )


def run_replicates(
    num_replicates: int,
    lr: float,
    tensor_board_dir: str,
    seed_log_path: str,
    model_dir: str,
    start_replicate: int = 0,
    gpi_eval_freq_ratio: float = 0.01,
) -> None:
    seeds = np.random.randint(low=0, high=10000, size=(num_replicates,))

    with open(seed_log_path.replace(".json", f"lr_{lr}.json"), "w") as f:
        json.dump({"seeds": seeds.tolist()}, f)

    for i in range(start_replicate, num_replicates):
        path_suffix: str = f"lr_{lr}_rep_{i}"

        writer = SummaryWriter(tensor_board_dir + path_suffix)
        seed: int = seeds[i]
        set_seed(seed=seed)
        register(
            id="multigrid-collect-more-v0",
            entry_point="gym_multigrid.envs:CollectGame3Obj2Agent",
        )
        env = gym.make("multigrid-collect-more-v0")

        w_test: NDArray = np.array([-1.0, 1.0, 1.0])

        ws: list[NDArray] = [
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 1.0]),
        ]  # red, orange, yellow
        agent = IndUSFDQNAgent(
            state_dim=env.grid.width * env.grid.height * 4,
            action_dim=env.ac_dim,
            feat_dim=env.phi_dim(),
            w=w_test,
            lr=lr,
            gamma=0.9,
            epsilon=0.1,
            nz=5,
        )
        frames = []
        episodes = 50000
        gpi_eval_freq = int(episodes * gpi_eval_freq_ratio)

        for ep in tqdm(range(episodes), desc="Ind-USFDQN-training"):
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
            episode_length: int = 100
            for t in range(episode_length):
                # use this code for live rendering
                # env.render(mode='human', highlight=True if env.partial_obs else False)
                # time.sleep(0.1)
                # frames.append(env.render(mode="rgb_array"))

                w: NDArray = random.choice(ws)
                zs: list[NDArray] = [agent.distribution(w) for _ in range(agent.nz)]
                # get agent action
                actions = []

                action = agent.gpi_action(obs.flatten(), zs, w)
                actions.append(action)

                # use this for random partner
                # action_p = env.action_space.sample()
                # actions.append(action_p)

                # step env with selected action
                obs_next, rew, done, truncated, info = env.step(actions)

                agent_pos = env.agents[0].pos
                idx = env.grid.width * agent_pos[0] + agent_pos[1]
                obs_next = env.toroid(idx)

                # shaped reward
                rew_a += np.dot(w, agent.phi(obs, obs_next))
                # standard env reward
                ep_rew += rew

                loss_sum = 0

                for z_i in zs:
                    losses = agent.update(obs, action, obs_next, z_i)
                    loss_sum += np.sum(losses)

                # loss = agent.update(obs, action, obs_next)
                # loss = np.sum(loss)

                running_loss += loss_sum / agent.nz

                # save gif of last episode for fun
                if ep == episodes - 1:
                    frames.append(env.render())
                if done:
                    break
                obs = obs_next

            # tensorboard logging
            writer.add_scalar("training loss", running_loss / t, ep)
            writer.add_scalar("reward", ep_rew, ep)
            writer.add_scalar("shaped_reward", rew_a, ep)
            writer.add_scalar("ep_length", t, ep)
            writer.add_scalar("num_balls_collected", env.collected_balls, ep)
            writer.add_scalar("num_agent1_ball1", info["agent1ball1"], ep)
            writer.add_scalar("num_agent1_ball2", info["agent1ball2"], ep)
            writer.add_scalar("num_agent1_ball3", info["agent1ball3"], ep)
            # writer.add_scalar('num_agent2_ball1', info['agent2ball1'], ep)
            # writer.add_scalar('num_agent2_ball2', info['agent2ball2'], ep)
            # writer.add_scalar('num_agent2_ball3', info['agent2ball3'], ep)

            # Do gpi evaluation
            if ep % gpi_eval_freq == 0:
                obs, _ = env.reset(seed=seed)
                agent_pos = env.agents[0].pos
                idx = env.grid.width * agent_pos[0] + agent_pos[1]
                obs = env.toroid(idx)
                test_reward: float = 0
                shaped_test_reward: float = 0
                # use this code to compare toroidal obs with standard env obs
                # plt.imshow(20*obs[:,:,0] + 50*obs[:,:,1] + 100*obs[:,:,2] + 70*obs[:,:,3])
                # plt.savefig('toroid.png')
                # print(str(env))
                done = False

                for t in range(episode_length):
                    zs: list[NDArray] = ws
                    # get agent action
                    actions = []

                    action = agent.gpi_action(obs.flatten(), zs, w_test)
                    actions.append(action)

                    # step env with selected action
                    obs_next, rew, done, truncated, info = env.step(actions)

                    agent_pos = env.agents[0].pos
                    idx = env.grid.width * agent_pos[0] + agent_pos[1]
                    obs_next = env.toroid(idx)

                    # shaped reward
                    shaped_test_reward += np.dot(w_test, agent.phi(obs, obs_next))
                    # standard env reward
                    test_reward += rew

                    # save gif of last episode for fun
                    if ep == episodes - 1:
                        frames.append(env.render())
                    if done:
                        break
                    obs = obs_next

                writer.add_scalar("test_reward", test_reward, ep)
                writer.add_scalar("shaped_test_reward", shaped_test_reward, ep)

        writer.close()
        save_frames_as_gif(
            frames,
            ep="ind-sfdqn-random",
            path="./plots/",
            filename="collect-" + path_suffix,
        )
        torch.save(agent.psi1, model_dir + f"{path_suffix}_psi1.torch")
        torch.save(agent.psi2, model_dir + f"{path_suffix}_psi2.torch")
        torch.save(agent.psi3, model_dir + f"{path_suffix}_psi3.torch")


if __name__ == "__main__":
    main()
