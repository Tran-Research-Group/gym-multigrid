from typing import TypeVar

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch import Tensor
from torch.types import Number
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from gymnasium.envs.registration import register
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


def gpi_policy(
    state,
    Psis: list[list[NNet]],
    w_test: NDArray,
    action_dim: int,
    device: torch.device,
) -> Number:
    with torch.no_grad():
        state = torch.from_numpy(state).to(device)
        Q: list[Tensor] = []
        for Psi in Psis:
            q_values = (
                Psi[0](state) * w_test[0]
                + Psi[1](state) * w_test[1]
                + Psi[2](state) * w_test[2]
            )
            Q.append(q_values.flatten())

        Q_tensor: Tensor = torch.stack(Q)
        action: Number = torch.argmax(Q_tensor).item() % action_dim

    return action


def phi(state, next_state) -> NDArray:
    # how many of each type of object was picked up between s and s'
    ball1 = np.sum(state[:, :, 0]) - np.sum(next_state[:, :, 0])
    ball2 = np.sum(state[:, :, 1]) - np.sum(next_state[:, :, 1])
    ball3 = np.sum(state[:, :, 2]) - np.sum(next_state[:, :, 2])
    return np.array([ball1, ball2, ball3])


def main():
    models_dir: str = "models/sfdqn"
    weights: list[tuple[float, ...]] = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0)]
    lr: float = 3e-5
    rep_idx: int = 0
    num_features: int = 3
    model_filename: str = f"lr_{lr}_rep_{rep_idx}"
    device = torch.device("cuda:1")

    tensor_board_path: str = f"runs/sfdqn/sfdqn_gpi_eval"

    Psis: list[list[NNet]] = []

    for weight in weights:
        Psi: list[NNet] = []

        for i in range(num_features):
            Psi.append(
                torch.load(
                    f"{models_dir}/{model_filename}_w_{'_'.join([str(e) for e in weight])}_psi{i+1}.torch"
                ).to(device)
            )

        Psis.append(Psi)

    writer = SummaryWriter(tensor_board_path)
    register(
        id="multigrid-collect-more-v0",
        entry_point="gym_multigrid.envs:CollectGame3Obj2Agent",
    )
    env = gym.make("multigrid-collect-more-v0")
    w_test = np.array([-1.0, 1.0, 1.0])  # red, orange, yellow

    frames = []
    episodes = 50_000
    for ep in tqdm(range(episodes), desc="GPI-SFDQN-eval"):
        obs, _ = env.reset()
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
        for t in range(100):
            # use this code for live rendering
            # env.render(mode='human', highlight=True if env.partial_obs else False)
            # time.sleep(0.1)
            # frames.append(env.render(mode="rgb_array"))

            # get agent action
            actions = []
            action = gpi_policy(obs.flatten(), Psis, w_test, env.ac_dim, device)
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
            rew_a += np.dot(w_test, phi(obs, obs_next))
            # standard env reward
            ep_rew += rew

            # save gif of last episode for fun
            if ep == episodes - 1:
                frames.append(env.render())
            if done:
                break
            obs = obs_next

        # tensorboard logging
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

    writer.close()
    save_frames_as_gif(
        frames,
        ep="ind-sfdqn-random",
        path="./plots/",
        filename="collect-" + "sfdqn_eval",
    )


if __name__ == "__main__":
    main()
