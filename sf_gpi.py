import gymnasium as gym
from gymnasium.envs.registration import register
from gym_multigrid.utils.misc import set_seed, save_frames_as_gif
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class SFPartnerAgent:
    def __init__(self, filename) -> None:
        self.psi1 = torch.load("models/" + filename + "_psi1.torch")
        self.psi1.eval()
        self.psi2 = torch.load("models/" + filename + "_psi2.torch")
        self.psi2.eval()
        self.psi3 = torch.load("models/" + filename + "_psi3.torch")
        self.psi3.eval()
        if filename == "red":
            self.w = np.array([1.0, 0.0, 0.0])
        elif filename == "orange":
            self.w = np.array([0.0, 1.0, 0.0])
        elif filename == "yellow":
            self.w = np.array([0.0, 0.0, 1.0])
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

class SFGPIAgent():
    def __init__(self, pretrained, w) -> None:
        self.pretrained = pretrained
        self.w = w
    
    def get_action(self, state):
        state = torch.from_numpy(state)
        qvals = []
        for psis in self.pretrained:
            vals = (
                psis[0](state) * self.w[0]
                + psis[1](state) * self.w[1]
                + psis[2](state) * self.w[2]
            )
            qvals.append(vals.detach().numpy())
        max_vals = np.max(qvals, axis=0)
        return np.argmax(max_vals)

    def phi(self, state, next_state):
        # how many of each type of object was picked up between s and s'
        ball1 = np.sum(state[:, :, 0]) - np.sum(next_state[:, :, 0])
        ball2 = np.sum(state[:, :, 1]) - np.sum(next_state[:, :, 1])
        ball3 = np.sum(state[:, :, 2]) - np.sum(next_state[:, :, 2])
        return np.array([ball1, ball2, ball3])

def main():
    seed = 42
    set_seed(seed=seed)
    register(
        id="multigrid-collect-v0",
        entry_point="gym_multigrid.envs:CollectGame3Obj2Agent",
    )
    env = gym.make("multigrid-collect-v0")
    w = np.array([1.0, 1.0, 1.0])  # red, orange, yellow
    pretrained = []
    pretrained.append([torch.load("models/psi1-redrandp-nofactor.torch").eval(),
                      torch.load("models/psi2-redrandp-nofactor.torch").eval(),
                      torch.load("models/psi3-redrandp-nofactor.torch").eval()])
    pretrained.append([torch.load("models/psi1-orangerandp-nofactor.torch").eval(),
                         torch.load("models/psi2-orangerandp-nofactor.torch").eval(),
                         torch.load("models/psi3-orangerandp-nofactor.torch").eval()])
    agent = SFGPIAgent(pretrained=pretrained, w=w)
    partner = SFPartnerAgent(filename='yellow')
    writer = SummaryWriter(comment='sf-yellowrandp-gpizero')
    frames = []
    episodes = 1#5000
    for ep in tqdm(range(episodes), desc="SF-yellow-newpartner-training"):
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
            action = agent.get_action(obs.flatten())
            actions.append(action)
            actions.append(partner.get_action(obs.flatten()))

            # step env with selected action
            obs_next, rew, done, truncated, info = env.step(actions)

            agent_pos = env.agents[0].pos
            idx = env.grid.width * agent_pos[0] + agent_pos[1]
            obs_next = env.toroid(idx)

            # shaped reward
            s_rew += np.dot(w, agent.phi(obs, obs_next))
            # standard env reward
            ep_rew += np.sum(rew)
            ep_rew_a += rew[0]
            ep_rew_p += rew[1]

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
        writer.add_scalar("learner_shaped_reward", info["agent1ball3"] + info["agent1ball2"] + info["agent1ball1"], ep)
        writer.add_scalar("partner_shaped_reward", info["agent2ball2"], ep)
        writer.add_scalar("ep_length", t, ep)
        writer.add_scalar("num_balls_collected", env.collected_balls, ep)
        writer.add_scalar("num_agent1_ball1", info["agent1ball1"], ep)
        writer.add_scalar("num_agent1_ball2", info["agent1ball2"], ep)
        writer.add_scalar("num_agent1_ball3", info["agent1ball3"], ep)
        writer.add_scalar('num_agent2_ball1', info['agent2ball1'], ep)
        writer.add_scalar('num_agent2_ball2', info['agent2ball2'], ep)
        writer.add_scalar('num_agent2_ball3', info['agent2ball3'], ep)

    writer.close()
    save_frames_as_gif(frames, path="./plots/", ep="sf-gpi-yellow")


if __name__ == "__main__":
    main()