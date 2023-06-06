from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium.envs.registration import register
from matplotlib import animation
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='collect-multiple', type=str)

args = parser.parse_args()

if args.env == 'collect-multiple':
        register(
            id='multigrid-collect-more-v0',
            entry_point='gym_multigrid.envs:CollectGame3Obj2Agent'
        )
        env = gym.make('multigrid-collect-more-v0')

def save_frames_as_gif(frames, path='./', filename='collect-random', ep=0):
    filename = filename + str(ep) + '.gif'
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    plt.close()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        info = self.locals["infos"][0]
        self.logger.record("agent1ball1", info['agent1ball1'])
        self.logger.record("agent1ball2", info['agent1ball2'])
        self.logger.record("agent1ball3", info['agent1ball3'])
        self.logger.record("agent2ball1", info['agent2ball1'])
        self.logger.record("agent2ball2", info['agent2ball2'])
        self.logger.record("agent2ball3", info['agent2ball3'])
        return True

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_random_collect_tensorboard/")
model.learn(total_timesteps=1_000_000, callback=TensorboardCallback())
vec_env = model.get_env()
obs = vec_env.reset()
frames = []
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    actions = [action, 0]
    obs, reward, done, info = vec_env.step(actions)
    frames.append(vec_env.render())

print(info)
save_frames_as_gif(frames)
model.save("ppo_collect")
model.load("ppo_collect")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
env.close()