import os

from stable_baselines3 import PPO
import torch
import imageio

from gym_multigrid.envs.labyrinth import LabyrinthEnv

total_timesteps: int = 1_000_000
tb_log_dir: str = "out/logs/laby/"
tb_log_name: str = "ctf_laby_ppo"
model_save_path: str = "out/models/laby_ppo"
# Create the environment
env = LabyrinthEnv()

print("GPU available: ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.exists(model_save_path + ".zip"):
    model = PPO.load(model_save_path, env=env, device=device)
else:
    # Create the RL model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_log_dir)

    # Train the model
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)

    # Save the model
    model.save(model_save_path)

# Save an animation
obs, _ = env.reset()

imgs = [env.render()]

while True:
    actions, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    imgs.append(env.render())

    if terminated or truncated:
        break

imageio.mimsave("out/animations/laby_ppo.gif", imgs, fps=5)
