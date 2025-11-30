import numpy as np
from environment_3d import LineFollowEnv3D  # updated import
from agent import DQNAgent
import pygame
import time
import os

RENDER = True
EPISODES = 800
MAX_STEPS = 1000
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

env = LineFollowEnv3D()
agent = DQNAgent()

episode_rewards = []

print("Starting training... (close window to stop)")

try:
    for ep in range(1, EPISODES+1):
        obs = env.reset()
        total_reward = 0.0
        for step in range(MAX_STEPS):
            if RENDER:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        raise SystemExit()

            action = agent.act(obs)
            next_obs, reward, done = env.step(action)
            agent.push(obs, action, reward, next_obs, float(done))
            agent.train_step()
            obs = next_obs
            total_reward += reward

            if RENDER:
                info = {
                    "Episode": ep,
                    "Reward": f"{total_reward:.2f}",
                    "Avg20": f"{np.mean(episode_rewards[-20:]):.2f}" if len(episode_rewards) >= 20 else "-",
                    "Epsilon": f"{agent.epsilon:.3f}"
                }
                env.render()
                time.sleep(0.001)

            if done:
                break

        episode_rewards.append(total_reward)
        if ep % 50 == 0:
            path = f"{SAVE_DIR}/dqn_ep{ep}.pth"
            agent.save(path)
            print("Saved", path)

    agent.save(f"{SAVE_DIR}/dqn_final.pth")
    print("Training finished. Model saved!")

finally:
    env.close()
