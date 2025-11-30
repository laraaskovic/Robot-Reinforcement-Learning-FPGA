# main.py
import numpy as np
from environment import LineFollowEnv
from agent import DQNAgent
import pygame
import time
import os

RENDER = True         # set True to view rendering; False trains much faster
EPISODES = 800         # increase for stronger policies
MAX_STEPS = 1000       # per episode
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

env = LineFollowEnv()
agent = DQNAgent()

print("Starting training... (close the window to stop)")

episode_rewards = []
try:
    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        total_reward = 0.0
        for step in range(MAX_STEPS):
            # handle quit events if rendering
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
                env.render()
            # small sleep only when rendering to avoid hogging CPU unnecessarily
            if RENDER:
                time.sleep(0.001)

            if done:
                break

        episode_rewards.append(total_reward)

        # logging
        if ep % 5 == 0:
            avg20 = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 1 else total_reward
            print(f"Episode {ep}/{EPISODES}  reward={total_reward:.2f}  avg20={avg20:.2f}  epsilon={agent.epsilon:.3f}")

        # save occasionally
        if ep % 100 == 0:
            path = f"{SAVE_DIR}/dqn_ep{ep}.pth"
            agent.save(path)
            print("Saved", path)

    print("Training finished. Saving final model.")
    agent.save(f"{SAVE_DIR}/dqn_final.pth")

finally:
    env.close()
