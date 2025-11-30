# agent.py
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")

class ReplayBuffer:
    def __init__(self, capacity=30000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim=5, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self,
                 input_dim=5,
                 action_count=3,
                 lr=1e-3,
                 gamma=0.98,
                 eps_start=1.0,
                 eps_end=0.05,
                 eps_decay=0.997,
                 batch_size=64,
                 target_update=400):
        self.action_count = action_count
        self.model = DQN(input_dim, action_count).to(device)
        self.target = DQN(input_dim, action_count).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma

        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.replay = ReplayBuffer()
        self.batch_size = batch_size
        self.learn_step = 0
        self.target_update = target_update

    def act(self, state):
        # state: numpy array
        if random.random() < self.epsilon or len(self.replay) < self.batch_size:
            return random.randrange(self.action_count)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.model(state_t)
        return int(torch.argmax(q, dim=1).item())

    def push(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # current Q
        curr_q = self.model(states_t).gather(1, actions_t)
        # target Q from target network
        with torch.no_grad():
            next_q = self.target(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1 - dones_t) * (self.gamma * next_q)

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping to stabilize
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        # decay epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target.load_state_dict(self.model.state_dict())

    def save(self, path="dqn_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="dqn_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.target.load_state_dict(self.model.state_dict())
