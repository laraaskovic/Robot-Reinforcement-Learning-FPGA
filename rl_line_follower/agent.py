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

class RecurrentQNetwork(nn.Module):
    def __init__(self, input_dim=5, output_dim=3, hidden_size=64, rnn_type="gru", num_layers=1):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embed = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(64, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(64, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_dim))

    def forward(self, x, h=None):
        # x: (B, T, input_dim)
        z = self.embed(x)
        out, h_new = self.rnn(z, h)
        q = self.head(out)
        return q, h_new

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
        if self.rnn_type == "lstm":
            return (h0, torch.zeros_like(h0))
        return h0

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

class SequenceReplayBuffer:
    def __init__(self, capacity=20000, seq_len=8):
        self.buffer = collections.deque(maxlen=capacity)
        self.seq_len = seq_len

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) <= self.seq_len:
            return None
        data = list(self.buffer)
        max_start = len(data) - self.seq_len
        idxs = np.random.randint(0, max_start, size=batch_size)
        seq_states, seq_actions, seq_rewards, seq_next_states, seq_dones = [], [], [], [], []
        for idx in idxs:
            seq = data[idx:idx + self.seq_len]
            s, a, r, ns, d = map(np.array, zip(*seq))
            seq_states.append(s)
            seq_actions.append(a)
            seq_rewards.append(r)
            seq_next_states.append(ns)
            seq_dones.append(d)
        return (
            np.stack(seq_states),
            np.stack(seq_actions),
            np.stack(seq_rewards),
            np.stack(seq_next_states),
            np.stack(seq_dones),
        )

    def __len__(self):
        return len(self.buffer)

class RecurrentDQNAgent:
    def __init__(
        self,
        input_dim=5,
        action_count=3,
        lr=1e-3,
        gamma=0.98,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        batch_size=16,
        target_update=600,
        seq_len=8,
        rnn_type="gru",
    ):
        self.action_count = action_count
        self.model = RecurrentQNetwork(input_dim, action_count, rnn_type=rnn_type).to(device)
        self.target = RecurrentQNetwork(input_dim, action_count, rnn_type=rnn_type).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.seq_len = seq_len

        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.replay = SequenceReplayBuffer(seq_len=seq_len)
        self.batch_size = batch_size
        self.learn_step = 0
        self.target_update = target_update
        self.hidden = None
        self.rnn_type = rnn_type.lower()

    def _detach_hidden(self, h):
        if h is None:
            return None
        if self.rnn_type == "lstm":
            return (h[0].detach(), h[1].detach())
        return h.detach()

    def reset_hidden(self):
        self.hidden = self.model.init_hidden(batch_size=1)

    def act(self, state):
        if self.hidden is None:
            self.reset_hidden()
        if random.random() < self.epsilon or len(self.replay) < self.batch_size:
            action = random.randrange(self.action_count)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, -1)
            with torch.no_grad():
                q, self.hidden = self.model(state_t, self.hidden)
            action = int(torch.argmax(q[:, -1, :], dim=1).item())
        self.hidden = self._detach_hidden(self.hidden)
        return action

    def push(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self):
        batch = self.replay.sample(self.batch_size)
        if batch is None:
            return

        states, actions, rewards, next_states, dones = batch
        B, T = states.shape[0], states.shape[1]
        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        q_pred, _ = self.model(states_t)
        q_pred = q_pred.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            q_next, _ = self.target(next_states_t)
            q_next = q_next.max(2)[0]
            target = rewards_t + (1 - dones_t) * (self.gamma * q_next)

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target.load_state_dict(self.model.state_dict())

    def save(self, path="dqn_rnn_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="dqn_rnn_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.target.load_state_dict(self.model.state_dict())
