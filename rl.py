# rl.py
import random
from collections import deque, namedtuple
import torch, torch.nn as nn, torch.optim as optim

Transition = namedtuple("T", ("s","a","r","s2","done"))

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

class Agent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.9):
        self.q = DQN(state_dim, n_actions)
        self.tgt = DQN(state_dim, n_actions)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = deque(maxlen=50000)
        self.gamma = gamma
        self.n_actions = n_actions
        self.steps = 0

    def act(self, s, eps=0.1):
        self.steps += 1
        import random
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def push(self, *args): self.buf.append(Transition(*args))

    def train_step(self, batch_size=64):
        if len(self.buf) < batch_size: return None
        batch = random.sample(self.buf, batch_size)
        batch = Transition(*zip(*batch))
        import torch.nn.functional as F
        s  = torch.tensor(batch.s, dtype=torch.float32)
        a  = torch.tensor(batch.a, dtype=torch.long).unsqueeze(1)
        r  = torch.tensor(batch.r, dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(batch.s2, dtype=torch.float32)
        d  = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

        qsa = self.q(s).gather(1, a)
        with torch.no_grad():
            maxq_s2 = self.tgt(s2).max(1, keepdim=True)[0]
            y = r + self.gamma * (1 - d) * maxq_s2
        loss = F.smooth_l1_loss(qsa, y)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        return float(loss.item())

    def hard_update(self):
        self.tgt.load_state_dict(self.q.state_dict())

class AgentRegistry:
    def __init__(self, state_dim, n_actions):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.store = {}
    def get(self, user_id: int) -> Agent:
        if user_id not in self.store:
            self.store[user_id] = Agent(self.state_dim, self.n_actions)
        return self.store[user_id]
