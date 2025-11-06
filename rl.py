# rl.py
# Hibrit DQN: Global backbone + kullanıcıya özel head
# + Hedef odaklı seçim (hedef ±1 bandında maske)
# + ε-decay (rastgeleliği zamanla azalt)
# + Warm-start (yeni kullanıcı head'i global'den kopyalanır)

import random
from collections import deque, namedtuple, defaultdict
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple("T", ("s","a","r","s2","done"))

# ----------------------------
# 1) MİMARİ: Backbone + Head
# ----------------------------
class QBackbone(nn.Module):
    """
    Ortak 'özellik çıkarıcı' gövde (tüm kullanıcılarda paylaşılır).
    Girdi: state (ör. 13 boyut)
    Çıktı: gizli vektör (128)
    """
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
    def forward(self, x):
        return self.body(x)

class QHead(nn.Module):
    """
    Kullanıcıya özel küçük çıkış katmanı (sadece 5 aksiyon skoru üretir).
    """
    def __init__(self, hidden: int = 128, n_actions: int = 5):
        super().__init__()
        self.out = nn.Linear(hidden, n_actions)
    def forward(self, h):
        return self.out(h)

class QNet(nn.Module):
    """
    Tam Q ağı: backbone (paylaşılan) + head (kullanıcıya özel)
    Not: forward sırasında backbone -> head zinciri.
    """
    def __init__(self, backbone: QBackbone, head: QHead):
        super().__init__()
        self.backbone = backbone       # paylaşılan referans
        self.head = head               # kullanıcıya özgü katman
    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)

# ----------------------------
# 2) AJAN
# ----------------------------
class Agent:
    """
    Bir kullanıcı için Agent:
      - Paylaşılan backbone'u kullanır
      - Kendine ait head (ve hedef kopyası) + optimizer
      - ε-decay, maskeli seçim, replay buffer
    """
    def __init__(
        self,
        shared_backbone: QBackbone,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 0.20,
        eps_min: float = 0.02,
        eps_decay: float = 0.995,
        warm_steps: int = 60,          # ilk N adım hedef ±1 bandında kal
    ):
        self.n_actions = n_actions
        self.gamma = gamma

        # Kullanıcıya özel head + hedef ağ (target)
        self.head = QHead(hidden=128, n_actions=n_actions)
        self.tgt_head = QHead(hidden=128, n_actions=n_actions)
        self.tgt_head.load_state_dict(self.head.state_dict())

        # Paylaşılan backbone referansı (tek kopya)
        self.backbone = shared_backbone
        # Q ağları (paylaşılan backbone + farklı head)
        self.q = QNet(self.backbone, self.head)
        self.tgt = QNet(self.backbone, self.tgt_head)

        # Sadece head'i optimize ediyoruz (backbone'u sabit bırak)
        # İstersen backbone'u da eğitmek için param ekleyebilirsin.
        self.opt = optim.Adam(self.head.parameters(), lr=lr)

        self.buf = deque(maxlen=50000)
        self.steps = 0

        # ε zamanla azalacak
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # kısa süreli hedef bant politikası
        self.warm_steps = warm_steps

    # --- Yardımcı: ε'yi her adımda biraz düşür.
    def _step_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # --- Aksiyon seçimi: hedef odaklı, maskeli argmax + ε-greedy
    def act_biased(self, s, target_idx: int):
        """
        Öğretici mantık:
          - İlk warm_steps boyunca seçimleri hedef ±1 bandında tut.
          - Sonra da ε-greedy ama bant dışını maskeleyerek (zayıf da olsa)
            hedef çevresinde kalmayı teşvik et.
        """
        self.steps += 1

        # 1) Warm-up: İlk N adımda hedef ±1 bandından seç
        if self.steps <= self.warm_steps:
            band = [i for i in range(self.n_actions) if abs(i - target_idx) <= 1]
            if random.random() < self.eps:
                return random.choice(band)
            # exploit: bant içindeki en iyi
            with torch.no_grad():
                q = self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                q_masked = q.clone()
                for i in range(self.n_actions):
                    if i not in band:
                        q_masked[i] = -1e9
                return int(q_masked.argmax().item())

        # 2) Normal faz: ε-greedy + bant dışını zayıf maskele
        if random.random() < self.eps:
            # exploration: bant içi tercihli
            if random.random() < 0.8:
                band = [i for i in range(self.n_actions) if abs(i - target_idx) <= 1]
                return random.choice(band)
            return random.randrange(self.n_actions)

        # exploitation: bant dışını cezalandır
        with torch.no_grad():
            q = self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            mask = torch.tensor([0.0 if abs(i-target_idx)<=1 else -1.0 for i in range(self.n_actions)])
            # Not: -1.0 ek maske 'yumuşak' ceza (istersen -1e9 ile tamamen kes)
            qm = q + mask
            return int(qm.argmax().item())

    def push(self, *args):
        self.buf.append(Transition(*args))

    def train_step(self, batch_size=64):
        """
        Standart DQN güncellemesi.
        Not: Sadece head'i optimize ediyoruz (backbone sabit).
        """
        if len(self.buf) < batch_size:
            return None

        batch = random.sample(self.buf, batch_size)
        batch = Transition(*zip(*batch))

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
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
        self.opt.step()

        # her train adımından sonra ε biraz azalsın:
        self._step_eps()
        return float(loss.item())

    def hard_update(self):
        # hedef ağa sadece HEAD'i kopyalıyoruz (backbone zaten paylaşılıyor)
        self.tgt_head.load_state_dict(self.head.state_dict())

# ----------------------------
# 3) KAYIT / REGISTRY
# ----------------------------
class AgentRegistry:
    """
    Tek bir global backbone; her kullanıcı için ayrı head + buffer + optimizer.
    """
    def __init__(self, state_dim: int, n_actions: int, warm_start_head: QHead | None = None):
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Paylaşılan backbone (tek kopya)
        self.backbone = QBackbone(in_dim=state_dim, hidden=128)
        # İstersen backbone'u eğitilebilir yapmak için parametrelerini
        # optimizer'a eklersin. MVP: sabit/dondurulmuş bırakıyoruz.

        # Global head (isteğe bağlı): Warm-start için şablon
        self.global_head = warm_start_head or QHead(hidden=128, n_actions=n_actions)

        # Kullanıcı -> Agent haritası
        self.store: Dict[int, Agent] = {}

    def get(self, user_id: int) -> Agent:
        if user_id not in self.store:
            ag = Agent(shared_backbone=self.backbone, n_actions=self.n_actions)

            # Warm-start: yeni kullanıcının head'ini global head'den kopyala
            ag.head.load_state_dict(self.global_head.state_dict())
            ag.tgt_head.load_state_dict(ag.head.state_dict())

            self.store[user_id] = ag
        return self.store[user_id]

    def update_global_from_user(self, user_id: int, tau: float = 0.0):
        """
        (Opsiyonel) Bir kullanıcının head'ini global head'e harmanla.
        tau=0: doğrudan kopya; tau∈(0,1): yumuşak güncelleme.
        """
        ag = self.store[user_id]
        gsd = self.global_head.state_dict()
        usd = ag.head.state_dict()
        for k in gsd.keys():
            gsd[k] = tau * usd[k] + (1 - tau) * gsd[k]
        self.global_head.load_state_dict(gsd)
