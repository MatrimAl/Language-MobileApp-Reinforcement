import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Tuple
import pickle
import os
from datetime import datetime

class DQNetwork(nn.Module):
    """
    Deep Q-Network Neural Network
    
    Architecture:
        Input: State vector (12 features)
        Hidden: 3 dense layers (128, 64, 32 units)
        Output: Q-values for each action (5 difficulty levels)
    """
    
    def __init__(self, state_size: int = 12, action_size: int = 5):
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for ReLU layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Linear output for Q-values
        return x


class DQNAgent:
    """
    Deep Q-Network Agent for Language Learning (PyTorch)
    
    Network Architecture:
        Input: State vector (12 features)
        Hidden: 3 dense layers (128, 64, 32 units)
        Output: Q-values for each action (5 difficulty levels)
    """
    
    def __init__(
        self,
        state_size: int = 12,
        action_size: int = 5,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_size: int = 10000
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Device configuration with detailed GPU info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"🚀 GPU Training Enabled!")
            print(f"   ├─ Device: {gpu_name}")
            print(f"   ├─ Memory: {gpu_memory:.2f} GB")
            print(f"   └─ CUDA Version: {torch.version.cuda}")
        else:
            print(f"⚠️  CPU Training Mode")
            print(f"   └─ For GPU acceleration (CUDA 11.8), install:")
            print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Networks
        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNetwork(state_size, action_size).to(self.device)
        self.update_target_model()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon_values': []
        }
    
    def update_target_model(self):
        """Target network'ü güncelle"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Experience'i memory'e ekle"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Action seç (epsilon-greedy policy)
        Training mode: epsilon-greedy
        Inference mode: greedy (best action)
        """
        if training and np.random.rand() <= self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)
        
        # Exploit: best action based on Q-values
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action = q_values.argmax().item()
        self.model.train()
        return action
    
    def replay(self) -> float:
        """
        Experience replay ile model eğit
        Returns: loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Random mini-batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        self.model.train()
        current_q_values = self.model(states_tensor)
        current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Double DQN)
        self.target_model.eval()
        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def train(self, env, episodes: int = 1000, update_target_freq: int = 10,
              save_path: str = None, tensorboard_log: bool = False):
        """
        DQN agent'i eğit
        
        Args:
            env: LanguageLearningEnv instance
            episodes: Toplam episode sayısı
            update_target_freq: Target network güncelleme sıklığı
            save_path: Model kaydetme yolu
            tensorboard_log: TensorBoard logging aktif mi
        """
        from torch.utils.tensorboard import SummaryWriter
        
        writer = None
        if tensorboard_log:
            log_dir = f"logs/dqn_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            writer = SummaryWriter(log_dir)
        
        print("🚀 DQN Training başlatılıyor...")
        print(f"Episodes: {episodes}, Batch Size: {self.batch_size}")
        print(f"Epsilon: {self.epsilon} → {self.epsilon_min} (decay: {self.epsilon_decay})")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Action seç
                action = self.act(state, training=True)
                
                # Environment'ta adım at
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Remember
                self.remember(state, action, reward, next_state, done)
                
                # Learn
                loss = self.replay()
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            # Update target network
            if episode % update_target_freq == 0:
                self.update_target_model()
            
            # Log metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['epsilon_values'].append(self.epsilon)
            
            # TensorBoard logging
            if writer:
                writer.add_scalar('Reward/Episode', episode_reward, episode)
                writer.add_scalar('Epsilon', self.epsilon, episode)
                if len(self.training_history['losses']) > 0:
                    writer.add_scalar('Loss/Episode', np.mean(self.training_history['losses'][-episode_length:]), episode)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg Reward (10): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Memory: {len(self.memory)}")
            
            # Save model periodically
            if save_path and (episode + 1) % 100 == 0:
                self.save(save_path)
                print(f"✅ Model saved: {save_path}")
        
        if writer:
            writer.close()
        
        print("\n🎉 Training tamamlandı!")
        print(f"Final Epsilon: {self.epsilon:.3f}")
        print(f"Avg Reward (last 100): {np.mean(self.training_history['episode_rewards'][-100:]):.2f}")
        
        if save_path:
            self.save(save_path)
            print(f"✅ Final model saved: {save_path}")
    
    def save(self, path: str):
        """Model ve training history'yi kaydet"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f"{path}_model.pth")
        
        # Save agent parameters
        agent_params = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'training_history': self.training_history
        }
        
        with open(f"{path}_params.pkl", 'wb') as f:
            pickle.dump(agent_params, f)
    
    def load(self, path: str):
        """Model ve parametreleri yükle"""
        # Load model checkpoint
        checkpoint = torch.load(f"{path}_model.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load agent parameters
        with open(f"{path}_params.pkl", 'rb') as f:
            agent_params = pickle.load(f)
        
        self.epsilon = agent_params['epsilon']
        self.training_history = agent_params['training_history']
        
        print(f"✅ Model loaded from {path}")
        print(f"Current Epsilon: {self.epsilon:.3f}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """State için Q-values'ları döndür (debugging için)"""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def get_action_explanation(self, state: np.ndarray) -> dict:
        """
        Seçilen action için açıklama (görselleştirme için)
        """
        q_values = self.get_q_values(state)
        action = np.argmax(q_values)
        
        difficulty_names = ['Beginner', 'Elementary', 'Intermediate', 'Advanced', 'Expert']
        
        return {
            'action': int(action),
            'difficulty': difficulty_names[action],
            'q_values': q_values.tolist(),
            'confidence': float(q_values[action] - np.mean(q_values)),
            'exploration_rate': float(self.epsilon)
        }
    
    def get_device_info(self) -> dict:
        """
        GPU/CPU kullanım bilgilerini döndür
        """
        info = {
            'device_type': str(self.device),
            'is_cuda': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda,
                'current_device': torch.cuda.current_device(),
                'memory_allocated_mb': torch.cuda.memory_allocated(0) / 1024**2,
                'memory_reserved_mb': torch.cuda.memory_reserved(0) / 1024**2,
                'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return info
