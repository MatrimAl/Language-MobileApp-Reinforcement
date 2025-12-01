"""
V3 Ultra Training with Anti-Repetition + Realistic Simulation

Complete optimized training with:
1. Anti-repetition system (prevents word spam)
2. Realistic user simulation (cognitive models)
3. Vectorized environments (64 parallel)
4. Batched Q-value evaluation
5. Feature caching

Expected: 30x speedup + better generalization
"""

import numpy as np
import torch
from tqdm import tqdm
import time
import os

from vocabulary_data import VocabularyDataset
from language_env_vectorized import VectorizedLanguageEnvV3
from models_v3 import DuelingDQNAgent
from prioritized_replay import PrioritizedReplayBuffer
from config import DQN_CONFIG_V3, REWARD_CONFIG_V3
try:
    from enhanced_analytics import TrainingAnalytics
except:
    TrainingAnalytics = None  # Optional


class V3UltraTrainerFinal:
    """Final ultra-optimized trainer with all improvements."""
    
    def __init__(
        self,
        num_envs: int = 64,
        device: str = 'cuda'
    ):
        """Initialize trainer."""
        print("="*70)
        print("🚀 V3 ULTRA TRAINER - Final Version")
        print("="*70)
        print()
        
        self.num_envs = num_envs
        self.device = device
        
        # Load vocabulary
        self.vocab = VocabularyDataset()
        
        # Vectorized environment with anti-repetition
        print(f"✓ Creating {num_envs} vectorized environments...")
        self.env = VectorizedLanguageEnvV3(
            num_envs=num_envs,
            vocab_dataset=self.vocab,
            max_steps=50
        )
        
        # Agent
        print(f"✓ Creating Dueling Double DQN agent...")
        self.agent = DuelingDQNAgent(
            state_dim=17,
            vocab_dataset=self.vocab,
            device=device
        )
        self.target_agent = DuelingDQNAgent(
            state_dim=17,
            vocab_dataset=self.vocab,
            device=device
        )
        self.target_agent.load_state_dict(self.agent.state_dict())
        
        # Prioritized replay
        print(f"✓ Creating prioritized replay buffer...")
        self.buffer = PrioritizedReplayBuffer(
            capacity=DQN_CONFIG_V3['replay_buffer_size'],
            alpha=DQN_CONFIG_V3['per_alpha'],
            beta_start=DQN_CONFIG_V3['per_beta_start'],
            beta_frames=DQN_CONFIG_V3['per_beta_frames']
        )
        
        # Feature cache
        print(f"✓ Caching word features...")
        self.word_feature_cache = {}
        for word_id in range(len(self.vocab.vocab_df)):
            features = self.vocab.get_word_features(word_id)
            self.word_feature_cache[word_id] = (
                features['difficulty'],
                features['concreteness']
            )
        
        # Analytics (simple arrays)
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_zpd_hits = []
        self.episode_lengths = []
        
        print(f"\n{'='*70}")
        print(f"Configuration:")
        print(f"  Environments: {num_envs}")
        print(f"  Device: {device}")
        print(f"  State dim: 17")
        print(f"  Vocabulary size: {len(self.vocab.vocab_df)}")
        print(f"\n  Features:")
        print(f"    ✓ Anti-repetition (3 mechanisms)")
        print(f"    ✓ Realistic simulation (6 cognitive models)")
        print(f"    ✓ Dueling Double DQN")
        print(f"    ✓ Prioritized Experience Replay")
        print(f"    ✓ Noisy Networks")
        print(f"{'='*70}\n")
    
    def _select_actions_batched(
        self,
        states: np.ndarray,
        available_batch: list,
        estimated_thetas: np.ndarray
    ) -> np.ndarray:
        """Batched action selection using agent's evaluate_actions."""
        selected_actions = np.zeros(self.num_envs, dtype=np.int32)
        
        with torch.no_grad():
            for env_idx in range(self.num_envs):
                state_tensor = torch.FloatTensor(states[env_idx]).to(self.device)
                available = available_batch[env_idx]
                theta = estimated_thetas[env_idx]
                
                # Use agent's method
                q_values = self.agent.evaluate_actions(state_tensor, available, theta)
                best_idx = q_values.argmax().item()
                selected_actions[env_idx] = available[best_idx]
        
        return selected_actions
    
    def train(
        self,
        num_episodes: int = 1000,
        save_freq: int = 100,
        test_mode: bool = False
    ):
        """Train the agent."""
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Expected time: ~{num_episodes * 0.035:.0f}s ({num_episodes * 0.035 / 60:.1f} minutes)\n")
        
        states = self.env.reset()
        episode_count = 0
        global_step = 0
        
        pbar = tqdm(total=num_episodes, desc="Training")
        
        start_time = time.time()
        
        while episode_count < num_episodes:
            # Get available actions
            available_batch = self.env.get_available_actions_batch()
            
            # Estimate thetas (simplified - use current user theta)
            estimated_thetas = self.env.users.theta.copy()
            
            # Select actions (batched)
            actions = self._select_actions_batched(states, available_batch, estimated_thetas)
            
            # Step
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # Store transitions
            for i in range(self.num_envs):
                self.buffer.push(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    done=dones[i],
                    next_available_actions=available_batch[i][:50],
                    estimated_theta=estimated_thetas[i]
                )
            
            # Train
            if len(self.buffer) >= DQN_CONFIG_V3['batch_size']:
                batch, indices, weights = self.buffer.sample(
                    DQN_CONFIG_V3['batch_size'],
                    self.device
                )
                
                loss, td_errors = self.agent.train_step_double_dqn(batch, self.target_agent)
                self.buffer.update_priorities(indices, td_errors)
                
                # Soft update
                self.agent.soft_update(
                    self.target_agent,
                    tau=DQN_CONFIG_V3['target_update_tau']
                )
            
            # Record completed episodes
            for i, info in enumerate(infos):
                if dones[i]:
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_accuracies.append(info['episode']['accuracy'])
                        self.episode_zpd_hits.append(info['episode']['zpd_hits'])
                        self.episode_lengths.append(info['episode']['l'])
                        episode_count += 1
                        pbar.update(1)
            
            # Reset done envs
            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                states[done_indices] = self.env.reset(done_indices)[done_indices]
            
            states = next_states
            global_step += 1
            
            # Save checkpoint
            if episode_count % save_freq == 0 and episode_count > 0:
                self.agent.save(f'checkpoints/v3_ultra_final_ep{episode_count}.pt')
        
        pbar.close()
        
        elapsed = time.time() - start_time
        
        # Final save
        self.agent.save('checkpoints/v3_ultra_final.pt')
        
        # Generate plots
        print(f"\n{'='*70}")
        print("📊 Generating detailed visualizations...")
        print(f"{'='*70}")
        
        try:
            from training_plots import plot_training_metrics
            plot_training_metrics(
                self.episode_rewards,
                self.episode_accuracies,
                self.episode_zpd_hits,
                self.episode_lengths,
                save_dir='plots_v3_final'
            )
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
        
        # Generate summary
        print(f"\n{'='*70}")
        print("📊 Training Summary")
        print(f"{'='*70}")
        print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"Episodes: {episode_count}")
        print(f"Steps/second: {global_step/elapsed:.1f}")
        print(f"Episodes/second: {episode_count/elapsed:.2f}")
        
        # Calculate metrics
        rewards_arr = np.array(self.episode_rewards)
        acc_arr = np.array(self.episode_accuracies)
        zpd_arr = np.array(self.episode_zpd_hits)
        
        print(f"\nFinal metrics (last 100 episodes):")
        print(f"  Avg reward: {rewards_arr[-100:].mean():.2f}")
        print(f"  Avg accuracy: {acc_arr[-100:].mean():.2%}")
        print(f"  Avg ZPD hits: {zpd_arr[-100:].mean():.1f}")
        print(f"\nOverall:")
        print(f"  Total reward: {rewards_arr.sum():.1f}")
        print(f"  Mean accuracy: {acc_arr.mean():.2%}")
        print(f"  Mean ZPD hits: {zpd_arr.mean():.1f}")
        
        print(f"\n{'='*70}")
        print("🎉 TRAINING COMPLETE!")
        print(f"Model saved: checkpoints/v3_ultra_final.pt")
        print(f"{'='*70}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='V3 Ultra Final Training')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test-mode', action='store_true')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = V3UltraTrainerFinal(
        num_envs=args.num_envs,
        device=args.device
    )
    
    # Train
    trainer.train(
        num_episodes=args.episodes,
        save_freq=100,
        test_mode=args.test_mode
    )


if __name__ == "__main__":
    main()
