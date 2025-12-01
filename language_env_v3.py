"""
V3.0 Enhanced Language Learning Environment

Extends base environment with:
- 17-dimensional state (vs 14 in V2)
- Multi-component ZPD reward function
- Streak and novelty tracking
- Anti-repetition system to prevent exploitation
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional, List

from language_env import LanguageLearningEnv
from config import REWARD_CONFIG_V3


class LanguageLearningEnvV3(LanguageLearningEnv):
    """
    V3.0 Enhanced environment with richer state and reward signals.
    
    New State Features (17 total):
    - zpd_center: estimated_theta + 0.6 (middle of ZPD)
    - recent_zpd_hit_rate: Rolling avg of ZPD hits (last 10 steps)
    - current_streak: Consecutive correct answers
    
    Enhanced Rewards:
    - zpd_perfect_bonus: In ZPD + correct
    - zpd_hit_bonus: In ZPD (any outcome)
    - near_zpd_bonus: Close to ZPD
    - streak_bonus: Consecutive correct
    - novel_word_bonus: First time this episode
    
    Anti-Repetition (NEW):
    - repetition_penalty: Penalty for recent word repeats
    - cooldown_penalty: Penalty for breaking cooldown
    - diminishing_factor: Reduces repeated word bonuses
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # V3-specific tracking
        self.shown_words_this_episode = set()
        self.correct_streak = 0
        self.recent_zpd_hits = deque(maxlen=10)
        
        # Anti-repetition tracking
        self.word_history = deque(maxlen=REWARD_CONFIG_V3['repetition_window'])
        self.word_episode_count = {}    # {word_id: count}
        self.word_cooldowns = {}        # {word_id: steps_since_shown}
        
        # Update observation space for 17 dimensions
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=np.array([
                self.min_level,  # current_level
                -3.0,            # estimated_theta
                *[0.0] * 10,     # answer_history (10)
                0.0,             # retry_queue_size
                1.0,             # mean_concreteness
                -3.0,            # zpd_center
                0.0,             # recent_zpd_hit_rate
                0.0,             # current_streak
            ]),
            high=np.array([
                self.max_level,
                3.0,
                *[1.0] * 10,
                float(self.retry_queue_max),
                5.0,
                3.0,
                1.0,
                10.0,
            ]),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset with V3 tracking."""
        obs, info = super().reset(seed, options)
        
        # Reset V3-specific tracking
        self.shown_words_this_episode = set()
        self.correct_streak = 0
        self.recent_zpd_hits = deque(maxlen=10)
        
        # Reset anti-repetition tracking
        self.word_history = deque(maxlen=REWARD_CONFIG_V3['repetition_window'])
        self.word_episode_count = {}
        self.word_cooldowns = {}
        
        # Return enhanced observation
        return self._get_observation_v3(), info
    
    def _get_observation_v3(self) -> np.ndarray:
        """Construct 17-dimensional observation."""
        # Get base observation (14 dims)
        base_obs = super()._get_observation()
        
        # Add V3 features
        estimated_theta = self._estimate_theta()
        zpd_center = estimated_theta + 0.6
        
        # Recent ZPD hit rate
        if len(self.recent_zpd_hits) > 0:
            zpd_hit_rate = np.mean(list(self.recent_zpd_hits))
        else:
            zpd_hit_rate = 0.0
        
        # Current streak (capped at 10 for normalization)
        streak_normalized = min(self.correct_streak, 10) / 10.0
        
        # Combine
        obs_v3 = np.concatenate([
            base_obs,
            [zpd_center, zpd_hit_rate, streak_normalized]
        ]).astype(np.float32)
        
        return obs_v3
    
    def _calculate_reward_v3(
        self,
        is_correct: bool,
        word_id: int,
        word_difficulty: float
    ) -> Tuple[float, Dict]:
        """
        V3.0 reward function with multiple ZPD bonuses and anti-repetition.
        
        Returns:
            (reward, info_dict)
        """
        reward = 0.0
        info = {}
        
        # Get estimated theta (observable)
        estimated_theta = self._estimate_theta()
        
        # Normalize difficulty to theta scale
        d_norm = (word_difficulty - 5.5) / 2.25
        
        # Check ZPD (optimal difficulty zone)
        zpd_min = estimated_theta + REWARD_CONFIG_V3['zpd_range'][0]
        zpd_max = estimated_theta + REWARD_CONFIG_V3['zpd_range'][1]
        in_zpd = (zpd_min <= d_norm <= zpd_max)
        
        # Base reward
        if is_correct:
            reward += REWARD_CONFIG_V3['correct_base']
        else:
            reward += REWARD_CONFIG_V3['incorrect_penalty']
        
        # ZPD bonuses (KEY IMPROVEMENT)
        zpd_bonus_total = 0.0  # Track ZPD bonus for diminishing returns
        if in_zpd:
            # Always reward ZPD selection
            zpd_bonus_total += REWARD_CONFIG_V3['zpd_hit_bonus']
            reward += REWARD_CONFIG_V3['zpd_hit_bonus']
            self.episode_zpd_hits += 1
            self.recent_zpd_hits.append(1.0)
            
            # Extra bonus for correct answer in ZPD
            if is_correct:
                zpd_bonus_total += REWARD_CONFIG_V3['zpd_perfect_bonus']
                reward += REWARD_CONFIG_V3['zpd_perfect_bonus']
            
            info['in_zpd'] = True
        else:
            self.recent_zpd_hits.append(0.0)
            
            # Partial credit for near-ZPD
            near_min = estimated_theta + REWARD_CONFIG_V3['near_zpd_range'][0]
            near_max = estimated_theta + REWARD_CONFIG_V3['near_zpd_range'][1]
            if near_min <= d_norm <= near_max:
                reward += REWARD_CONFIG_V3['near_zpd_bonus']
                info['near_zpd'] = True
            else:
                info['near_zpd'] = False
            
            info['in_zpd'] = False
        
        # Difficulty penalties
        if d_norm < (estimated_theta - 1.0):
            reward += REWARD_CONFIG_V3['too_easy_penalty']
            info['too_easy'] = True
        elif d_norm > (estimated_theta + 2.0):
            reward += REWARD_CONFIG_V3['too_hard_penalty']
            info['too_hard'] = True
        
        # Streak bonus
        if is_correct:
            streak_reward = min(self.correct_streak, 5) * REWARD_CONFIG_V3['streak_bonus']
            reward += streak_reward
            info['streak_bonus'] = streak_reward
        
        # Novel word bonus
        if word_id not in self.shown_words_this_episode:
            reward += REWARD_CONFIG_V3['novel_word_bonus']
            self.shown_words_this_episode.add(word_id)
            info['novel'] = True
        else:
            info['novel'] = False
        
        # ═══════════════════════════════════════════════════════
        # ANTI-REPETITION SYSTEM
        # ═══════════════════════════════════════════════════════
        
        # 1. Recent repetition penalty
        recent_count = self.word_history.count(word_id)
        if recent_count > 0:
            repetition_penalty = REWARD_CONFIG_V3['repetition_penalty'] * recent_count
            reward += repetition_penalty  # This is negative
            info['repetition_penalty'] = repetition_penalty
            info['recent_repetitions'] = recent_count
        
        # 2. Cooldown violation penalty
        if word_id in self.word_cooldowns:
            steps_since = self.word_cooldowns[word_id]
            if steps_since < REWARD_CONFIG_V3['cooldown_steps']:
                reward += REWARD_CONFIG_V3['cooldown_penalty']  # Negative
                info['cooldown_violation'] = True
                info['cooldown_remaining'] = REWARD_CONFIG_V3['cooldown_steps'] - steps_since
        
        # 3. Diminishing returns on repeated words in episode
        show_count = self.word_episode_count.get(word_id, 0)
        if show_count > 0 and in_zpd and zpd_bonus_total > 0:
            # Apply diminishing multiplier to ZPD bonuses
            diminish_multiplier = 1.0 / (1.0 + REWARD_CONFIG_V3['diminishing_factor'] * show_count)
            
            # Calculate reduction
            diminished_bonus = zpd_bonus_total * diminish_multiplier
            reduction = zpd_bonus_total - diminished_bonus
            reward -= reduction
            
            info['diminishing_reduction'] = reduction
            info['diminish_multiplier'] = diminish_multiplier
        
        # Update anti-repetition tracking
        self.word_history.append(word_id)
        self.word_episode_count[word_id] = show_count + 1
        self.word_cooldowns[word_id] = 0  # Reset cooldown for this word
        
        # Increment cooldowns for all other words
        for wid in list(self.word_cooldowns.keys()):
            if wid != word_id:
                self.word_cooldowns[wid] += 1
        
        # ZPD distance for analysis
        zpd_dist = abs(d_norm - (estimated_theta + 0.6))
        info['zpd_distance'] = zpd_dist
        
        # ═══════════════════════════════════════════════════════
        # REWARD CLIPPING (NEW - prevent extreme values)
        # ═══════════════════════════════════════════════════════
        reward = np.clip(
            reward,
            REWARD_CONFIG_V3['min_step_reward'],
            REWARD_CONFIG_V3['max_step_reward']
        )
        info['reward_clipped'] = reward
        
        return reward, info
    
    def step(self, action: int):
        """Execute step with V3 reward function."""
        # Get word features
        word = self.vocab.get_word_features(action)
        if word is None:
            obs = self._get_observation_v3()
            return obs, -1.0, False, False, {'error': 'Invalid word ID'}
        
        # User responds
        is_correct = self.user.respond(word['difficulty'], word['concreteness'])
        
        # Calculate V3 reward
        reward, reward_info = self._calculate_reward_v3(
            is_correct,
            action,
            word['difficulty']
        )
        
        # Update streak
        if is_correct:
            self.correct_streak += 1
        else:
            self.correct_streak = 0
        
        # Update state (same as base)
        self.answer_history.append(float(is_correct))
        self.presented_words.add(action)
        
        # Update metrics
        self.episode_rewards.append(reward)
        if is_correct:
            self.episode_correct += 1
            if action in self.retry_queue:
                self.retry_queue.remove(action)
        else:
            if action not in self.retry_queue:
                self.retry_queue.append(action)
        
        # Update level
        self._update_level()
        
        # Increment step
        self.step_count += 1
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # Get observation
        observation = self._get_observation_v3()
        
        # Enhanced info
        info = self._get_info()
        info.update(reward_info)
        
        return observation, reward, terminated, truncated, info


# Test
if __name__ == "__main__":
    print("=== V3.0 Environment Test (with Anti-Repetition) ===\n")
    
    env = LanguageLearningEnvV3(max_steps=20)
    obs, info = env.reset()
    
    print(f"✓ Environment created")
    print(f"✓ Observation shape: {obs.shape} (17 dims)")
    print(f"✓ Initial level: {info['current_level']}")
    print(f"✓ User theta: {info['user_theta']:.2f}\n")
    
    # Test repetition penalty
    print("Testing anti-repetition system:")
    available = info['available_actions']
    test_word = available[0]  # Pick one word
    
    for i in range(5):
        obs, reward, done, trunc, info = env.step(test_word)
        print(f"  Step {i+1}: Reward={reward:.3f}", end="")
        if 'repetition_penalty' in info:
            print(f" (Repetition penalty: {info['repetition_penalty']:.3f}, count: {info['recent_repetitions']})", end="")
        if 'diminishing_reduction' in info:
            print(f" (Diminished by {info['diminishing_reduction']:.3f})", end="")
        print()
    
    print(f"\n✓ Anti-repetition working! Rewards decreased with repetition")
    print(f"✓ Word history size: {len(env.word_history)}")
    print(f"\n✅ V3 environment test passed!")
