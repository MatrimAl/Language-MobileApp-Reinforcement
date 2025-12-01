"""
Configuration file for RL-based Adaptive Language Learning System
"""

# Environment Settings
ENV_CONFIG = {
    'promotion_threshold': 0.70,
    'demotion_threshold': 0.40,
    'rolling_window': 10,
    'max_level': 10,
    'min_level': 1,
    'retry_queue_max': 50,
}

# IRT Model Parameters
IRT_CONFIG = {
    'discrimination': 1.5,
    'concreteness_weight': 0.3,
    'ability_drift': 0.01,
    'initial_theta_range': (-2.0, 2.0),
}

# Reward Function Parameters
REWARD_CONFIG = {
    'correct_base': 0.1,
    'incorrect_penalty': -0.1,
    'zpd_bonus': 0.5,
    'zpd_range': (0.0, 1.2),
    'easy_penalty': -0.02,
    'easy_threshold': -0.5,
}

# DQN Hyperparameters
DQN_CONFIG = {
    'backbone_hidden': [256, 256, 128],
    'learning_rate': 1e-5,
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.995,
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'target_update_freq': 500,
    'dropout': 0.2,
}

# Training Parameters
TRAIN_CONFIG = {
    'num_episodes': 1000,
    'max_steps_per_episode': 50,
    'num_simulated_users': 100,
    'save_freq': 100,
    'log_freq': 10,
}

# Data Parameters
DATA_CONFIG = {
    'vocab_file': 'turkish_english_vocab_from_xlsx.csv',
    'concreteness_seed': 42,
}

# GPU Training Parameters
GPU_TRAIN_CONFIG = {
    'num_parallel_envs': 8,
    'use_pinned_memory': True,
    'async_env_reset': False,
}

# CEFR to Difficulty Mapping
CEFR_TO_DIFFICULTY = {
    'A1': 1,
    'A2': 3,
    'B1': 5,
    'B2': 7,
    'C1': 9,
    'C2': 10,
}

# V3 Configuration - REBALANCED REWARDS
REWARD_CONFIG_V3 = {
    # Increased positive rewards
    'correct_base': 0.15,            # Was 0.1 (+50%)
    'incorrect_penalty': -0.03,
    'zpd_perfect_bonus': 1.5,
    'zpd_hit_bonus': 1.0,            # Was 0.8 (+25%)
    'near_zpd_bonus': 0.4,
    'zpd_range': (-0.3, 1.5),
    'near_zpd_range': (-0.8, 2.0),
    'too_easy_penalty': -0.03,
    'too_hard_penalty': -0.03,
    'streak_bonus': 0.08,
    'novel_word_bonus': 0.15,
    
    # Anti-repetition (REBALANCED - less aggressive)
    'repetition_penalty': -0.15,     # Was -0.3 (50% reduction)
    'repetition_window': 10,
    'cooldown_steps': 5,
    'cooldown_penalty': -0.2,        # Was -0.5 (60% reduction)
    'diminishing_factor': 0.3,       # Was 0.5 (less aggressive)
    
    # Reward clipping (NEW)
    'min_step_reward': -0.5,         # Prevent huge negatives
    'max_step_reward': 3.0,          # Cap maximum
}

DQN_CONFIG_V3 = {
    'backbone_hidden': [256, 256, 128],
    'learning_rate': 2e-4,
    'gamma': 0.95,
    'batch_size': 256,
    'replay_buffer_size': 200000,
    'target_update_tau': 0.01,
    'gradient_clip': 1.0,
    'weight_decay': 1e-5,
    'dropout': 0.1,
    'use_noisy_nets': True,
    'use_dueling': True,
    'use_double_dqn': True,
    'use_prioritized_replay': True,
    'noisy_std_init': 0.5,
    'per_alpha': 0.7,
    'per_beta_start': 0.4,
    'per_beta_frames': 100000,
    'per_epsilon': 1e-6,
}