"""Test reward accumulation fix"""
from language_env_vectorized import VectorizedLanguageEnvV3
import numpy as np

print("Testing reward accumulation...")

env = VectorizedLanguageEnvV3(num_envs=2, max_steps=5)
states = env.reset()

actions = np.array([0, 1])

print("\nRunning 5 steps:")
for i in range(5):
    states, rewards, dones, infos = env.step(actions)
    print(f"Step {i+1}:")
    print(f"  Rewards: {rewards}")
    print(f"  Accumulated: {env.episode_rewards_sum}")
    print(f"  Dones: {dones}")
    
    if any(dones):
        for j, info in enumerate(infos):
            if 'episode' in info:
                print(f"  Env {j} finished! Total reward: {info['episode']['r']:.2f}")

print("\n✅ Test complete!")
print(f"Final accumulated sums: {env.episode_rewards_sum}")
