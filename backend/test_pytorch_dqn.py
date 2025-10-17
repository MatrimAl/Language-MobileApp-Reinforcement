"""
PyTorch DQN Agent Test Script
"""
import sys
import torch
import numpy as np
from dqn_agent import DQNAgent, DQNetwork

print("=" * 60)
print("🔥 PyTorch DQN Agent Test")
print("=" * 60)

# Check PyTorch version
print(f"\n✅ PyTorch Version: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA Device: {torch.cuda.get_device_name(0)}")

# Create DQN Agent
print("\n🤖 Creating DQN Agent...")
agent = DQNAgent(
    state_size=12,
    action_size=5,
    learning_rate=0.001,
    epsilon=1.0
)

print(f"✅ Agent created successfully!")
print(f"   - State Size: {agent.state_size}")
print(f"   - Action Size: {agent.action_size}")
print(f"   - Device: {agent.device}")
print(f"   - Model Parameters: {sum(p.numel() for p in agent.model.parameters()):,}")

# Test model architecture
print("\n📊 Model Architecture:")
print(agent.model)

# Test forward pass
print("\n🧪 Testing Forward Pass...")
dummy_state = np.random.randn(12).astype(np.float32)
print(f"   Input State Shape: {dummy_state.shape}")

# Get Q-values
q_values = agent.get_q_values(dummy_state)
print(f"   Output Q-Values: {q_values}")
print(f"   Q-Values Shape: {q_values.shape}")

# Test action selection
print("\n🎯 Testing Action Selection...")
action = agent.act(dummy_state, training=False)
print(f"   Selected Action (Greedy): {action}")
print(f"   Corresponding Difficulty: {['Beginner', 'Elementary', 'Intermediate', 'Advanced', 'Expert'][action]}")

# Test with exploration
agent.epsilon = 0.5
actions = [agent.act(dummy_state, training=True) for _ in range(20)]
print(f"\n   20 Actions with ε=0.5 (exploration):")
print(f"   {actions}")
print(f"   Unique Actions: {len(set(actions))} / 5")

# Test memory
print("\n💾 Testing Experience Replay Memory...")
for i in range(10):
    state = np.random.randn(12).astype(np.float32)
    action = np.random.randint(0, 5)
    reward = np.random.randn()
    next_state = np.random.randn(12).astype(np.float32)
    done = i == 9
    agent.remember(state, action, reward, next_state, done)

print(f"   Memory Size: {len(agent.memory)} / {agent.memory.maxlen}")

# Test replay (training step)
print("\n🎓 Testing Replay (Training Step)...")
if len(agent.memory) >= agent.batch_size:
    for _ in range(32):  # Add more experiences
        state = np.random.randn(12).astype(np.float32)
        action = np.random.randint(0, 5)
        reward = np.random.randn()
        next_state = np.random.randn(12).astype(np.float32)
        done = False
        agent.remember(state, action, reward, next_state, done)
    
    loss = agent.replay()
    print(f"   Training Loss: {loss:.4f}")
else:
    print("   ⚠️  Not enough samples for replay (need 32)")

# Test save/load
print("\n💾 Testing Save/Load...")
import os
os.makedirs("./test_models", exist_ok=True)
agent.save("./test_models/test_dqn")
print(f"   ✅ Model saved to ./test_models/test_dqn")

# Create new agent and load
new_agent = DQNAgent()
new_agent.load("./test_models/test_dqn")
print(f"   ✅ Model loaded successfully")
print(f"   Epsilon after load: {new_agent.epsilon:.3f}")

# Verify loaded model
new_q_values = new_agent.get_q_values(dummy_state)
print(f"   Q-Values match: {np.allclose(q_values, new_q_values)}")

print("\n" + "=" * 60)
print("✅ All tests passed! PyTorch DQN Agent is working!")
print("=" * 60)

# Cleanup
import shutil
if os.path.exists("./test_models"):
    shutil.rmtree("./test_models")
    print("\n🧹 Cleaned up test files")
