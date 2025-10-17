from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
from dqn_agent import DQNAgent
from rl_environment import LanguageLearningEnv
import os

router = APIRouter()

# Global agent instance (production'da Redis/DB'den yükle)
global_agent = None
training_status = {"is_training": False, "progress": 0, "episode": 0}

class TrainRequest(BaseModel):
    episodes: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001

class PredictRequest(BaseModel):
    state: List[float]

@router.post("/train")
async def train_agent(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    DQN agent'i arka planda eğit
    """
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Background task olarak eğitimi başlat
    background_tasks.add_task(run_training, request.episodes, request.batch_size, request.learning_rate)
    
    return {
        "message": "Training started",
        "episodes": request.episodes,
        "status": "running"
    }

def run_training(episodes: int, batch_size: int, learning_rate: float):
    """Arka planda training çalıştır"""
    global global_agent, training_status
    
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["episode"] = 0
    
    try:
        # Sample word pool (gerçek uygulamada DB'den çek)
        word_pool = generate_sample_words()
        
        # Create environment
        env = LanguageLearningEnv(word_pool=word_pool)
        
        # Create agent
        agent = DQNAgent(
            state_size=12,
            action_size=5,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Train
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.act(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()
                
                state = next_state
                episode_reward += reward
            
            # Update status
            training_status["episode"] = episode + 1
            training_status["progress"] = int((episode + 1) / episodes * 100)
            
            # Update target network
            if (episode + 1) % 10 == 0:
                agent.update_target_model()
        
        # Save model
        os.makedirs("./models", exist_ok=True)
        agent.save("./models/dqn_model")
        
        global_agent = agent
        
        training_status["is_training"] = False
        training_status["progress"] = 100
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        raise

@router.get("/training/status")
async def get_training_status():
    """Training durumunu getir"""
    return training_status

@router.post("/predict")
async def predict_action(request: PredictRequest):
    """
    Verilen state için en iyi action'ı tahmin et
    """
    global global_agent
    
    if global_agent is None:
        # Try to load saved model
        if os.path.exists("./models/dqn_model_model.pth"):
            global_agent = DQNAgent()
            global_agent.load("./models/dqn_model")
        else:
            raise HTTPException(status_code=404, detail="Model not trained yet")
    
    # Convert state to numpy array with proper shape
    if isinstance(request.state, (int, float)):
        # If single value provided, create default state
        state = np.array([0.5, 10, 0.8, 0.85, 0.75, 5, 2.5, 0.6, 0.2, 0.15, 0.05, 3], dtype=np.float32)
    else:
        state = np.array(request.state, dtype=np.float32)
    
    # Ensure state has correct shape (12 features)
    if state.shape[0] != 12:
        raise HTTPException(
            status_code=400, 
            detail=f"State must have 12 features, got {state.shape[0]}"
        )
    
    # Get action and explanation
    explanation = global_agent.get_action_explanation(state)
    
    return explanation

@router.get("/model/info")
async def get_model_info():
    """Model bilgilerini getir"""
    global global_agent
    
    if global_agent is None:
        return {"status": "not_loaded", "model_exists": os.path.exists("./models/dqn_model_model.pth")}
    
    return {
        "status": "loaded",
        "epsilon": global_agent.epsilon,
        "memory_size": len(global_agent.memory),
        "training_episodes": len(global_agent.training_history.get("episode_rewards", []))
    }

@router.get("/model/metrics")
async def get_model_metrics():
    """Model eğitim metriklerini getir"""
    global global_agent
    
    if global_agent is None or not global_agent.training_history:
        raise HTTPException(status_code=404, detail="No training history available")
    
    history = global_agent.training_history
    
    # Calculate statistics
    recent_rewards = history["episode_rewards"][-100:]
    
    return {
        "total_episodes": len(history["episode_rewards"]),
        "avg_reward": float(np.mean(history["episode_rewards"])) if history["episode_rewards"] else 0,
        "recent_avg_reward": float(np.mean(recent_rewards)) if recent_rewards else 0,
        "max_reward": float(np.max(history["episode_rewards"])) if history["episode_rewards"] else 0,
        "current_epsilon": float(history["epsilon_values"][-1]) if history["epsilon_values"] else 0,
        "episode_rewards": history["episode_rewards"][-50:],  # Son 50 episode
        "epsilon_values": history["epsilon_values"][-50:]
    }

def generate_sample_words():
    """Sample word pool oluştur (test için)"""
    words = [
        # Beginner (1)
        {"id": "1", "word": "hello", "translation": "merhaba", "difficulty": 1, "language": "en"},
        {"id": "2", "word": "goodbye", "translation": "hoşçakal", "difficulty": 1, "language": "en"},
        {"id": "3", "word": "thank you", "translation": "teşekkür ederim", "difficulty": 1, "language": "en"},
        {"id": "4", "word": "yes", "translation": "evet", "difficulty": 1, "language": "en"},
        {"id": "5", "word": "no", "translation": "hayır", "difficulty": 1, "language": "en"},
        
        # Elementary (2)
        {"id": "6", "word": "book", "translation": "kitap", "difficulty": 2, "language": "en"},
        {"id": "7", "word": "table", "translation": "masa", "difficulty": 2, "language": "en"},
        {"id": "8", "word": "computer", "translation": "bilgisayar", "difficulty": 2, "language": "en"},
        {"id": "9", "word": "water", "translation": "su", "difficulty": 2, "language": "en"},
        {"id": "10", "word": "food", "translation": "yemek", "difficulty": 2, "language": "en"},
        
        # Intermediate (3)
        {"id": "11", "word": "environment", "translation": "çevre", "difficulty": 3, "language": "en"},
        {"id": "12", "word": "development", "translation": "gelişme", "difficulty": 3, "language": "en"},
        {"id": "13", "word": "opportunity", "translation": "fırsat", "difficulty": 3, "language": "en"},
        {"id": "14", "word": "relationship", "translation": "ilişki", "difficulty": 3, "language": "en"},
        {"id": "15", "word": "experience", "translation": "deneyim", "difficulty": 3, "language": "en"},
        
        # Advanced (4)
        {"id": "16", "word": "sophisticated", "translation": "karmaşık", "difficulty": 4, "language": "en"},
        {"id": "17", "word": "contemporary", "translation": "çağdaş", "difficulty": 4, "language": "en"},
        {"id": "18", "word": "fundamental", "translation": "temel", "difficulty": 4, "language": "en"},
        {"id": "19", "word": "substantial", "translation": "önemli", "difficulty": 4, "language": "en"},
        {"id": "20", "word": "comprehensive", "translation": "kapsamlı", "difficulty": 4, "language": "en"},
        
        # Expert (5)
        {"id": "21", "word": "juxtaposition", "translation": "yan yana koyma", "difficulty": 5, "language": "en"},
        {"id": "22", "word": "paradigm", "translation": "paradigma", "difficulty": 5, "language": "en"},
        {"id": "23", "word": "ubiquitous", "translation": "her yerde bulunan", "difficulty": 5, "language": "en"},
        {"id": "24", "word": "ephemeral", "translation": "geçici", "difficulty": 5, "language": "en"},
        {"id": "25", "word": "quintessential", "translation": "özünde", "difficulty": 5, "language": "en"},
    ]
    
    return words

@router.post("/initialize")
async def initialize_sample_data():
    """Sample data ile hızlı başlatma (development için)"""
    global global_agent
    
    word_pool = generate_sample_words()
    
    # Create environment
    env = LanguageLearningEnv(word_pool=word_pool)
    
    # Create agent (küçük model)
    global_agent = DQNAgent(
        state_size=12,
        action_size=5,
        batch_size=16,
        learning_rate=0.001
    )
    
    # Quick training (50 episodes)
    for episode in range(50):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = global_agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            global_agent.remember(state, action, reward, next_state, done)
            global_agent.replay()
            
            state = next_state
        
        if (episode + 1) % 10 == 0:
            global_agent.update_target_model()
    
    # Save
    os.makedirs("./models", exist_ok=True)
    global_agent.save("./models/dqn_model")
    
    return {
        "message": "Sample model initialized and trained",
        "episodes": 50,
        "words": len(word_pool)
    }
