import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

class LanguageLearningEnv(gym.Env):
    """
    Dil öğrenme için custom Gymnasium environment
    
    State Space:
        - Kullanıcı seviyesi (1-100)
        - Toplam öğrenilen kelime sayısı
        - Ortalama doğruluk oranı
        - Son 10 sorunun doğruluk oranı
        - Mevcut streak (ardışık doğru cevap)
        - Son oturumdan beri geçen zaman
        - Zorluk seviyesi dağılımı (5 kategori)
    
    Action Space:
        - Hangi kelime gösterilecek (kelime indeksi)
        - Pratik amaçlı olarak difficulty-based action space kullanıyoruz
    
    Reward:
        - Doğru cevap: +1.0
        - Yanlış cevap: -0.5
        - Hız bonusu: Hızlı doğru cevap +0.2
        - Retention bonusu: Eski kelimeyi hatırlama +0.3
        - Zorluk bonusu: Zor kelimeyi doğru yapma +0.5
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, word_pool: List[Dict], user_data: Dict = None):
        super(LanguageLearningEnv, self).__init__()
        
        self.word_pool = word_pool
        self.num_words = len(word_pool)
        
        # State space: 12 features
        # [level, total_learned, avg_acc, recent_acc, streak, 
        #  time_hours, mastery_0, mastery_1, mastery_2, mastery_3, mastery_4]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 10000, 1, 1, 100, 168, 1, 1, 1, 1, 1, 5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 5 zorluk seviyesi (1-5)
        self.action_space = spaces.Discrete(5)
        
        # User state initialization
        self.user_data = user_data or {
            'level': 1,
            'total_learned': 0,
            'history': [],
            'word_progress': {},
            'last_session': datetime.now()
        }
        
        self.current_state = self._get_state()
        self.episode_rewards = []
        self.episode_length = 0
        
    def _get_state(self) -> np.ndarray:
        """Mevcut kullanıcı durumunu state vector'e çevir"""
        history = self.user_data['history']
        
        # Calculate metrics
        level = self.user_data['level']
        total_learned = len(self.user_data['word_progress'])
        
        # Average accuracy (all time)
        if len(history) > 0:
            avg_acc = sum(1 for h in history if h['correct']) / len(history)
        else:
            avg_acc = 0.0
        
        # Recent accuracy (last 10)
        recent_history = history[-10:] if len(history) >= 10 else history
        if len(recent_history) > 0:
            recent_acc = sum(1 for h in recent_history if h['correct']) / len(recent_history)
        else:
            recent_acc = 0.0
        
        # Current streak
        streak = 0
        for h in reversed(history):
            if h['correct']:
                streak += 1
            else:
                break
        
        # Time since last session (hours)
        time_diff = (datetime.now() - self.user_data['last_session']).total_seconds() / 3600
        time_diff = min(time_diff, 168)  # Max 1 hafta
        
        # Mastery distribution
        mastery_dist = [0, 0, 0, 0, 0]  # [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
        for word_id, progress in self.user_data['word_progress'].items():
            mastery = progress.get('mastery_level', 0)
            idx = min(int(mastery * 5), 4)
            mastery_dist[idx] += 1
        
        # Normalize mastery distribution
        total_words = sum(mastery_dist) or 1
        mastery_dist = [m / total_words for m in mastery_dist]
        
        # Average difficulty of mastered words
        avg_difficulty = 0
        if self.user_data['word_progress']:
            difficulties = []
            for word_id, progress in self.user_data['word_progress'].items():
                if progress.get('mastery_level', 0) > 0.5:
                    word = next((w for w in self.word_pool if w['id'] == word_id), None)
                    if word:
                        difficulties.append(word['difficulty'])
            avg_difficulty = np.mean(difficulties) if difficulties else 0
        
        state = np.array([
            level,
            total_learned,
            avg_acc,
            recent_acc,
            streak,
            time_diff,
            *mastery_dist,
            avg_difficulty
        ], dtype=np.float32)
        
        return state
    
    def _select_word_by_difficulty(self, difficulty: int) -> Dict:
        """Belirtilen zorluk seviyesinden kelime seç (spaced repetition ile)"""
        # Filter words by difficulty
        difficulty_words = [w for w in self.word_pool if w['difficulty'] == difficulty + 1]
        
        if not difficulty_words:
            # Fallback: random word
            return random.choice(self.word_pool)
        
        # Spaced repetition: öncelikle review edilmesi gereken kelimeler
        words_need_review = []
        for word in difficulty_words:
            progress = self.user_data['word_progress'].get(word['id'], {})
            next_review = progress.get('next_review')
            
            if next_review is None or next_review <= datetime.now():
                words_need_review.append(word)
        
        if words_need_review:
            # Mastery'si düşük olanları önceliklendir
            words_need_review.sort(key=lambda w: self.user_data['word_progress'].get(w['id'], {}).get('mastery_level', 0))
            return words_need_review[0]
        else:
            # Yeni kelime veya rastgele
            new_words = [w for w in difficulty_words if w['id'] not in self.user_data['word_progress']]
            if new_words:
                return random.choice(new_words)
            else:
                return random.choice(difficulty_words)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Bir adım at
        action: 0-4 arası zorluk seviyesi
        """
        # Action'dan kelime seç
        selected_word = self._select_word_by_difficulty(action)
        
        # Simüle edilmiş cevap (gerçek uygulamada kullanıcıdan gelecek)
        # Şimdilik state'e göre probabilistic cevap
        user_skill = self.current_state[2]  # avg accuracy
        word_difficulty = selected_word['difficulty']
        
        # Success probability
        success_prob = max(0.1, user_skill - (word_difficulty - 1) * 0.1)
        is_correct = random.random() < success_prob
        
        # Response time simulation (2-10 saniye)
        response_time = random.uniform(2, 10)
        
        # Calculate reward
        reward = self._calculate_reward(selected_word, is_correct, response_time)
        
        # Update user data
        self._update_user_progress(selected_word, is_correct, response_time)
        
        # Update state
        self.current_state = self._get_state()
        
        # Episode termination (20 kelime sonra)
        self.episode_length += 1
        terminated = self.episode_length >= 20
        truncated = False
        
        self.episode_rewards.append(reward)
        
        info = {
            'word': selected_word['word'],
            'difficulty': selected_word['difficulty'],
            'is_correct': is_correct,
            'response_time': response_time,
            'episode_reward': sum(self.episode_rewards)
        }
        
        return self.current_state, reward, terminated, truncated, info
    
    def _calculate_reward(self, word: Dict, is_correct: bool, response_time: float) -> float:
        """Reward hesapla"""
        base_reward = 1.0 if is_correct else -0.5
        
        # Speed bonus (< 5 saniye)
        speed_bonus = 0.2 if (is_correct and response_time < 5) else 0
        
        # Difficulty bonus
        difficulty_bonus = 0.1 * word['difficulty'] if is_correct else 0
        
        # Retention bonus (eski kelimeyi doğru hatırlama)
        progress = self.user_data['word_progress'].get(word['id'], {})
        last_seen = progress.get('last_seen')
        retention_bonus = 0
        if last_seen and is_correct:
            days_since = (datetime.now() - last_seen).days
            if days_since > 7:
                retention_bonus = 0.3
            elif days_since > 3:
                retention_bonus = 0.2
        
        total_reward = base_reward + speed_bonus + difficulty_bonus + retention_bonus
        
        return total_reward
    
    def _update_user_progress(self, word: Dict, is_correct: bool, response_time: float):
        """Kullanıcı ilerlemesini güncelle"""
        word_id = word['id']
        
        if word_id not in self.user_data['word_progress']:
            self.user_data['word_progress'][word_id] = {
                'correct_count': 0,
                'incorrect_count': 0,
                'mastery_level': 0.0,
                'last_seen': None,
                'next_review': None
            }
        
        progress = self.user_data['word_progress'][word_id]
        
        if is_correct:
            progress['correct_count'] += 1
        else:
            progress['incorrect_count'] += 1
        
        # Update mastery level (0-1)
        total_attempts = progress['correct_count'] + progress['incorrect_count']
        progress['mastery_level'] = progress['correct_count'] / total_attempts
        
        # Spaced repetition: next review time
        if is_correct:
            interval_days = min(30, 2 ** progress['correct_count'])
        else:
            interval_days = 1
        
        progress['last_seen'] = datetime.now()
        progress['next_review'] = datetime.now() + timedelta(days=interval_days)
        
        # Add to history
        self.user_data['history'].append({
            'word_id': word_id,
            'correct': is_correct,
            'response_time': response_time,
            'timestamp': datetime.now()
        })
        
        # Update level (her 10 kelime öğrenildiğinde)
        mastered_words = sum(1 for p in self.user_data['word_progress'].values() if p['mastery_level'] > 0.7)
        self.user_data['level'] = 1 + (mastered_words // 10)
        self.user_data['last_session'] = datetime.now()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Episode'u sıfırla"""
        super().reset(seed=seed)
        
        self.episode_rewards = []
        self.episode_length = 0
        self.current_state = self._get_state()
        
        return self.current_state, {}
    
    def render(self, mode='human'):
        """Environment'i görselleştir"""
        print(f"Level: {self.user_data['level']}")
        print(f"Total Words: {len(self.user_data['word_progress'])}")
        print(f"Episode Length: {self.episode_length}")
        print(f"Episode Reward: {sum(self.episode_rewards):.2f}")
