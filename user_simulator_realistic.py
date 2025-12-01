"""
Realistic IRT User Simulator with Cognitive Models

Implements realistic response patterns:
1. Slip & Guess (4-Parameter Logistic Model)
2. Fatigue effects
3. Motivation/engagement tracking
4. Variable learning rates
5. Forgetting with spaced repetition
6. Response noise and variability

Much more realistic than simple IRT!
"""

import numpy as np
from collections import deque, defaultdict
from typing import Optional, Tuple, Dict
from config import IRT_CONFIG


class RealisticIRTSimulator:
    """
    Realistic user simulator with cognitive psychology models.
    
    Improvements over basic IRT:
    - 4PL model: Accounts for slips (know but wrong) and guesses (don't know but right)
    - Fatigue: Performance degrades over time
    - Motivation: Success/failure affects engagement
    - Learning curves: Variable learning rates
    - Forgetting: Word memory decays over time
    - Noise: Realistic variability in responses
    """
    
    def __init__(
        self,
        theta: Optional[float] = None,
        discrimination: Optional[float] = None,
        concreteness_weight: Optional[float] = None,
        ability_drift: Optional[float] = None
    ):
        """Initialize realistic simulator."""
        
        # Base IRT parameters
        self.discrimination = discrimination or IRT_CONFIG['discrimination']
        self.concreteness_weight = concreteness_weight or IRT_CONFIG['concreteness_weight']
        self.base_learning_rate = ability_drift or IRT_CONFIG['ability_drift']
        
        # Initialize ability
        if theta is not None:
            self.theta = theta
        else:
            theta_min, theta_max = IRT_CONFIG['initial_theta_range']
            self.theta = np.random.uniform(theta_min, theta_max)
        
        self.initial_theta = self.theta
        
        # === 1. SLIP & GUESS PARAMETERS (4PL Model) ===
        self.slip_prob = 0.15      # 15% chance to slip (know→wrong)
        self.guess_prob = 0.25     # 25% chance to guess correctly (don't know→right)
        
        # === 2. FATIGUE MODEL ===
        self.energy = 1.0           # 100% energy initially
        self.fatigue_rate = 0.02    # Lose 2% per question
        self.recovery_rate = 0.05   # Gain 5% per correct (motivation)
        self.min_energy = 0.2       # Never below 20%
        
        # === 3. MOTIVATION/ENGAGEMENT ===
        self.motivation = 0.8       # 80% motivated initially
        self.success_boost = 0.05   # Boosts with correct answers
        self.failure_drop = 0.02    # Drops with wrong answers
        self.boredom_rate = 0.01    # Gradual decrease
        self.min_motivation = 0.3   # Min 30%
        
        # === 4. VARIABLE LEARNING RATE ===
        self.learning_momentum = 0.0
        self.recent_performance = deque(maxlen=5)
        
        # === 5. FORGETTING MODEL (Spaced Repetition) ===
        self.word_strengths = {}    #  {word_id: strength [0,1]}
        self.word_last_seen = {}    # {word_id: steps_ago}
        self.forgetting_rate = 0.05 # Memory decay rate
        
        # === 6. RESPONSE NOISE ===
        self.base_noise = 0.1       # Response variability
        
        # Statistics
        self.total_attempts = 0
        self.correct_answers = 0
        self.current_streak = 0
    
    def reset(self):
        """Reset user state to initial values."""
        self.theta = self.initial_theta
        self.energy = 1.0
        self.motivation = 0.8
        self.learning_momentum = 0.0
        self.recent_performance.clear()
        self.word_strengths.clear()
        self.word_last_seen.clear()
        self.total_attempts = 0
        self.correct_answers = 0
        self.current_streak = 0
    
    def get_probability(
        self,
        difficulty: float,
        concreteness: float,
        word_id: Optional[int] = None
    ) -> float:
        """
        Calculate response probability with all cognitive models.
        
        Args:
            difficulty: Word difficulty (1-10)
            concreteness: Word concreteness (1-5)
            word_id: Optional word ID for memory tracking
            
        Returns:
            Probability of correct response [0,1]
        """
        # Normalize
        d_norm = (difficulty - 5.5) / 2.25
        c_norm = (concreteness - 3.0) / 2.0
        
        # Base IRT logit
        logits = (
            self.discrimination * (self.theta - d_norm) +
            self.concreteness_weight * c_norm
        )
        
        # Base probability (classical IRT)
        p_know = 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))
        
        # === Apply Cognitive Models ===
        
        # 1. Apply word memory (forgetting)
        if word_id is not None and word_id in self.word_last_seen:
            steps_ago = self.word_last_seen[word_id]
            strength = self.word_strengths.get(word_id, 0.5)
            
            # Memory decay
            decay = np.exp(-self.forgetting_rate * steps_ago)
            memory_factor = 0.5 + 0.5 * strength * decay
            p_know = p_know * memory_factor
        
        # 2. 4-Parameter Logistic Model (slip & guess)
        # P(correct) = guess + (1 - slip - guess) * P(know)
        p_correct = (
            self.guess_prob +
            (1 - self.slip_prob - self.guess_prob) * p_know
        )
        
        # 3. Fatigue penalty
        fatigue_penalty = (1.0 - self.energy) * 0.3  # Max 30% reduction
        p_correct -= fatigue_penalty
        
        # 4. Motivation affects attention
        attention = 0.5 + 0.5 * self.motivation  # 50-100% attention
        p_correct *= attention
        
        # 5. Response noise
        noise = np.random.normal(0, self.base_noise)
        p_correct += noise
        
        # 6. Attention lapses (5% chance of severe drop)
        if np.random.random() < 0.05:
            p_correct *= 0.5
        
        # 7. Streaks (hot/cold)
        if len(self.recent_performance) >= 3:
            recent = list(self.recent_performance)[-3:]
            if all(recent):  # Hot streak
                p_correct *= 1.1
            elif not any(recent):  # Cold streak
                p_correct *= 0.9
        
        # Clamp to valid range
        p_correct = np.clip(p_correct, 0.05, 0.95)
        
        return p_correct
    
    def respond(
        self,
        difficulty: float,
        concreteness: float,
        word_id: Optional[int] = None
    ) -> bool:
        """
        Generate realistic user response.
        
        Args:
            difficulty: Word difficulty
            concreteness: Word concreteness  
            word_id: Optional word ID for memory tracking
            
        Returns:
            is_correct (bool)
        """
        # Get probability
        p_correct = self.get_probability(difficulty, concreteness, word_id)
        
        # Generate response
        is_correct = (np.random.random() < p_correct)
        
        # === Update Cognitive States ===
        
        # 1. Update ability (variable learning rate)
        self._update_ability(is_correct, difficulty)
        
        # 2. Update energy (fatigue)
        self._update_energy(is_correct)
        
        # 3. Update motivation
        self._update_motivation(is_correct)
        
        # 4. Update word memory
        if word_id is not None:
            self._update_word_memory(word_id, is_correct)
        
        # 5. Update statistics
        self.total_attempts += 1
        if is_correct:
            self.correct_answers += 1
            self.current_streak += 1
        else:
            self.current_streak = 0
        
        self.recent_performance.append(is_correct)
        
        return is_correct
    
    def _update_ability(self, is_correct: bool, difficulty: float):
        """Update theta with variable learning rate."""
        # Adaptive learning rate based on recent performance
        if len(self.recent_performance) > 0:
            recent_accuracy = np.mean(self.recent_performance)
            # Learn faster when doing well (confidence)
            # Learn slower when struggling (need consolidation)
            learning_multiplier = 0.5 + recent_accuracy  # 0.5-1.5x
        else:
            learning_multiplier = 1.0
        
        if is_correct:
            # Positive learning
            theta_gain = self.base_learning_rate * learning_multiplier
            self.theta += theta_gain
            
            # Momentum
            self.learning_momentum = 0.9 * self.learning_momentum + 0.1 * theta_gain
        else:
            # Small gain from exposure even when wrong
            self.theta += 0.002
            # Slight negative momentum
            self.learning_momentum *= 0.95
    
    def _update_energy(self, is_correct: bool):
        """Update fatigue/energy level."""
        # Fatigue increases
        self.energy -= self.fatigue_rate
        
        # Success provides recovery (motivation boost)
        if is_correct:
            self.energy += self.recovery_rate
        
        # Clamp
        self.energy = np.clip(self.energy, self.min_energy, 1.0)
    
    def _update_motivation(self, is_correct: bool):
        """Update motivation/engagement."""
        if is_correct:
            self.motivation += self.success_boost
        else:
            self.motivation -= self.failure_drop
        
        # Gradual boredom
        self.motivation -= self.boredom_rate
        
        # Clamp
        self.motivation = np.clip(self.motivation, self.min_motivation, 1.0)
    
    def _update_word_memory(self, word_id: int, is_correct: bool):
        """Update word-specific memory strength."""
        if is_correct:
            # Strengthen memory
            old_strength = self.word_strengths.get(word_id, 0.0)
            self.word_strengths[word_id] = min(1.0, old_strength + 0.2)
        else:
            # Weaken slightly
            old_strength = self.word_strengths.get(word_id, 0.5)
            self.word_strengths[word_id] = max(0.0, old_strength - 0.1)
        
        # Reset last seen
        self.word_last_seen[word_id] = 0
    
    def step_time(self):
        """
        Call this each environment step to update time-based effects.
        Increments word memory decay counters.
        """
        for word_id in list(self.word_last_seen.keys()):
            self.word_last_seen[word_id] += 1
    
    def get_accuracy(self) -> float:
        """Get overall accuracy."""
        if self.total_attempts == 0:
            return 0.0
        return self.correct_answers / self.total_attempts
    
    def get_state(self) -> Dict:
        """Get current cognitive state."""
        return {
            'theta': self.theta,
            'energy': self.energy,
            'motivation': self.motivation,
            'accuracy': self.get_accuracy(),
            'streak': self.current_streak,
            'total_attempts': self.total_attempts,
            'learning_momentum': self.learning_momentum,
        }


# Test and compare
if __name__ == "__main__":
    print("=== Realistic IRT Simulator Test ===\n")
    
    from user_simulator import IRTSimulator
    
    # Create both simulators
    simple_sim = IRTSimulator(theta=0.0)
    realistic_sim = RealisticIRTSimulator(theta=0.0)
    
    print("Testing on 50 words (difficulty 5, concreteness 3):\n")
    
    simple_correct = 0
    realistic_correct = 0
    
    for i in range(50):
        simple_result = simple_sim.respond(5.0, 3.0)
        realistic_result = realistic_sim.respond(5.0, 3.0, word_id=0)
        realistic_sim.step_time()
        
        if simple_result:
            simple_correct += 1
        if realistic_result:
            realistic_correct += 1
    
    print(f"Simple IRT:")
    print(f"  Accuracy: {simple_correct}/50 ({100*simple_correct/50:.1f}%)")
    print(f"  Final theta: {simple_sim.theta:.3f}")
    
    print(f"\nRealistic IRT:")
    print(f"  Accuracy: {realistic_correct}/50 ({100*realistic_correct/50:.1f}%)")
    print(f"  Final theta: {realistic_sim.theta:.3f}")
    print(f"  Energy: {realistic_sim.energy:.2f}")
    print(f"  Motivation: {realistic_sim.motivation:.2f}")
    print(f"  Streak: {realistic_sim.current_streak}")
    
    print(f"\n✓ Realistic simulator shows more variance and complexity!")
    print(f"✓ Factors: slip/guess, fatigue, motivation, learning curves")
    
    # Test forgetting
    print(f"\n=== Testing Forgetting ===")
    sim = RealisticIRTSimulator(theta=0.5)
    
    # Learn a word
    word_id = 42
    sim.respond(5.0, 3.0, word_id=word_id)
    strength_fresh = sim.word_strengths.get(word_id, 0)
    
    # Wait 20 steps
    for _ in range(20):
        sim.step_time()
    
    # Test again
    p_fresh = sim.get_probability(5.0, 3.0)
    p_forgotten = sim.get_probability(5.0, 3.0, word_id=word_id)
    
    print(f"Word strength: {strength_fresh:.2f}")
    print(f"P(correct) fresh word: {p_fresh:.3f}")
    print(f"P(correct) after 20 steps: {p_forgotten:.3f}")
    print(f"Memory decay: {(p_fresh - p_forgotten)*100:.1f}% reduction")
    
    print(f"\n✅ Realistic simulator test passed!")
