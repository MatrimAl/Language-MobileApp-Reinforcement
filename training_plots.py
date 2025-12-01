"""
Simple Training Visualizations

Creates detailed plots for training metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_metrics(
    episode_rewards,
    episode_accuracies,
    episode_zpd_hits,
    episode_lengths,
    save_dir='plots_final'
):
    """Generate all training plots."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('V3 Ultra Training Results - Anti-Repetition + Realistic Simulation', 
                 fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
    
    # Rolling average
    window = min(20, len(episode_rewards) // 5)
    if window > 1:
        rolling_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], rolling_avg, color='darkblue', linewidth=2, label=f'{window}-episode avg')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax = axes[0, 1]
    acc_percent = np.array(episode_accuracies) * 100
    ax.plot(episodes, acc_percent, alpha=0.3, color='green', label='Raw')
    
    if window > 1:
        rolling_avg = np.convolve(acc_percent, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], rolling_avg, color='darkgreen', linewidth=2, label=f'{window}-episode avg')
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Learning Accuracy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # 3. ZPD Hits
    ax = axes[1, 0]
    ax.plot(episodes, episode_zpd_hits, alpha=0.3, color='orange', label='Raw')
    
    if window > 1:
        rolling_avg = np.convolve(episode_zpd_hits, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], rolling_avg, color='darkorange', linewidth=2, label=f'{window}-episode avg')
    
    ax.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Target: 15 (30%)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('ZPD Hits per Episode', fontsize=12)
    ax.set_title('ZPD Targeting Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Episode Length
    ax = axes[1, 1]
    ax.plot(episodes, episode_lengths, alpha=0.3, color='purple', label='Raw')
    
    if window > 1:
        rolling_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], rolling_avg, color='indigo', linewidth=2, label=f'{window}-episode avg')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps per Episode', fontsize=12)
    ax.set_title('Episode Length', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_overview.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/training_overview.png")
    plt.close()
    
    # Additional detailed plots
    _plot_learning_curve(episode_rewards, episode_accuracies, save_dir)
    _plot_zpd_analysis(episode_zpd_hits, episode_accuracies, save_dir)
    _plot_distribution(episode_rewards, episode_accuracies, episode_zpd_hits, save_dir)
    
    print(f"\n📊 All plots saved to {save_dir}/")


def _plot_learning_curve(rewards, accuracies, save_dir):
    """Detailed learning curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cumulative reward
    cumulative = np.cumsum(rewards)
    episodes = np.arange(1, len(rewards) + 1)
    
    ax1.plot(episodes, cumulative, color='blue', linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Cumulative Reward', fontsize=12)
    ax1.set_title('Cumulative Learning Progress', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy trend
    acc_percent = np.array(accuracies) * 100
    ax2.scatter(episodes, acc_percent, alpha=0.3, s=20, color='green')
    
    # Fit polynomial trend
    if len(episodes) > 10:
        z = np.polyfit(episodes, acc_percent, 2)
        p = np.poly1d(z)
        ax2.plot(episodes, p(episodes), "r-", linewidth=2, label='Trend')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Trend', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curve.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/learning_curve.png")
    plt.close()


def _plot_zpd_analysis(zpd_hits, accuracies, save_dir):
    """ZPD targeting analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ZPD hit rate distribution
    zpd_rates = (np.array(zpd_hits) / 50) * 100  # Assuming 50 steps per episode
    
    ax1.hist(zpd_rates, bins=20, color='orange', edgecolor='black', alpha=0.7)
    ax1.axvline(x=zpd_rates.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {zpd_rates.mean():.1f}%')
    ax1.axvline(x=30, color='green', linestyle='--', linewidth=2, alpha=0.5,
                label='Target: 30%')
    ax1.set_xlabel('ZPD Hit Rate (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('ZPD Hit Rate Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ZPD vs Accuracy correlation
    acc_percent = np.array(accuracies) * 100
    ax2.scatter(zpd_rates, acc_percent, alpha=0.5, s=50, color='purple')
    
    # Correlation line
    if len(zpd_rates) > 2:
        z = np.polyfit(zpd_rates, acc_percent, 1)
        p = np.poly1d(z)
        x_line = np.linspace(zpd_rates.min(), zpd_rates.max(), 100)
        ax2.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('ZPD Hit Rate (%)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('ZPD Targeting vs Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/zpd_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/zpd_analysis.png")
    plt.close()


def _plot_distribution(rewards, accuracies, zpd_hits, save_dir):
    """Distribution plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Reward distribution
    axes[0].hist(rewards, bins=30, color='blue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=np.mean(rewards), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(rewards):.2f}')
    axes[0].set_xlabel('Episode Reward', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy distribution
    acc_percent = np.array(accuracies) * 100
    axes[1].hist(acc_percent, bins=30, color='green', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=np.mean(acc_percent), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(acc_percent):.1f}%')
    axes[1].set_xlabel('Accuracy (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # ZPD hits distribution
    axes[2].hist(zpd_hits, bins=30, color='orange', edgecolor='black', alpha=0.7)
    axes[2].axvline(x=np.mean(zpd_hits), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(zpd_hits):.1f}')
    axes[2].set_xlabel('ZPD Hits per Episode', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('ZPD Hits Distribution', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/distributions.png")
    plt.close()


if __name__ == "__main__":
    # Test with dummy data
    print("Testing visualization...")
    
    episodes = 100
    rewards = np.random.randn(episodes).cumsum() + np.arange(episodes) * 0.1
    accuracies = np.clip(0.5 + np.random.randn(episodes) * 0.1 + np.arange(episodes) * 0.001, 0.3, 0.8)
    zpd_hits = np.clip(20 + np.random.randn(episodes) * 5 + np.arange(episodes) * 0.05, 10, 35)
    lengths = np.full(episodes, 50)
    
    plot_training_metrics(rewards, accuracies, zpd_hits, lengths, 'test_plots')
    print("\n✅ Visualization test complete!")
