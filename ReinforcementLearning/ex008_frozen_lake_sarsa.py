import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.9995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with small random values instead of zeros
        self.q_table = np.random.uniform(-0.1, 0.1, (env.observation_space.n, env.action_space.n))
        
        # Learning statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.evaluation_rewards = []
        
    def epsilon_greedy_policy(self, state):
        """epsilon-greedy policy for action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # explore
        else:
            return np.argmax(self.q_table[state])  # exploit
    
    def greedy_policy(self, state):
        """Greedy policy without exploration (for evaluation)"""
        return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state, next_action, done):
        """SARSA update rule"""
        current_q = self.q_table[state, action]
        
        if done:
            # Terminal state: next Q-value is 0
            next_q = 0
        else:
            next_q = self.q_table[next_state, next_action]
        
        # SARSA update: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state, action] = current_q + self.alpha * td_error
    
    def train_episode(self):
        """Train for one episode"""
        state, _ = self.env.reset()
        action = self.epsilon_greedy_policy(state)
        
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            # Execute action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Select next action (key part of SARSA)
            next_action = self.epsilon_greedy_policy(next_state)
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_reward, steps
    
    def evaluate(self, n_episodes=100):
        """Evaluate current policy"""
        total_rewards = 0
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 100
            
            while steps < max_steps:
                action = self.greedy_policy(state)  # No exploration, just greedy
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            total_rewards += total_reward
        
        return total_rewards / n_episodes
    
    def train(self, n_episodes=100000, eval_interval=5000):
        """Full training process"""
        print(f"Starting SARSA training - {n_episodes} episodes")
        print(f"Initial parameters: alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}")
        
        for episode in tqdm(range(n_episodes)):
            reward, length = self.train_episode()
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            # Periodic evaluation
            if episode % eval_interval == 0 and episode > 0:
                avg_reward = self.evaluate(n_episodes=100)
                self.evaluation_rewards.append(avg_reward)
                print(f"Episode {episode}: Average reward = {avg_reward:.4f}, epsilon = {self.epsilon:.4f}")
        
        print(f"Training completed! Final epsilon: {self.epsilon:.4f}")
        
    def plot_results(self):
        """Visualize results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Episode rewards (moving average)
        window = 1000
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(moving_avg)
            axes[0, 0].set_title(f'Episode Rewards (Moving Average, window={window})')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Evaluation rewards
        if self.evaluation_rewards:
            axes[0, 1].plot(self.evaluation_rewards)
            axes[0, 1].set_title('Evaluation Rewards (100 episodes average)')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-table heatmap
        sns.heatmap(self.q_table, annot=True, fmt='.2f', cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('Final Q-table')
        axes[1, 0].set_xlabel('Action')
        axes[1, 0].set_ylabel('State')
        
        # 4. Optimal policy visualization
        optimal_policy = np.argmax(self.q_table, axis=1)
        policy_map = optimal_policy.reshape(4, 4)
        
        # Frozen Lake environment map
        lake_map = [
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ]
        
        action_symbols = ['←', '↓', '→', '↑']  # LEFT, DOWN, RIGHT, UP
        
        axes[1, 1].text(0.5, 0.5, 'Frozen Lake Optimal Policy', ha='center', va='center', 
                        transform=axes[1, 1].transAxes, fontsize=16, fontweight='bold')
        
        # Policy visualization
        for i in range(4):
            for j in range(4):
                if lake_map[i][j] != 'H':  # Show policy only for non-hole states
                    action = policy_map[i, j]
                    axes[1, 1].text(j + 0.5, 3 - i + 0.5, action_symbols[action], 
                                   ha='center', va='center', fontsize=20)
        
        # Draw grid
        for i in range(5):
            axes[1, 1].axhline(y=i, color='black', linewidth=0.5)
            axes[1, 1].axvline(x=i, color='black', linewidth=0.5)
        
        axes[1, 1].set_xlim(0, 4)
        axes[1, 1].set_ylim(0, 4)
        axes[1, 1].set_title('Optimal Policy (S:Start, G:Goal, H:Hole)')
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('/home/jso1/local/teaching/RL/sarsa_frozen_lake_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Create environment (slippery environment for more challenging learning)
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # Create SARSA agent with optimized parameters for slippery environment
    agent = SARSAAgent(
        env=env,
        alpha=0.05,     # smaller learning rate for more stable learning
        gamma=0.99,     # discount factor
        epsilon=0.4,    # higher initial exploration rate
        epsilon_decay=0.9999,  # slower epsilon decay
        epsilon_min=0.01       # minimum epsilon
    )
    
    # Training with more episodes for slippery environment
    agent.train(n_episodes=100000, eval_interval=5000)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_performance = agent.evaluate(n_episodes=1000)
    print(f"Final average reward (1000 episodes): {final_performance:.4f}")
    
    # Visualize results
    agent.plot_results()
    
    # Print final Q-table and policy
    print("\n=== Final Q-table ===")
    print(agent.q_table.round(3))
    
    print("\n=== Optimal Policy ===")
    optimal_policy = np.argmax(agent.q_table, axis=1)
    action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    for state, action in enumerate(optimal_policy):
        print(f"State {state}: {action_names[action]}")

if __name__ == "__main__":
    main()
