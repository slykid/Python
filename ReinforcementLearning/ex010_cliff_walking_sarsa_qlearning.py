import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

class CliffWalkingAgent:
    def __init__(self, env, algorithm='sarsa', alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.01):
        self.env = env
        self.algorithm = algorithm
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        
        # Learning statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def epsilon_greedy_policy(self, state):
        """epsilon-greedy policy for action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # explore
        else:
            return np.argmax(self.q_table[state])  # exploit
    
    def greedy_policy(self, state):
        """Greedy policy without exploration (for evaluation)"""
        return np.argmax(self.q_table[state])
    
    def update_q_value_sarsa(self, state, action, reward, next_state, next_action, done):
        """SARSA update rule"""
        current_q = self.q_table[state, action]
        
        if done:
            next_q = 0
        else:
            next_q = self.q_table[next_state, next_action]
        
        # SARSA update: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state, action] = current_q + self.alpha * td_error
    
    def update_q_value_qlearning(self, state, action, reward, next_state, done):
        """Q-learning update rule"""
        current_q = self.q_table[state, action]
        
        if done:
            next_q = 0
        else:
            next_q = np.max(self.q_table[next_state])
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max_a'Q(s',a') - Q(s,a)]
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state, action] = current_q + self.alpha * td_error
    
    def train_episode(self):
        """Train for one episode"""
        state, _ = self.env.reset()
        action = self.epsilon_greedy_policy(state)
        
        total_reward = 0
        steps = 0
        max_steps = 1000
        
        while steps < max_steps:
            # Execute action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            if self.algorithm == 'sarsa':
                # Select next action for SARSA
                next_action = self.epsilon_greedy_policy(next_state)
                # Update Q-value using SARSA
                self.update_q_value_sarsa(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
            else:  # Q-learning
                # Update Q-value using Q-learning
                self.update_q_value_qlearning(state, action, reward, next_state, done)
                state = next_state
                # Select action for next step
                action = self.epsilon_greedy_policy(next_state)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_reward, steps
    
    def train(self, n_episodes=5000):
        """Full training process"""
        print(f"Starting {self.algorithm.upper()} training - {n_episodes} episodes")
        
        for episode in tqdm(range(n_episodes)):
            reward, length = self.train_episode()
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
        
        print(f"Training completed! Final epsilon: {self.epsilon:.4f}")
    
    def get_optimal_path(self):
        """Get the optimal path using greedy policy"""
        state, _ = self.env.reset()
        path = [state]
        total_reward = 0
        
        for _ in range(100):  # Max steps
            action = self.greedy_policy(state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            path.append(state)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return path, total_reward
    
    def plot_results(self):
        """Visualize results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Episode rewards (moving average)
        window = 100
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(moving_avg)
            axes[0, 0].set_title(f'{self.algorithm.upper()} - Episode Rewards (Moving Average)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Episode lengths (moving average)
        if len(self.episode_lengths) >= window:
            moving_avg = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'{self.algorithm.upper()} - Episode Lengths (Moving Average)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-table heatmap
        sns.heatmap(self.q_table, annot=False, cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title(f'{self.algorithm.upper()} - Q-table')
        axes[1, 0].set_xlabel('Action')
        axes[1, 0].set_ylabel('State')
        
        # 4. Optimal policy visualization
        optimal_policy = np.argmax(self.q_table, axis=1)
        policy_map = optimal_policy.reshape(4, 12)  # Cliff Walking is 4x12
        
        # Cliff Walking environment map
        cliff_map = [
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['S', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'G']
        ]
        
        action_symbols = ['↑', '→', '↓', '←']  # UP, RIGHT, DOWN, LEFT (CliffWalking action mapping)
        
        # Draw the environment
        for i in range(4):
            for j in range(12):
                if cliff_map[i][j] == 'C':  # Cliff
                    axes[1, 1].text(j + 0.5, 3 - i + 0.5, 'C', ha='center', va='center', 
                                   fontsize=12, color='red', fontweight='bold')
                elif cliff_map[i][j] == 'S':  # Start (state 36, row=3, col=0)
                    axes[1, 1].text(j + 0.5, 3 - i + 0.5, 'S', ha='center', va='center', 
                                   fontsize=12, color='green', fontweight='bold')
                elif cliff_map[i][j] == 'G':  # Goal (state 47, row=3, col=11)
                    axes[1, 1].text(j + 0.5, 3 - i + 0.5, 'G', ha='center', va='center', 
                                   fontsize=12, color='green', fontweight='bold')
                else:  # Show policy
                    action = policy_map[i, j]
                    axes[1, 1].text(j + 0.5, 3 - i + 0.5, action_symbols[action], 
                                   ha='center', va='center', fontsize=10)
        
        # Draw grid
        for i in range(5):
            axes[1, 1].axhline(y=i, color='black', linewidth=0.5)
        for j in range(13):
            axes[1, 1].axvline(x=j, color='black', linewidth=0.5)
        
        axes[1, 1].set_xlim(0, 12)
        axes[1, 1].set_ylim(0, 4)
        axes[1, 1].set_title(f'{self.algorithm.upper()} - Optimal Policy (S:Start, G:Goal, C:Cliff)')
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'/home/jso1/local/teaching/RL/cliff_walking_{self.algorithm}_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def compare_algorithms():
    """Compare SARSA and Q-learning on Cliff Walking"""
    # Create environment
    env = gym.make('CliffWalking-v1')
    
    # Train SARSA agent with more episodes
    print("=" * 50)
    print("Training SARSA Agent (Extended Training)")
    print("=" * 50)
    sarsa_agent = CliffWalkingAgent(env, algorithm='sarsa', alpha=0.1, gamma=0.9, 
                                   epsilon=0.1, epsilon_decay=0.9999, epsilon_min=0.001)
    sarsa_agent.train(n_episodes=50000)  # Much more episodes
    
    # Train Q-learning agent with more episodes
    print("\n" + "=" * 50)
    print("Training Q-learning Agent (Extended Training)")
    print("=" * 50)
    qlearning_agent = CliffWalkingAgent(env, algorithm='qlearning', alpha=0.1, gamma=0.9, 
                                       epsilon=0.1, epsilon_decay=0.9999, epsilon_min=0.001)
    qlearning_agent.train(n_episodes=50000)  # Much more episodes
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    
    # Get optimal paths
    sarsa_path, sarsa_reward = sarsa_agent.get_optimal_path()
    qlearning_path, qlearning_reward = qlearning_agent.get_optimal_path()
    
    print(f"SARSA optimal path reward: {sarsa_reward}")
    print(f"Q-learning optimal path reward: {qlearning_reward}")
    
    print(f"\nSARSA optimal path: {sarsa_path}")
    print(f"Q-learning optimal path: {qlearning_path}")
    
    # Training performance (during learning)
    sarsa_training_performance = np.mean(sarsa_agent.episode_rewards[-100:])
    qlearning_training_performance = np.mean(qlearning_agent.episode_rewards[-100:])
    
    print(f"\nSARSA training performance (last 100 episodes): {sarsa_training_performance:.2f}")
    print(f"Q-learning training performance (last 100 episodes): {qlearning_training_performance:.2f}")
    
    # True final performance (after learning, no exploration)
    print(f"\n=== True Final Performance (No Exploration) ===")
    sarsa_agent.epsilon = 0  # No exploration
    qlearning_agent.epsilon = 0  # No exploration
    
    sarsa_final_performance = np.mean([sarsa_agent.get_optimal_path()[1] for _ in range(100)])
    qlearning_final_performance = np.mean([qlearning_agent.get_optimal_path()[1] for _ in range(100)])
    
    print(f"SARSA final performance (100 evaluations, no exploration): {sarsa_final_performance:.2f}")
    print(f"Q-learning final performance (100 evaluations, no exploration): {qlearning_final_performance:.2f}")
    
    # Check if optimal path is actually being followed
    print(f"\n=== Detailed Analysis ===")
    print(f"Q-learning epsilon during optimal path: {qlearning_agent.epsilon:.4f}")
    
    # Test optimal path multiple times
    print(f"\nTesting Q-learning optimal path 10 times:")
    for i in range(10):
        path, reward = qlearning_agent.get_optimal_path()
        print(f"Run {i+1}: reward = {reward}, path length = {len(path)}")
    
    # Visualize results
    sarsa_agent.plot_results()
    qlearning_agent.plot_results()
    
    return sarsa_agent, qlearning_agent

if __name__ == "__main__":
    sarsa_agent, qlearning_agent = compare_algorithms()
