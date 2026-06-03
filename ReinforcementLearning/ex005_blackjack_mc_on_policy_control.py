import gymnasium as gym
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm 

env = gym.make("Blackjack-v1")
num_timesteps = 100

def epsilon_greedy_policy(state, Q):
    epsilon = 0.5
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state, x)])
    
def greedy_policy(state, Q):
    return max(list(range(env.action_space.n)), key = lambda x: Q[(state, x)])

def generate_episode(Q):
    episode = []
    state, _ = env.reset()
    for t in range(num_timesteps):
        action = epsilon_greedy_policy(state, Q)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        if terminated or truncated:
            break
        state = next_state
    return episode

def evaluate_policy(Q=None):
    num_episodes = 500000
    num_wins = 0
    num_draws = 0
    num_losses = 0
    num_total = 0
    for i in tqdm(range(num_episodes)):
        state, _ = env.reset()
        for t in range(num_timesteps):
            if Q is None:
                action = env.action_space.sample()
            else:
                action = greedy_policy(state, Q)
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if reward == 1:
                    num_wins += 1
                elif reward == 0:
                    num_draws += 1
                elif reward == -1:
                    num_losses += 1
                num_total += 1
                break
            state = next_state

    print("wins: %d (%.2f%%), draws: %d (%.2f%%), losses: %d (%.2f%%)"%(num_wins, num_wins/num_total*100, num_draws, num_draws/num_total*100, num_losses, num_losses/num_total*100))


def main():

    Q = defaultdict(float)
    total_return = defaultdict(float)
    N = defaultdict(int)

    num_iterations = 500000
    for i in tqdm(range(num_iterations)):
        episode = generate_episode(Q)
        all_state_action_pairs = [(s,a) for (s,a,r) in episode]
        rewards = [r for (s,a,r) in episode]
        for t, (state, action, _) in enumerate(episode):
            if not (state, action) in all_state_action_pairs[0:t]:
                R = sum(rewards[t:])
                total_return[(state,action)] = total_return[(state,action)] + R
                N[(state,action)] += 1
                Q[(state,action)] = total_return[(state,action)] / N[(state,action)]

    print("Evaluating random policy...")
    evaluate_policy(None)
    print("Evaluating learned policy...")
    evaluate_policy(Q)

if __name__=='__main__':
    main()

