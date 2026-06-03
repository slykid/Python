import sys
import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
env = gym.make('Blackjack-v1')

def random_policy(nA):
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

def greedy_policy(Q):
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn

def mc_off_policy(env, num_episodes, behavior_policy, max_time=100, discount_factor=1.0):
    Q = defaultdict(lambda:np.zeros(env.action_space.n))
    C = defaultdict(lambda:np.zeros(env.action_space.n))

    target_policy = greedy_policy(Q)

    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state, _ = env.reset()
        for t in range(max_time):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            if terminated or truncated:
                break
            state = next_state

        G = 0.0
        W = 1.0
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action != np.argmax(target_policy(state)):
                break
            W = W * 1./behavior_policy(state)[action]

    return Q, target_policy

def evaluate_policy(env, policy):
    num_episodes = 500000
    num_timesteps = 100
    num_wins = 0
    num_draws = 0
    num_losses = 0
    num_total = 0
    for i in tqdm(range(num_episodes)):
        state, _ = env.reset()
        for t in range(num_timesteps):
            action = np.argmax(policy(state))
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

random_policy = random_policy(env.action_space.n)
Q, policy = mc_off_policy(env, num_episodes=500000, behavior_policy=random_policy)
evaluate_policy(env, policy)
