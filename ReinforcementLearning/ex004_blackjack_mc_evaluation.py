import gymnasium as gym
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

env = gym.make('Blackjack-v1')
num_timesteps = 100

def policy(state):
	return 0 if state[0] > 19 else 1

def generate_episode(policy):
	episode = []
	state, _ = env.reset()

	for t in range(num_timesteps):
		action = policy(state)
		next_state, reward, terminated, truncated, info = env.step(action)
		episode.append((state, action, reward))
		if terminated or truncated:
			break
		state = next_state

	return episode

total_return = defaultdict(float)
N = defaultdict(int)

num_iterations = 50000
for i in tqdm(range(num_iterations)):
	episode = generate_episode(policy)
	states, actions, rewards = zip(*episode)
	for t, state in enumerate(states):
		if state not in states[0:t]:
			R = (sum(rewards[t:]))
			total_return[state] = total_return[state] + R
			N[state] = N[state] + 1

total_return = pd.DataFrame(total_return.items(), columns=['state', 'total_return'])
N = pd.DataFrame(N.items(), columns=['state', 'N'])
df = pd.merge(total_return, N, on='state')
df['value'] = df['total_return'] / df['N']
#print(sum(df['total_return'].values))

print(df)

