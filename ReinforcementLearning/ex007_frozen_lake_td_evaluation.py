import gymnasium as gym
import pandas as pd
from tqdm import tqdm
env = gym.make('FrozenLake-v1', render_mode="ansi", is_slippery=True)

def random_policy():
    return env.action_space.sample()

V = {}
for s in range(env.observation_space.n):
    V[s] = 0.0

alpha = 0.1
gamma = 0.9

num_episodes = 500000
num_timesteps = 1000

for i in tqdm(range(num_episodes)):
    s, _ = env.reset()
    for t in range(num_timesteps):
        a = random_policy()
        s_, r, terminated, truncated, _ = env.step(a)
        V[s] += alpha * (r + gamma * V[s_] - V[s])    # TD update rule
        s = s_
        if terminated or truncated:
            break

df = pd.DataFrame(list(V.items()), columns=['state', 'value'])
print(df)
