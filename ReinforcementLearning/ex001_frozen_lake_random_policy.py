from tqdm import tqdm
import gymnasium as gym
env = gym.make("FrozenLake-v1", render_mode="human")

num_episodes = 100
num_timesteps = 50
total_reward = 0
total_timestep = 0

for i in tqdm(range(num_episodes)):
    state, info = env.reset()

    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        new_state, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward

        if terminated or truncated:
            break

    total_timestep += t

print("Number of successful episodes: %d/%d" % (total_reward, num_episodes))
print("Average number of timesteps per episode: %.2f" % (total_timestep/num_episodes))

