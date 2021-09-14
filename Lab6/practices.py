import gym
import random
import numpy as np

env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('LunarLander-v2')

state = env.reset()
action_space = env.action_space

next_state, reward, done, _ = env.step(action_space.sample())
print(action_space)
print(action_space.sample())
print(next_state)
print(reward)
print(done)

env.close()

print(random.choice(np.arange(4)))

