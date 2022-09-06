# gym-numpad

## Install
```
pip install -U git+https://github.com/jeakwon/gym-numpad
```

## gym-numpad test
```python
import gym
import gym_numpad

env=gym.make('numpad2x2_test-v0')
for episode in range(2):
    env.reset()
    t, score = 0, 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, info =env.step(action)
        env.render(mode='human')
        t+=1
        score+=reward
        print(t, score, obs, reward, done, info, end='\r')
        if done:
            break
```
