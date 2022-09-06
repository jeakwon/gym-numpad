# gym-numpad

```
pip install -U git+https://github.com/jeakwon/gym-numpad
```

```python
import gym
import gym_numpad

env=gym.make('numpad2x2-test')
for episode in range(2):
    env.reset()
    t, score = 0, 0
    while true:
        action = env.action_space.sample()
        obs, reward, done, info =env.step(action)
        env.render(mode='human')
        t+=1
        score+=reward
        print(t, score, obs, reward, done, info)
        if done:
            break
```
