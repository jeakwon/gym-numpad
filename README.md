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

### gym-numpad + sb3 RecurrentPPO
```python
import gym
import gym_numpad
from sb3_contrib import RecurrentPPO

model = RecurrentPPO("MlpLstmPolicy", 'numpad2x2_test-v0', verbose=1)
env = model.get_env()

for episode in range(5):
    t, score, done, lstm_states = 0, 0, False, None
    obs = env.reset()
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states)
        obs, reward, done, info = env.step(action)
        
        t+=1
        score+=reward
        print(t, score, obs, reward, done, info, end='\r')
        # env.render(mode='human')
        if done:
            break

model.learn(10000)

for episode in range(5):
    t, score, done, lstm_states = 0, 0, False, None
    obs = env.reset()
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states)
        obs, reward, done, info = env.step(action)
        
        t+=1
        score+=reward
        # env.render(mode='human')
        print(t, score, obs, reward, done, info, end='\r')
        if done:
            print(t, score, obs, reward, done, info)
            break
```
