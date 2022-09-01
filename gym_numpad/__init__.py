from gym.envs.registration import register

register(
    id='numpad2x2-v0',
    entry_point='gym_numpad.envs:NumPadEnv2x2',
)
