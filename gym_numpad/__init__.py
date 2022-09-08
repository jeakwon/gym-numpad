from gym.envs.registration import register

register(
    id='numpad2x2_test-v0',
    entry_point='gym_numpad.envs:NumPad2x2_test',
)


register(
    id='numpad2x2-v0',
    entry_point='gym_numpad.envs:NumPad2x2',
)


register(
    id='numpad3x3-v0',
    entry_point='gym_numpad.envs:NumPad3x3',
)


register(
    id='numpad4x4-v0',
    entry_point='gym_numpad.envs:NumPad4x4',
)
