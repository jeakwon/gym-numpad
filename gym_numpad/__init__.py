from gym.envs.registration import register

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

register(
    id='numpad2x2rand-v0',
    entry_point='gym_numpad.envs:NumPad2x2RandomInit',
)

register(
    id='numpad3x3rand-v0',
    entry_point='gym_numpad.envs:NumPad3x3RandomInit',
)

register(
    id='numpad4x4rand-v0',
    entry_point='gym_numpad.envs:NumPad4x4RandomInit',
)
