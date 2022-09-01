
class NumPadEnv2x2(NumPadEnv):
    def __init__(self, render_mode='human', cues=[1, 2, 3, 4, 5]):
        super(NumPadEnv2x2, self).__init__(render_mode=render_mode, size=5)
