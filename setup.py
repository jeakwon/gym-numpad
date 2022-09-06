from setuptools import setup

setup(name="gym_numpad",
      version="0.1",
      url="https://github.com/jeakwon/gym-numpad",
      author="Jea Kwon",
      license="MIT",
      packages=["gym_numpad", "gym_numpad.envs"],
      install_requires = ["gym==0.21", "pygame", "numpy", "pyglet"]
)
