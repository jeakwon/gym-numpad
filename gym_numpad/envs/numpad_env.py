import math
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer


class NumPadEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }
    def __init__(self, render_mode: Optional[str] = None, size=2, cues=['a', 'b', 'c', 'd', 'e'], init_pos=[0, 0]):
        self.size = 2*size+1
        self.cues = cues
        self.init_pos = init_pos
        self.MIN = 0
        self.MAX = self.size-1
        self.reward_zones = []
        self.vacant_zones = []
        for i in range(self.size):
            for j in range(self.size):
                if (i%2==1)&(j%2==1):
                    self.reward_zones.append([i, j])
                else:
                    self.vacant_zones.append([i, j])
        self.Q = self.np_random.choice(range(len(cues)), (self.size, self.size))

        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(len(cues))

    def step(self, action:int): 
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        
        if bool(action): # 0:Stay 1:North 2:West 3:South: 4:East
            th = action*np.pi/2
            i = np.clip(self.pos[0]+np.sin(th), self.MIN, self.MAX).astype(np.int32)
            j = np.clip(self.pos[1]+np.cos(th), self.MIN, self.MAX).astype(np.int32)
            self.pos = [i, j]
            self.state = self.cues[self.Q[i, j]]
        
        reward = 0
        terminated = False
        if self.pos in self.reward_seqs:
            if self.pos == self.reward_seqs[0]:
                reward=1
                self.reward_seqs = self.reward_seqs[1:]
            else:
                terminated = True
        if len(self.reward_seqs)==0:
            terminated = True

        self.renderer.render_step()
        return self.state, reward, terminated, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, 0, len(self.vacant_zones))
        if self.init_pos == None:
            self.pos = self.vacant_zones[int(self.np_random.integers(low, high, 1))]
        else:
            self.pos = self.init_pos
        self._init_pos = self.pos
        self.state = self.cues[self.Q[self.pos[0], self.pos[1]]]
        self.reward_seqs = self.np_random.permutation(self.reward_zones).tolist()
        self.renderer.reset()
        self.renderer.render_step()
        return self.state, {}

    def render(self):
        return self.renderer.get_renders()

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))
        
        L = self.screen_dim/self.size

        # show init position 
        x = int(self._init_pos[0] * L + 0.5 + 1)
        y = int(self._init_pos[1] * L + 0.5 + 1)
        w = int(L + 0.5 - 1)
        h = int(L + 0.5 - 1)
        pygame.draw.rect(self.surf, [128, 128, 128], [x, y, w, h])

        # show grid lines
        for i in range(self.size+1):
            pygame.draw.line(self.surf, [0 ,0, 0], (0, i*L), (self.screen_dim, i*L))
            pygame.draw.line(self.surf, [0 ,0, 0], (i*L, 0), (i*L, self.screen_dim))

        # show agent
        for i, reward_seq in enumerate(self.reward_seqs):
            x = int(reward_seq[0] * L + 0.5 + 1)
            y = int(reward_seq[1] * L + 0.5 + 1)
            w = int(L + 0.5 - 1)
            h = int(L + 0.5 - 1)
            reward_pos = [x, y, w, h]
            color = [0, 255, 0] if i==0 else [255, 0, 0]
            pygame.draw.rect(self.surf, color, reward_pos)
            
        agent_pos = L*(np.array(self.pos)+0.5)
        pygame.draw.circle(self.surf, [0, 0, 0, 0], agent_pos, L/2)

        self.screen.blit(self.surf, (0, 0))
        
        # show cues
        for i in range(self.size):
            for j in range(self.size):
                font = pygame.font.SysFont("arial", 24) 
                text = font.render(self.cues[self.Q[i, j]], False, [0, 0, 0])
                self.screen.blit(text, [i*L, j*L])

        # show reward sequence
        for k, [i, j] in enumerate(self.reward_seqs):
            R_tot = len(self.reward_zones)
            R_rem = len(self.reward_seqs)
            K = np.arange(R_tot)[-R_rem:][k]+1
            font = pygame.font.SysFont("arial", 30) 
            text = font.render(str(K), False, [0, 0, 0])
            zone = text.get_rect(center=[(i+0.5)*L, (j+0.5)*L])
            self.screen.blit(text, zone)
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is  not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

class NumPadEnv2x2(NumPadEnv):
    def __init__(self, render_mode='human', cues=[1, 2, 3, 4, 5]):
        super(NumPadEnv2x2, self).__init__(render_mode=render_mode, size=5)

if __name__ == "__main__":
    env = NumPadEnv(render_mode='human', size=7, init_pos=None)
    for episode in range(30):
        env.reset(seed=episode)
        env.action_space.seed(episode)
        score = 0
        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, _, _ = env.step(action)
            print(action, observation, reward)
            score+=reward
            if done:
                print("Episode finished after {} timesteps. Score: {}".format(t+1, score))
                break
