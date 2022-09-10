import random
import gym
from gym import spaces
import numpy as np
from typing import Tuple, Optional, List, Union

class NumPadEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, 
        size: int,
        cues: List[Union[int, float, str]],
        n_maps: int = 1,
        steps_per_episode: int = 1000,
    ):
        self.tile_size = 2*size+1
        self.screen_dim = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.steps_per_episode = steps_per_episode

        self.maps = [self.create_numpad(shape=(size, size), cues=cues, seed=seed) for seed in range(n_maps)]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(len(cues))

    def reset(self):
        self.T = 0
        self.map = random.choice(self.maps)
        non_reward_positions = np.argwhere(self.map[2]==0)
        i, j = random.choice(non_reward_positions)

        self.state = self.map.copy()
        self.state[0][i, j] = 1 # set agent init position
        obs = self.state[1][i, j] # get current position cue
        return obs

    def step(self, action:int): 
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        i, j = np.argwhere(self.state[0]==1)[0]
        
        if bool(action): # 0:Stay 1:North 2:West 3:South: 4:East
            self.state[0][i, j] = 0 # set agent old position 0
            th = action*np.pi/2
            i = np.clip(i+np.sin(th), 0, self.state[0].shape[0]-1).astype(np.int32)
            j = np.clip(j+np.cos(th), 0, self.state[0].shape[1]-1).astype(np.int32)
            self.state[0][i, j] = 1 # set agent new position 1

        obs = self.state[1][i, j] # get current position cue
        reward = 0
        done = False

        R = self.state[2]
        S = self.state[3]
        if S[i, j]>0:
            if S[i, j]==S[S>0].min(): # if found right reward sequence
                self.state[3][i, j] = 0 # set reward sequence 0
                if R[i, j]>0: 
                    reward+=R[i, j]
                    self.state[2][i, j] = 0 # set reward value 0
            else: # if found wrong reward sequence
                self.state[0][i, j] = 0 # set agent old position 0
                self.state[3] = self.map[3] # rollback sequences
                non_reward_positions = np.argwhere(self.map[2]==0)
                i, j = random.choice(non_reward_positions)
                self.state[0][i, j] = 1 # set agent new position 1

        if not (R>0).any(): # if no positive reward left
            self.state[0][i, j] = 0 # set agent old position 0
            self.state = self.map.copy()
            non_reward_positions = np.argwhere(self.map[2]==0)
            i, j = random.choice(non_reward_positions)
            self.state[0][i, j] = 1 # set agent new position 1

        self.T += 1
        if self.T == self.steps_per_episode:
            done = True
        
        return obs, reward, done, {}

    @staticmethod
    def create_numpad(
        shape: Tuple[int, int],
        cues: List[Union[int, float, str]],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        :param shape: reward shape (width, height)
        :param cues: list of cues distributed to tiles. ex) [1, 2, 3], ['A', 'B', 'C']
        :seed: map random control
        
        :returns: numpad, [4 x W x H]
        """
        I, J = shape # reward distribution
        H, W = I*2+1, J*2+1 # map size
        n_rwds = I*J
        n_cues = H*W

        rng = np.random.default_rng(seed)
        cues = rng.choice(cues, size=(n_cues,), replace=len(cues)<n_cues)
        rwds = [(2*i+1, 2*j+1) for i in range(I) for j in range(J)]
        seqs = rng.permutation(range(1, n_rwds+1))

        pos_layer = np.zeros(shape=(H, W), dtype=object)
        cue_layer = cues.reshape(H, W)
        rwd_layer = np.zeros(shape=(H, W), dtype=object)
        seq_layer = np.zeros(shape=(H, W), dtype=object)
        
        for (h, w), seq in zip(rwds, seqs):
            rwd_layer[h, w] = 1
            seq_layer[h, w] = seq
            
        return np.stack([pos_layer, cue_layer, rwd_layer, seq_layer])

    def render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
        except ImportError:
            raise gym.error.DependencyNotInstalled(
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
        
        scale = self.screen_dim/self.tile_size
        color = {
            'background' : (100, 100, 100),
            'line' : (255, 255, 255),
            'agent' : (0, 50, 100),
            'cue' : (255, 255, 255),
            'reward_o' : (243, 135, 47),
            'reward_x' : (255, 255, 255),
            'reward_tile' : (35,35,35),
        }

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill(color['background'])        
        blit_pairs = []
        blit_pairs.append( (self.surf, (0,0)) )

        # show grid lines
        for i in range(self.tile_size+1):
            pygame.draw.line(self.surf, color['line'], (0, i*scale), (self.screen_dim, i*scale))
            pygame.draw.line(self.surf, color['line'], (i*scale, 0), (i*scale, self.screen_dim))

        L, M, N = self.state.shape
        for i in range(M):
            for j in range(N):
                p, q, r, s = self.state[:, i, j]
                
                agent_loc = scale*(np.array([i, j])+0.5)
                if p==1:
                    pygame.draw.circle(self.surf, color['agent'], agent_loc, scale/4)
                
                if q!=None:
                    cue_loc = scale*(np.array([i, j]))
                    font = pygame.font.SysFont("arial", int(scale/3), bold=False)
                    text = font.render(f' {q}', False, color['cue'])
                    blit_pairs.append((text, cue_loc))

                if s>0:
                    x, y, w, h = int(scale*i+1), int(scale*j+1), int(scale)-1, int(scale)-1
                    reward_loc = (x, y, w, h)
                    pygame.draw.rect(self.surf, color['reward_tile'], reward_loc)

                s_init = self.map[:, i, j][3]
                if s_init>0:
                    sequence_loc = scale*(np.array([i, j])+0.5)
                    font = pygame.font.SysFont("arial", int(scale/2), bold=True)
                    text = font.render(f'{s_init}', False, color['reward_o'] if r>0 else color['reward_x'])
                    zone = text.get_rect(center=sequence_loc)
                    blit_pairs.append((text, zone))

        for item, loc in blit_pairs:
            self.screen.blit(item, loc)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

class NumPad2x2(NumPadEnv):
    def __init__(self, size=2, cues=range(25), n_maps=1, steps_per_episode=1000):
        super().__init__(size=size, cues=cues, n_maps=n_maps, steps_per_episode=steps_per_episode)

class NumPad3x3(NumPadEnv):
    def __init__(self, size=3, cues=range(49), n_maps=1, steps_per_episode=1000):
        super().__init__(size=size, cues=cues, n_maps=n_maps, steps_per_episode=steps_per_episode)

class NumPad4x4(NumPadEnv):
    def __init__(self, size=4, cues=range(81), n_maps=1, steps_per_episode=1000):
        super().__init__(size=size, cues=cues, n_maps=n_maps, steps_per_episode=steps_per_episode)

if __name__=="__main__":
    env = NumPad2x2()
    for episode in range(2):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
