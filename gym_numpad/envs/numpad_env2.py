
from typing import Optional, Union, List, Tuple
import numpy as np
import gym
from gym import spaces

class NumPadEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 50,
    }
    def __init__(
        self, 
        size: int = 1,
        cues: Optional[List[Union[int, float, str]]] = None,
        init_policy: Optional[str] = None, 
        sequence_policy: Optional[str] = None, 
        value_policy: Optional[str] = None, 
        normalize: bool = False,
        n_maps: Optional[int] = None,
        total_steps: Optional[int] = None,
    ):
        self.params = dict(
            size=size,
            cues=cues,
            init_policy=init_policy,
            sequence_policy=sequence_policy,
            value_policy=value_policy,
            normalize=normalize,
        )
        self.tile_size = 2*size+1
        self.screen_dim = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.total_steps = total_steps
        self.steps_count = None

        self.maps = [self.create_numpad(seed=i, **self.params) for i in range(n_maps)]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(len(cues))

    def step(self, action:int): 
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        n_rows = self.state.shape[1]
        n_cols = self.state.shape[2]
        i, j = np.where(self.state[0]==1)
        
        if bool(action): # 0:Stay 1:North 2:West 3:South: 4:East
            self.state[0, i, j] = 0
            th = action*np.pi/2
            i = np.clip(i+np.sin(th), 0, n_rows-1).astype(np.int32)
            j = np.clip(j+np.cos(th), 0, n_cols-1).astype(np.int32)
            self.state[0, i, j] = 1
        

        P, Q, R, S = self.state
        p, q, r, s = self.state[:, i, j]

        obs = q
        reward = 0
        done = False
        
        if self.total_steps is None:
            if R[i, j]>0:
                if s==S[R>0].min():
                    reward+=r[0]
                    self.state[2, i, j] = 0
                    self.state[3, i, j] = 0
                else:
                    done = True
            
            if not (R>0).any(): # if no positive reward left
                done = True
        else:
            assert isinstance(self.total_steps, int), 'total_steps must be positive integer'
            if s>0:
                if s==S[S>0].min(): # if found right reward sequence
                    self.state[3, i, j] = 0 # set reward sequence 0
                    if r>0: 
                        reward+=r[0] 
                        self.state[2, i, j] = 0 # set reward value 0
                else: # if found wrong reward sequence
                    self.state[3] = self.init_state[3].copy() # reset only reward sequence not reward value
                    vacant_positions = np.argwhere(self.state[3]==0)
                    idx = np.random.randint(vacant_positions.shape[1])
                    m, n = vacant_positions[idx]   
                    # regen agent position
                    self.state[0][i, j] = 0
                    self.state[0][-1, -1] = 1
                    
        
            if not (R>0).any(): # if no positive reward left
                self.state = self.init_state.copy()

            # Termination
            self.steps_count += 1
            if self.steps_count == self.total_steps:
                done = True
        return obs, reward, done, {}

    def reset(self):
        if self.total_steps is not None:
            self.steps_count = 0
        idx = np.random.randint(len(self.maps))
        self.init_state = self.maps[idx]
        self.state = self.init_state.copy()
        P, Q, R, S = self.state
        i, j = np.where(P==1)
        obs = Q[i, j]
        return obs

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

                s_init = self.init_state[:, i, j][3]
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

    def create_numpad(self, 
        size: int, 
        cues: Optional[List[Union[int, float, str]]] = None,
        init_policy: Optional[str] = None, 
        sequence_policy: Optional[str] = None, 
        value_policy: Optional[str] = None, 
        normalize: bool = False,
        seed: Optional[int] = None,
    ):
        """
        :param size: 
            determines number of rewards (size*size) and map shape (2*size+1, 2*size+1).
        :param cues: 
            list of cues distributed to tiles. ex) [1, 2, 3], ['A', 'B', 'C']
        :param init_policy: {top_left|top_right|bottom_left|bottom_right|center|random} 
            determines agent init position
        :param sequence_policy: {l2r_l2r|l2r_r2l|shuffle} 
            determines sequence of rewards. l2r means left to right, and r2l means right to left
            if None, 0
        :param value_policy: {equal|sequential} 
            determines reward values for each reward location. 
            if None, 0
        :param normalize: default False
            if True, reward values are normalized by dividing with total rewards
        :param seed:
            if provided, random sequences are fixed.
        
        :returns: map. np.ndarray with shape of (4, size, size)
            reward_map <= np.stack([P, Q, R, S])
            - P : Agent Position
            - Q : Cue Distribution
            - R : Reward Location
            - S : Reward Sequence

        :example:

            reward_map[:, i, j] -> np.array([agent, cue, rew_seq, rew_val])
        """
        rng = np.random.default_rng(seed)
        map_shape = (2*size+1, 2*size+1)    
        num_rewards = size*size

        M = np.zeros([4, size, size])
        for i in range(size):
            for j in range(size):
                M[0, i, j] = i
                M[1, i, j] = j

        seq = np.zeros(shape=(size, size))
        if sequence_policy is not None:
            if sequence_policy == 'l2r_l2r':
                seq = np.arange(1, num_rewards+1)
                seq = seq.reshape([size, size])
            elif sequence_policy == 'l2r_r2l':
                seq = np.arange(1, num_rewards+1)
                seq = seq.reshape([size, size])
                seq[1::2] = np.fliplr(seq[1::2])
            elif sequence_policy == 'shuffle':
                seq = np.arange(1, num_rewards+1)
                seq = rng.permutation(seq)
                seq = seq.reshape((size, size))
            else:
                raise Exception("Invalid sequence_policy. should be one of [ None | 'l2r_l2r' | 'l2r_r2l' | 'shuffle' ] ")
            
            M[2, :, :] += seq
            
        val = np.zeros(shape=(size, size))
        if value_policy is not None:
            if value_policy == 'equal':
                val+=1
            elif value_policy == 'sequential':
                val+=seq
            else:
                raise Exception("Invalid value_policy. should be one of [ None | 'equal' | 'sequential' ] ")
            
            M[3, :, :] += val/val.sum() if normalize else val

        rewards = [(int(2*i+1), int(2*j+1), int(s), float(v)) for i, j, s, v in M.reshape(4, -1).T]

        init_position = (0, 0)
        if init_policy is not None:
            if init_policy == 'top_left':
                init_position = (0, 0)
            elif init_policy == 'top_right':
                init_position = (0, 2*size)
            elif init_policy == 'bottom_left':
                init_position = (2*size, 0)
            elif init_policy == 'bottom_right':
                init_position = (2*size, 2*size)
            elif init_policy == 'center':
                init_position = (size, size)
            elif init_policy == 'random':
                init_position = (rng.integers(0, map_shape[0]), rng.integers(0, map_shape[1]))
            else:
                raise Exception("Invalid init_policy. should be one of [ None | 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right' | 'center' | 'random' ] ")
            
            if init_position in [(i, j) for i, j, _, _ in rewards]:
                i, j = init_position
                shift = rng.choice([-1, 1])
                init_position = rng.choice([(i, j+shift), (i+shift, j)])
            
        return self.get_reward_map(
            shape=map_shape, 
            position=init_position, 
            cues=cues, 
            rewards=rewards, 
            seed=seed)

    @staticmethod
    def get_reward_map(
        shape: Tuple[int, int],
        position: Optional[Tuple[int, int]] = None,
        cues: Optional[List[Union[int, float, str]]] = None,
        rewards: Optional[List[Tuple[int, int, int, float]]] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        :param shape: reward_map shape (width, height)
        :param position: agent position (i, j)
        :param cues: list of cues distributed to tiles. ex) [1, 2, 3], ['A', 'B', 'C']
        :param rewards: list of reward tuples. {(i, j) | (i, j, rwd_seq) | (i, j, rwd_seq, rwd_val)}
        
        :returns: map. np.ndarray with shape of (4, width, height)
            reward_map <= np.stack([P, Q, R, S])
            - P : Agent Position
            - Q : Cue Distribution
            - R : Reward Values
            - S : Reward Sequence

        :example:

            reward_map[:, i, j] -> np.array([agent, cue, rew_seq, rew_val])
        """
        width, height = shape
        P = np.zeros(shape=shape, dtype=object) # Agent Position
        Q = np.zeros(shape=shape, dtype=object) # Cue Distribution
        R = np.zeros(shape=shape, dtype=object) # Reward Location
        S = np.zeros(shape=shape, dtype=object) # Reward Sequence

        rng = np.random.default_rng(seed)

        if position is not None:
            i, j = position
            P[i, j] = 1

        if cues is not None:
            Q[:, :] = rng.choice(cues, size=shape, replace=len(cues)<width*height)
        
        if rewards is not None:
            for i, j, *x in rewards:
                S[i, j] = x[0] if len(x)>=1 else 0
                R[i, j] = x[1] if len(x)==2 else 1

        return np.stack([P, Q, R, S])

class NumPad2x2_test(NumPadEnv):
    def __init__(self):
        super().__init__(size=2, cues=range(25), sequence_policy='l2r_r2l', value_policy='equal', init_policy='top_left', total_steps=1000, n_maps=4)

        
class NumPad2x2(NumPadEnv):
    def __init__(self, cues=range(2*(2*2+1)), sequence_policy='shuffle', value_policy='equal', init_policy='random', total_steps=1000, n_maps=4, **kwargs):
        super().__init__(size=2, cues=cues, sequence_policy=sequence_policy, value_policy=value_policy, init_policy=init_policy, total_steps=total_steps, n_maps=n_maps, **kwargs)

class NumPad3x3(NumPadEnv):
    def __init__(self, cues=range(3*(3*2+1)), sequence_policy='shuffle', value_policy='equal', init_policy='random', total_steps=1000, n_maps=4, **kwargs):
        super().__init__(size=3, cues=cues, sequence_policy=sequence_policy, value_policy=value_policy, init_policy=init_policy, total_steps=total_steps, n_maps=n_maps, **kwargs)

class NumPad4x4(NumPadEnv):
    def __init__(self,cues=range(4*(4*2+1)), sequence_policy='shuffle', value_policy='equal', init_policy='random', total_steps=1000, n_maps=4, **kwargs):
        super().__init__(size=4, cues=cues, sequence_policy=sequence_policy, value_policy=value_policy, init_policy=init_policy, total_steps=total_steps, n_maps=n_maps, **kwargs)
