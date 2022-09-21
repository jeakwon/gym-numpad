import random
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces
from gym_numpad.envs.utils import create_2d_connected_sequences

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
        random_init: bool = False,
        random_regen: bool = False,
        neighbor_sequence: bool = True,
        diag_neighbors = False, 
        custom_maps: Optional[List] = None,
        seed: int = 0,
    ):
        self.tile_size = 2 * size - 1
        self.screen_dim = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.steps_per_episode = steps_per_episode
        self.random_init = random_init
        self.random_regen = random_regen

        if custom_maps is not None:
            assert isinstance(
                custom_maps, list
            ), "custom_maps should be provided as list"
            for map_ in custom_maps:
                self.map_sanity_check(map_)
            self.maps = custom_maps
        else:
            if neighbor_sequence:
                self.maps = self.create_hamiltonian_numpads(n_maps=n_maps, shape=(size, size), cues=cues, diag_neighbors=diag_neighbors, seed=seed)

            else:
                self.maps = self.create_numpads(n_maps=n_maps, shape=(size, size), cues=cues, seed=seed)


        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([len(cues), 2, 3])
        self.possible_actions = np.arange(self.action_space.n)
        self.invalid_actions: List[int] = []

    def reset(self):
        self.T = 0
        self.map = random.choice(self.maps)
        non_reward_positions = np.argwhere(self.map[2] == 0)
        i, j = random.choice(non_reward_positions)
        
        possible_rwd_cycles = (self.steps_per_episode - 1) / self.rwd_seq_1cycle_dist(self.map[3])
        rwds_per_cycle = self.map[2].sum()
        self.expected_max_rwd = possible_rwd_cycles * rwds_per_cycle

        self.state = deepcopy(self.map)
        self.state[0][i, j] = 1  # set agent init position
        cue = self.state[1][i, j]  # get current position cue
        obs = np.array([cue, 0, False], dtype=np.int32)
        invalid_actions = []
        if j == self.tile_size - 1:
            invalid_actions.append(2)
        elif j == 0:
            invalid_actions.append(3)
        if i == self.tile_size - 1:
            invalid_actions.append(1)
        elif i == 0:
            invalid_actions.append(4)
        self.invalid_actions = invalid_actions
        return obs

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        i, j = np.argwhere(self.state[0] == 1)[0]

        cue = self.state[1][i, j]  # get current position cue
        reward = 0
        done = False
        invalid_sequence = 0  # 0: empty signal, 1: correct signal, 2: fail signal

        action_pool = [
            (0, 0),
            (1, 0),
            (0, 1),
            (0, -1),
            (-1, 0),
        ]  # 0:Stay 1:East 2:South 3:North: 4:West
        di, dj = action_pool[action]
        i_dst, j_dst = int(i + di), int(j + dj)

        self.state[0][i, j] = 0  # set agent old position 0
        i = np.clip(i_dst, 0, self.tile_size - 1).astype(np.int32)
        j = np.clip(j_dst, 0, self.tile_size - 1).astype(np.int32)
        self.state[0][i, j] = 1  # set agent new position 1
        invalid_actions = []
        if j == self.tile_size - 1:
            invalid_actions.append(2)
        elif j == 0:
            invalid_actions.append(3)
        if i == self.tile_size - 1:
            invalid_actions.append(1)
        elif i == 0:
            invalid_actions.append(4)
        self.invalid_actions = invalid_actions

        if self.state[3][i, j] > 0:
            if (
                self.state[3][i, j] == self.state[3][self.state[3] > 0].min()
            ):  # if found right reward sequence
                self.state[3][i, j] = 0  # set reward sequence 0
                invalid_sequence = 1
                if self.state[2][i, j] > 0:
                    reward += self.state[2][i, j]
                    self.state[2][i, j] = 0  # set reward value 0
            else:  # if found wrong reward sequence
                if not (self.state[3] == 1).any():
                    invalid_sequence = 2
                    self.reset_seq()

                    self.reset_pos()
                    if self.random_regen:
                        non_reward_positions = np.argwhere(self.map[2] == 0)
                        i, j = random.choice(non_reward_positions)
                    self.state[0][i, j] = 1  # set agent new position 1

        if not (self.state[2] > 0).any():  # if no positive reward left
            self.reset_rwd()
            self.reset_seq()

            self.reset_pos()
            if self.random_init:
                non_reward_positions = np.argwhere(self.map[2] == 0)
                i, j = random.choice(non_reward_positions)
            self.state[0][i, j] = 1  # set agent new position 1

        self.T += 1
        if self.T == self.steps_per_episode:
            done = True

        obs = np.array([cue, reward, invalid_sequence], dtype=np.int32)
        return obs, reward, done, {'expected_max_rwd':self.expected_max_rwd}

    def reset_pos(self):  # reset_pos gives no agent position
        self.state[0] = deepcopy(self.map[0])

    def reset_cue(self):  # reset_cue gives orignal map cue distribution
        self.state[1] = deepcopy(self.map[1])

    def reset_rwd(self):  # reset_rwd gives original map reward distribution
        self.state[2] = deepcopy(self.map[2])

    def reset_seq(self):  # reset_seq give original map reward sequence distiribution
        self.state[3] = deepcopy(self.map[3])

    def action_masks(self) -> List[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]

    @staticmethod
    def create_numpads(
        n_maps: int,
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
        numpads = []

        I, J = shape  # reward distribution
        H, W = I * 2 - 1, J * 2 - 1  # map size
        n_rwds = I * J
        n_cues = H * W

        rng = np.random.default_rng(seed)
        
        for i in range(n_maps):
            cues = rng.choice(cues, size=(n_cues,), replace=len(cues) < n_cues)
            rwds = [(2 * i, 2 * j) for i in range(I) for j in range(J)]
            seqs = rng.permutation(range(1, n_rwds + 1))

            pos_layer = np.zeros(shape=(H, W), dtype=object)
            cue_layer = cues.reshape(H, W)
            rwd_layer = np.zeros(shape=(H, W), dtype=object)
            seq_layer = np.zeros(shape=(H, W), dtype=object)

            for (h, w), seq in zip(rwds, seqs):
                rwd_layer[h, w] = 1
                seq_layer[h, w] = seq
            
            numpad = np.stack([pos_layer, cue_layer, rwd_layer, seq_layer])
            
            numpads.append(numpad)
        
        return numpads
    
    @staticmethod
    def create_hamiltonian_numpads(
        n_maps: int,
        shape: Tuple[int, int],
        cues: List[Union[int, float, str]],
        seed: Optional[int] = None,
        diag_neighbors: bool = False,
    ) -> np.ndarray:
        """
        :param shape: reward shape (width, height)
        :param cues: list of cues distributed to tiles. ex) [1, 2, 3], ['A', 'B', 'C']
        :seed: map random control

        :returns: numpad, [4 x W x H]
        """
        numpads = []
        
        I, J = shape  # reward distribution
        H, W = I * 2 - 1, J * 2 - 1  # map size
        n_rwds = I * J
        n_cues = H * W

        rng = np.random.default_rng(seed)

        all_seqs = create_2d_connected_sequences(I, J, diag_neighbors=diag_neighbors, seed=seed, num_paths=n_maps)        
        for i in range(n_maps):
           
            cues = rng.choice(cues, size=(n_cues,), replace=len(cues) < n_cues)
            rwds = [(2 * i, 2 * j) for i in range(I) for j in range(J)]
            seqs = all_seqs[i].flatten()

            pos_layer = np.zeros(shape=(H, W), dtype=object)
            cue_layer = cues.reshape(H, W)
            rwd_layer = np.zeros(shape=(H, W), dtype=object)
            seq_layer = np.zeros(shape=(H, W), dtype=object)

            for (h, w), seq in zip(rwds, seqs):
                rwd_layer[h, w] = 1
                seq_layer[h, w] = seq
            numpad = np.stack([pos_layer, cue_layer, rwd_layer, seq_layer])
            
            numpads.append(numpad)
        
        return numpads
    
    @staticmethod
    def map_sanity_check(map_):
        assert isinstance(map_, np.ndarray), "map should be provided as numpy.ndarray"
        assert map_.ndim == 3, "custom_map array shape should be (4, H, W)"
        assert set(map_[0].flatten().tolist()) == set(
            [0]
        ), "Map first layer is agent position layer, which should contain only zeros"
        assert set(map_[2].flatten().tolist()) == set(
            [0, 1]
        ), "Map third layer is reward location layer, which should contain only zeros and ones"

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

        scale = self.screen_dim / self.tile_size
        color = {
            "background": (100, 100, 100),
            "line": (255, 255, 255),
            "agent": (0, 50, 100),
            "cue": (255, 255, 255),
            "reward_o": (243, 135, 47),
            "reward_x": (255, 255, 255),
            "reward_tile": (35, 35, 35),
        }

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill(color["background"])
        blit_pairs = []
        blit_pairs.append((self.surf, (0, 0)))

        # show grid lines
        for i in range(self.tile_size + 1):
            pygame.draw.line(
                self.surf, color["line"], (0, i * scale), (self.screen_dim, i * scale)
            )
            pygame.draw.line(
                self.surf, color["line"], (i * scale, 0), (i * scale, self.screen_dim)
            )

        L, M, N = self.state.shape
        for i in range(M):
            for j in range(N):
                p, q, r, s = self.state[:, i, j]

                agent_loc = scale * (np.array([i, j]) + 0.5)

                if q != None:
                    cue_loc = scale * (np.array([i, j]))
                    font = pygame.font.SysFont("arial", int(scale / 3), bold=False)
                    text = font.render(f" {q}", False, color["cue"])
                    blit_pairs.append((text, cue_loc))

                if s > 0:
                    x, y, w, h = (
                        int(scale * i + 1),
                        int(scale * j + 1),
                        int(scale) - 1,
                        int(scale) - 1,
                    )
                    reward_loc = (x, y, w, h)
                    pygame.draw.rect(self.surf, color["reward_tile"], reward_loc)

                s_init = self.map[:, i, j][3]
                if s_init > 0:
                    sequence_loc = scale * (np.array([i, j]) + 0.5)
                    font = pygame.font.SysFont("arial", int(scale / 2), bold=True)
                    text = font.render(
                        f"{s_init}",
                        False,
                        color["reward_o"] if r > 0 else color["reward_x"],
                    )
                    zone = text.get_rect(center=sequence_loc)
                    blit_pairs.append((text, zone))

                if p == 1:
                    pygame.draw.circle(self.surf, color["agent"], agent_loc, scale / 4)

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
    
    @staticmethod
    def rwd_seq_1cycle_dist(M):
        seq = M[np.where(M>0)]
        pos = np.argwhere(M>0)[np.argsort(seq)]
        pos_ = np.roll(pos, -1, axis=0)
        dist = np.linalg.norm(pos_-pos, ord=1, axis=1) # L1 norm
        return dist.sum()
            

class NumPad2x2(NumPadEnv):
    def __init__(
        self, size=2, cues=range(9), n_maps=1, steps_per_episode=1000, **kwargs
    ):
        super().__init__(
            size=size,
            cues=cues,
            n_maps=n_maps,
            steps_per_episode=steps_per_episode,
            **kwargs,
        )


class NumPad3x3(NumPadEnv):
    def __init__(
        self, size=3, cues=range(25), n_maps=1, steps_per_episode=1000, **kwargs
    ):
        super().__init__(
            size=size,
            cues=cues,
            n_maps=n_maps,
            steps_per_episode=steps_per_episode,
            **kwargs,
        )


class NumPad4x4(NumPadEnv):
    def __init__(
        self, size=4, cues=range(49), n_maps=1, steps_per_episode=1000, **kwargs
    ):
        super().__init__(
            size=size,
            cues=cues,
            n_maps=n_maps,
            steps_per_episode=steps_per_episode,
            **kwargs,
        )

if __name__ == "__main__":
    # Parallel environments
    n_envs = 4
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env

    # Parallel environments
    n_envs = 4
    # map_ = NumPadEnv.create_numpad(shape=(3, 3), cues=range(10))
    env = make_vec_env(NumPad3x3, n_envs=n_envs, env_kwargs=dict(n_maps=3, diag_neighbors=False, cues=range(10)))
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    model.learn(total_timesteps=1000_000)
    # model.save("rppo_numpad2x2_test")

    # del model  # remove to demonstrate saving and loading

    # model = RecurrentPPO.load("rppo_numpad2x2_test")

    obs = env.reset()

    lstm_states = None
    episode_starts = np.ones((n_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=False
        )
        obs, rewards, dones, info = env.step(action)
        print(info)
        episode_starts = dones
        env.render()
