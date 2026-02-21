import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cpp_env

class GameEnv(gym.Env):
    """Lightweight wrapper around C++ game engine for performance"""
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=48, shape=(2, 6), dtype=np.int32)
        self.action_space = spaces.Discrete(6)
        
        # C++ backend for fast gameplay
        self.cpp_env = cpp_env.GameEnvCpp()
        self.max_steps = 200
        
    @property
    def board(self): return self.cpp_env.board

    @property
    def captured_seeds(self): return self.cpp_env.captured_seeds

    @property
    def current_player(self): return self.cpp_env.current_player

    @property
    def current_step(self): return self.cpp_env.current_step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        board, info = self.cpp_env.reset(seed)
        return board, info

    def step(self, action: int):
        return self.cpp_env.step(action)

    def valid_moves(self) -> list:
        return self.cpp_env.valid_moves()

    def clone(self) -> "GameEnv":
        new_env = GameEnv()
        new_env.cpp_env = self.cpp_env.clone()
        return new_env

    def get_state(self) -> dict:
        return self.cpp_env.get_state()

    def set_state(self, state: dict):
        self.cpp_env.set_state(state)
        
    def render(self): pass
    def close(self): pass