import numpy as np
import math
import pygame
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import random

class GameEnv(gym.Env):
    """2-player Mancala-style board game environment with Pygame rendering"""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(GameEnv, self).__init__()
        assert render_mode in [None, "human"]
        self.render_mode = render_mode

        if render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            display_info = pygame.display.Info()
            self.screen_width = display_info.current_w
            self.screen_height = display_info.current_h
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height)) if render_mode == "human" else None
            pygame.display.set_caption("The Game")
            self.clock = pygame.time.Clock() if render_mode == "human" else None
            self.font = pygame.font.SysFont("Arial", 50, bold=True)
            self.ui_font = pygame.font.SysFont("Arial", 28, bold=True)
            pit_spacing = self.screen_width // 8
            self.pit_centers = [
                [(pit_spacing + i * pit_spacing, self.screen_height // 4) for i in range(6)],
                [(pit_spacing + i * pit_spacing, self.screen_height * 3 // 4) for i in range(6)]
            ]
            self.pit_width = 80
            self.pit_height = 60
            self.sow_sound = self._create_sow_sound()
            self.capture_sound = self._create_capture_sound()
            self.bg_music = self._create_background_music()
            self.bg_channel = pygame.mixer.Channel(0)
            self.bg_channel.set_volume(0.2)
            self.bg_channel.play(self.bg_music, loops=-1)
        else:
            self.screen = None
            self.clock = None
            self.font = None
            self.ui_font = None
            self.pit_centers = None
            self.pit_width = None
            self.pit_height = None
            self.sow_sound = None
            self.capture_sound = None
            self.bg_music = None
            self.bg_channel = None

        self.board = np.full((2, 6), 4, dtype=np.int32)
        self.captured_seeds = np.zeros(2, dtype=np.int32)
        self.current_player = 0
        self.current_step = 0
        self.max_steps = 120
        self.episode_count = 0
        self.last_move = "Game Start"
        self.player0_streak = 0
        self.player1_streak = 0

        self.history_length = 4
        self.state_history = deque(maxlen=self.history_length)

        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(12,), dtype=np.float64),
            "history": spaces.Box(low=0, high=1, shape=(self.history_length, 12), dtype=np.float64),
            "history_players": spaces.Box(low=0, high=1, shape=(self.history_length,), dtype=np.int8),
            "captured_seeds": spaces.Box(low=0, high=48, shape=(2,), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(6)

        self.animation_states = []
        self.animation_frame = 0
        self.hand_pos = None
        self.seed_positions = []
        self.particles = []
        self.last_sow_pos = None

        if render_mode == "human":
            self.bg_color = (30, 20, 10)
            self.pit_base = (100, 70, 40)
            self.pit_inner = (60, 40, 20)
            self.seed_color = (255, 200, 100)
            self.glow_color = (0, 255, 150)
            self.quit_button_color = (150, 50, 50)
            self.quit_button_hover = (200, 80, 80)
            self.last_phase = None
            self.commentary_text = ""
            self.tts_engine = None  # Placeholder for TTS

    def get_valid_moves(self):
        return self.get_observation()["action_mask"]

    def clone(self):
        """Create a copy of this environment state"""
        env = GameEnv(render_mode=None)
        env.board = self.board.copy()
        env.captured_seeds = self.captured_seeds.copy()
        env.current_player = self.current_player
        env.current_step = self.current_step
        env.state_history = deque(self.state_history, maxlen=self.history_length)
        return env

    def set_state(self, state):
        self.board = state["unnormalised_board"].copy()
        self.action_mask = state["action_mask"].copy()
        self.current_player = state["current_player"]
        self.captured_seeds = state.get("captured_seeds", np.zeros(2, dtype=np.int32)).copy()
        if "history" in state and "history_players" in state:
            self.state_history.clear()
            for hist_board, hist_player in zip(state["history"], state["history_players"]):
                self.state_history.append((hist_board.reshape(2, 6) * 48.0, hist_player))
        else:
            if not self.state_history:
                for _ in range(self.history_length):
                    self.state_history.append((self.board.copy(), self.current_player))

    def get_legal_actions(self, obs):
        return [i for i in range(6) if obs["action_mask"][i] == 1]

    def compute_evaluation_reward(self, action, player):
        original_board = self.board.copy()
        original_captured = self.captured_seeds.copy()
        original_step = self.current_step
        original_streaks = (self.player0_streak, self.player1_streak)
        original_last_move = self.last_move

        _, reward, done, truncated, info = self.step(action)
        capture_seeds = info["seeds_captured"]
        captured_pits = info["captured_pits"]

        phase = self._get_game_phase()
        phase_potential_scale = {"opening": 0.5, "middle": 1.0, "endgame": 1.5}
        prev_potential = self._compute_potential((original_board, original_captured), player)

        capture_reward = 0
        reward = 0
        invalid_action = False
        if capture_seeds > 0:
            streak = self.player0_streak if player == 0 else self.player1_streak
            capture_reward = 0.05 * capture_seeds if phase != "endgame" else 0.1 * capture_seeds
            if captured_pits >= 2:
                capture_reward *= 1.5 ** captured_pits
            if captured_pits >= 3:
                capture_reward += 3.0
            if capture_seeds >= 5:
                capture_reward += 0.5
            capture_reward += 0.2 * streak
            if np.sum(self.board[player] > 0) == 1:
                reward += 1.0
            if np.sum(self.board[1 - player]) == 0:
                reward += 1.0

        if phase == "opening":
            if np.sum(self.board[player] > 0) >= 5:
                reward += 0.15
            if np.all(self.board[player] >= 1):
                reward += 0.1
            if np.sum(self.board[player] > 0) <= 3:
                reward -= 0.3
        elif phase == "middle":
            if np.any(self.board[player] >= 8):
                reward += 0.2
        elif phase == "endgame":
            if np.sum(self.board) < 5:
                reward += 0.5
            if np.sum(self.board[player] > 0) == 1 and np.sum(self.board[player]) == np.sum(self.board[player]):
                reward -= 0.7

        for col in range(6):
            opp_row = 1 if player == 0 else 0
            if self.board[opp_row, col] in [2, 3]:
                reward += 0.2
                break

        opponent = 1 - player
        total_seeds = np.sum(self.board)
        player_seeds = np.sum(self.board[player])
        if total_seeds > 0 and player_seeds / total_seeds > 0.5:
            reward += 2.25 if phase == "opening" else 3.3

        opp_valid_moves = np.sum(self.board[1 - player] > 0)
        if opp_valid_moves <= 2:
            reward += 1.4

        key_pits = [0, 1, 4, 5]
        player_key_seeds = sum(self.board[player, pit] for pit in key_pits)
        opponent_key_seeds = sum(self.board[opponent, pit] for pit in key_pits)
        reward += 0.05 * (player_key_seeds - opponent_key_seeds)

        if phase in ["middle", "endgame"]:
            vulnerable_pits = self._check_vulnerable_seeds(player)
            reward -= 1.2 * vulnerable_pits

        temp_board = self.board.copy()
        next_capture = 0
        for next_action in range(6):
            if temp_board[player, next_action] > 0:
                next_simulated = self._simulate_opponent_move(player, next_action, board=temp_board)
                next_capture = max(next_capture, next_simulated)
        reward += 0.15 * next_capture

        if total_seeds < 10:
            reward *= 1.5

        simulated_capture = 0
        max_opponent_capture = 0
        for opp_action in range(6):
            if self.board[opponent, opp_action] > 0:
                simulated_capture = self._simulate_opponent_move(opponent, opp_action)
                max_opponent_capture = max(max_opponent_capture, simulated_capture)
        reward -= 1.5 * max_opponent_capture

        if np.sum(self.board[player] > 0) <= 2:
            reward -= 5.0

        new_potential = self._compute_potential((self.board, self.captured_seeds), player)
        reward += phase_potential_scale[phase] * (0.99 * new_potential - prev_potential)

        done = self._check_game_over()
        winner = None
        if done:
            winner = 0 if self.captured_seeds[0] > self.captured_seeds[1] else (1 if self.captured_seeds[1] > self.captured_seeds[0] else -1)
            if winner == player:
                reward += 5.0
                lead = self.captured_seeds[player] - self.captured_seeds[1 - player]
                if lead >= 10:
                    reward += 2.0
                if self.current_step < 40:
                    reward += 0.5 * (40 - self.current_step)
                if np.sum(self.board) < 10:
                    reward += 1.5
            elif winner == 1 - player:
                reward -= 100.0

        self.board = original_board
        self.captured_seeds = original_captured
        self.current_step = original_step
        self.player0_streak, self.player1_streak = original_streaks
        self.last_move = original_last_move
        self.current_player = player

        return reward + capture_reward

    def _create_sow_sound(self):
        sample_rate = 44100
        duration = 0.1
        samples = int(sample_rate * duration)
        sound = np.zeros((samples, 2), dtype=np.int16)
        for i in range(samples):
            t = i / sample_rate
            value = int(32767 * 0.5 * math.sin(2 * math.pi * 1000 * t) * math.exp(-10 * t))
            sound[i, 0] = value
            sound[i, 1] = value
        return pygame.sndarray.make_sound(sound)

    def _create_capture_sound(self):
        sample_rate = 44100
        duration = 0.3
        samples = int(sample_rate * duration)
        sound = np.zeros((samples, 2), dtype=np.int16)
        for i in range(samples):
            t = i / sample_rate
            value = int(32767 * 0.3 * (math.sin(2 * math.pi * 2000 * t) +
                                       math.sin(2 * math.pi * 3000 * t)) * math.exp(-5 * t))
            sound[i, 0] = value
            sound[i, 1] = value
        return pygame.sndarray.make_sound(sound)

    def _create_background_music(self):
        sample_rate = 44100
        beat_duration = 0.5
        loop_duration = 2.0
        samples = int(sample_rate * loop_duration)
        sound = np.zeros((samples, 2), dtype=np.int16)
        for i in range(samples):
            t = i / sample_rate
            beat = t % beat_duration
            value = int(32767 * 0.2 * math.sin(2 * math.pi * 200 * beat) * math.exp(-10 * beat))
            sound[i, 0] = value
            sound[i, 1] = value
        return pygame.sndarray.make_sound(sound)

    def _compute_potential(self, state, player):
        board, captured_seeds = state
        opponent = 1 - player
        player_seeds = np.sum(board[player])
        opponent_seeds = np.sum(board[opponent])
        player_valid_moves = np.sum(board[player] > 0)
        opponent_valid_moves = np.sum(board[opponent] > 0)
        player_capture_potential = sum(board[opponent, col] for col in range(6) if board[opponent, col] in [2, 3])
        opponent_capture_potential = sum(board[player, col] for col in range(6) if board[player, col] in [2, 3])
        key_pits = [0, 1, 4, 5]
        player_key_seeds = sum(board[player, pit] for pit in key_pits)
        opponent_key_seeds = sum(board[opponent, pit] for pit in key_pits)
        return (captured_seeds[player] - captured_seeds[opponent]) + \
               0.1 * (player_seeds - opponent_seeds) + \
               0.05 * (player_valid_moves - opponent_valid_moves) + \
               0.2 * (player_capture_potential - opponent_capture_potential) + \
               0.1 * (player_key_seeds - opponent_key_seeds)

    def _get_game_phase(self):
        total_seeds = np.sum(self.board)
        captured = self.captured_seeds[self.current_player]
        legal_moves = len(self.get_legal_actions(self.get_observation()))
        step_opening = 20
        step_middle = 40
        seed_opening = 40
        seed_middle = 20
        capture_threshold = 20
        if (self.current_step <= step_opening and total_seeds >= seed_opening) or legal_moves >= 5:
            phase = "opening"
        elif (self.current_step > step_middle or total_seeds < seed_middle or
              captured >= capture_threshold or legal_moves <= 2):
            phase = "endgame"
        else:
            phase = "middle"
        return phase

    def _simulate_opponent_move(self, player, action, board=None):
        temp_board = self.board.copy() if board is None else board.copy()
        seeds = temp_board[player, action]
        seed_count = seeds
        temp_board[player, action] = 0
        position = (player, action)
        first_loop = True
        original_pos = position
        while seeds > 0:
            if position[0] == 0 and position[1] > 0:
                position = (0, position[1] - 1)
            elif position[0] == 0 and position[1] == 0:
                position = (1, 0)
            elif position[0] == 1 and position[1] < 5:
                position = (1, position[1] + 1)
            elif position[0] == 1 and position[1] == 5:
                position = (0, 5)
            if seed_count >= 12 and first_loop and original_pos == position:
                first_loop = False
                continue
            temp_board[position] += 1
            seeds -= 1
        capture_seeds = 0
        if (player == 1 and position[0] == 0 and temp_board[position] in [2, 3]) or \
           (player == 0 and position[0] == 1 and temp_board[position] in [2, 3]):
            sim_board, _, sim_capture_seeds, _ = self._simulate_capture(player, position, temp_board, np.zeros(2, dtype=np.int32))
            opponent_row = 1 - player
            if np.sum(sim_board[opponent_row]) > 0:
                capture_seeds = sim_capture_seeds
        return capture_seeds

    def _check_vulnerable_seeds(self, player):
        opponent = 1 - player
        vulnerable_pits = 0
        for col in range(6):
            if self.board[player, col] in [2, 3]:
                for opp_action in range(6):
                    if self.board[opponent, opp_action] > 0:
                        temp_board = self.board.copy()
                        seeds = temp_board[opponent, opp_action]
                        seed_count = seeds
                        first_loop = True
                        temp_board[opponent, opp_action] = 0
                        pos = (opponent, opp_action)
                        original_pos = pos
                        while seeds > 0:
                            if pos[0] == 0 and pos[1] > 0:
                                pos = (0, pos[1] - 1)
                            elif pos[0] == 0 and pos[1] == 0:
                                pos = (1, 0)
                            elif pos[0] == 1 and pos[1] < 5:
                                pos = (1, pos[1] + 1)
                            elif pos[0] == 1 and pos[1] == 5:
                                pos = (0, 5)
                            if seed_count >= 12 and first_loop and original_pos == pos:
                                first_loop = False
                                continue
                            temp_board[pos] += 1
                            seeds -= 1
                        if pos[0] == player and pos[1] == col and temp_board[pos] in [2, 3]:
                            can_recapture = False
                            for player_action in range(6):
                                if temp_board[player, player_action] > 0:
                                    recapture_seeds = self._simulate_opponent_move(player, player_action, temp_board)
                                    if recapture_seeds > 2:
                                        can_recapture = True
                                        break
                            if not can_recapture:
                                vulnerable_pits += 1
        return vulnerable_pits

    def _would_starve_opponent(self, player, action):
        if self.board[player, action] == 0:
            return True
        temp_board = self.board.copy()
        seeds = temp_board[player, action]
        seed_count = seeds
        temp_board[player, action] = 0
        position = (player, action)
        first_loop = True
        original_pos = position
        opponent_row = 1 - player
        while seeds > 0:
            if position[0] == 0 and position[1] > 0:
                position = (0, position[1] - 1)
            elif position[0] == 0 and position[1] == 0:
                position = (1, 0)
            elif position[0] == 1 and position[1] < 5:
                position = (1, position[1] + 1)
            elif position[0] == 1 and position[1] == 5:
                position = (0, 5)
            if seed_count >= 12 and first_loop and original_pos == position:
                first_loop = False
                continue
            temp_board[position] += 1
            seeds -= 1
        return np.sum(temp_board[opponent_row]) == 0

    def _has_non_starving_move(self, player):
        for action in range(6):
            if self.board[player, action] > 0 and not self._would_starve_opponent(player, action):
                return True
        return False

    def get_observation(self):
        board = self.board.copy() / 48.0
        action_mask = (self.board[self.current_player] > 0).astype(np.int8)

        history_boards = np.zeros((self.history_length, 12), dtype=np.float64)
        history_players = np.zeros(self.history_length, dtype=np.int8)
        for i, (hist_board, hist_player) in enumerate(reversed(self.state_history)):
            if i < self.history_length:
                history_boards[i] = hist_board.flatten() / 48.0
                history_players[i] = hist_player

        opponent_pits = self.board[1 - self.current_player]
        if self._has_non_starving_move(self.current_player) and np.sum(opponent_pits) == 0:
            for action in range(6):
                if action_mask[action] == 1 and self._would_starve_opponent(self.current_player, action):
                    action_mask[action] = 0

        return {
            "board": board.flatten(),
            "current_player": self.current_player,
            # "unnormalised_board": self.board.copy(),
            "action_mask": action_mask,
            "captured_seeds": self.captured_seeds.copy(),
            "history": history_boards,
            "history_players": history_players
        }
    
    def _get_potential_values(self, player, board):
        opponent = 1 - player

        player_seeds_on_board = np.sum(board[player])
        total_seeds_on_board = np.sum(board)
        bcp = player_seeds_on_board / (total_seeds_on_board + 1e-6) if total_seeds_on_board > 0 else 0.0

        player_legal_moves = [i for i in range(6) if board[player, i] > 0]
        ovp = float(len(player_legal_moves) + np.sum([board[player, i] for i in player_legal_moves]))

        opponent_legal_moves = [i for i in range(6) if board[opponent, i] > 0]
        osp = float(-(len(opponent_legal_moves) + np.sum([board[opponent, i] for i in opponent_legal_moves])))

        return bcp, ovp, osp

    def step(self, action):
        is_first = False
        if self.current_step == 0:
            is_first = True
        self.state_history.append((self.board.copy(), self.current_player))
        player = self.current_player

        prev_bcp, prev_ovp, prev_osp = self._get_potential_values(player, self.board)

        assert action in range(6)
        self.last_action = action
        animation_states = [(self.board.copy(), self.captured_seeds.copy(), "start", (player, action))]
        reward = 0
        capture_seeds = 0
        captured_pits = 0
        invalid_action = False

        if self.board[player, action] == 0:
            reward = -1.0
            invalid_action = True
            self.last_move = f"Player {player + 1} chose empty pit {action}"
        elif self._has_non_starving_move(player) and self._would_starve_opponent(player, action):
            reward = -1.0
            invalid_action = True
            self.last_move = f"Player {player + 1} made a starving move from pit {action}"
        else:
            seeds_to_sow = self.board[player, action]
            self.board[player, action] = 0
            animation_states.append((self.board.copy(), self.captured_seeds.copy(), "pickup", (player, action)))
            position = (player, action)
            seed_count = seeds_to_sow
            original_position = position
            first_loop = True
            while seeds_to_sow > 0:
                if position[0] == 0 and position[1] > 0:
                    position = (0, position[1] - 1)
                elif position[0] == 0 and position[1] == 0:
                    position = (1, 0)
                elif position[0] == 1 and position[1] < 5:
                    position = (1, position[1] + 1)
                elif position[0] == 1 and position[1] == 5:
                    position = (0, 5)
                if original_position == position and seed_count >= 12 and first_loop:
                    first_loop = False
                    continue
                self.board[position] += 1
                seeds_to_sow -= 1
                animation_states.append((self.board.copy(), self.captured_seeds.copy(), "sow", position))
            self.last_sow_pos = position
            if self.render_mode == "human":
                self.seed_positions = []
                for i in range(seed_count):
                    cols = max(1, int(np.ceil(np.sqrt(seed_count))))
                    rows = max(1, int(np.ceil(seed_count / cols)))
                    row = i // cols
                    col = i % cols
                    seed_x = self.pit_centers[player][action][0] - self.pit_width + (col + 0.5) * (2 * self.pit_width / cols)
                    seed_y = self.pit_centers[player][action][1] - self.pit_height + (row + 0.5) * (2 * self.pit_height / rows)
                    self.seed_positions.append((seed_x, seed_y, *self.pit_centers[player][action], 0))

            if (player == 1 and position[0] == 0 and self.board[position] in [2, 3]) or \
               (player == 0 and position[0] == 1 and self.board[position] in [2, 3]):
                temp_board, temp_captured_seeds, temp_capture_seeds, temp_captured_pits = self._simulate_capture(player, position, self.board, self.captured_seeds)
                opponent_row = 1 - player
                if np.sum(temp_board[opponent_row]) == 0:
                    capture_seeds = 0
                    captured_pits = 0
                    animation_states.append((self.board.copy(), self.captured_seeds.copy(), "no_capture", position))
                else:
                    capture_seeds, captured_pits = self._capture_seeds(player, position, animation_states)

        self.current_player = 1 - player
        self.current_step += 1
        done = self._check_game_over()
        truncated = self.current_step >= self.max_steps
    
        current_bcp, current_ovp, current_osp = self._get_potential_values(player, self.board)

        reward_bcp = current_bcp - prev_bcp
        reward_ovp = current_ovp - prev_ovp
        reward_osp = current_osp - prev_osp

        w_bcp = 0.02
        w_ovp = 0.001
        w_osp = 0.001

        # Scale capture reward to be less dominant
        scaled_capture_reward = capture_seeds / 96.0

        total_reward = (w_bcp * reward_bcp) + (w_ovp * reward_ovp) + (w_osp * reward_osp) + scaled_capture_reward

        winner = "Episode not finished!"
        is_terminal = False
        if done or truncated:
            is_terminal = True
            winner = 0 if self.captured_seeds[0] > self.captured_seeds[1] else (1 if self.captured_seeds[1] > self.captured_seeds[0] else -1)
            if winner == player:
                total_reward += 1.0 # Scaled win bonus
            elif winner == 1 - player:
                total_reward -= 1.0 # Scaled loss penalty

        # Final reward clipping for stability
        total_reward = np.clip(total_reward, -1.0, 1.0)

        self.animation_states = animation_states
        self.animation_frame = 0
        self.hand_pos = None
        if self.render_mode == "human":
            self.sow_sound.play()
            if capture_seeds > 0:
                self.capture_sound.play()

        info = {
            "seeds_captured": capture_seeds,
            "winner": winner,
            "last_move": self.last_move,
            "is_first": is_first,
            "is_terminal": is_terminal,
            "current_step": self.current_step,
            "reward": total_reward,
            "captured_pits": captured_pits,
            "action": action,
            "acting_player": player,
            "invalid_action": invalid_action
        }

        self.info_capture = info["seeds_captured"]
        self.info_action = info["action"]
        next_obs = self.get_observation()
        if not np.any(next_obs["action_mask"]):
            done = True
            self.last_move = f"Player {self.current_player + 1} has no valid moves — game ends"

        reward = total_reward

        return next_obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.board = np.full((2, 6), 4, dtype=np.int32)
        self.captured_seeds = np.zeros(2, dtype=np.int32)
        self.current_player = self.episode_count % 2
        self.episode_count += 1
        self.current_step = 0
        self.player0_streak = 0
        self.player1_streak = 0
        self.animation_states = []
        self.animation_frame = 0
        self.hand_pos = None
        self.seed_positions = []
        self.particles = []
        self.last_sow_pos = None
        self.last_move = "Game Start"

        self.state_history.clear()
        for _ in range(self.history_length):
            self.state_history.append((np.zeros((2, 6), dtype=np.int32), 0))

        if self.render_mode == "human" and (self.bg_channel is None or not self.bg_channel.get_busy()):
            self.bg_channel.play(self.bg_music, loops=-1)
        return self.get_observation(), {"is_terminal": False, "is_first": True}

    def _capture_seeds(self, player, position, animation_states):
        captured_pits = 0
        capture_seeds = 0
        if self.board[position] in [2, 3]:
            capture_seeds += self.board[position]
            self.board[position] = 0
            captured_pits += 1
            pit_number = position[1]
            if player == 1:
                while pit_number < 5 and self.board[0, pit_number + 1] in [2, 3]:
                    capture_seeds += self.board[0, pit_number + 1]
                    self.board[0, pit_number + 1] = 0
                    captured_pits += 1
                    pit_number += 1
            else:
                while pit_number > 0 and self.board[1, pit_number - 1] in [2, 3]:
                    capture_seeds += self.board[1, pit_number - 1]
                    self.board[1, pit_number - 1] = 0
                    captured_pits += 1
                    pit_number -= 1
            self.captured_seeds[player] += capture_seeds
            animation_states.append((self.board.copy(), self.captured_seeds.copy(), "capture", position))
        return capture_seeds, captured_pits

    def _simulate_capture(self, player, position, board, captured_seeds):
        temp_board = board.copy()
        temp_captured_seeds = captured_seeds.copy()
        capture_seeds = 0
        captured_pits = 0
        if temp_board[position] in [2, 3]:
            capture_seeds += temp_board[position]
            temp_board[position] = 0
            captured_pits += 1
            pit_number = position[1]
            if player == 1:
                while pit_number < 5 and temp_board[0, pit_number + 1] in [2, 3]:
                    capture_seeds += temp_board[0, pit_number + 1]
                    temp_board[0, pit_number + 1] = 0
                    captured_pits += 1
                    pit_number += 1
            else:
                while pit_number > 0 and temp_board[1, pit_number - 1] in [2, 3]:
                    capture_seeds += temp_board[1, pit_number - 1]
                    temp_board[1, pit_number - 1] = 0
                    captured_pits += 1
                    pit_number -= 1
            temp_captured_seeds[player] += capture_seeds
        return temp_board, temp_captured_seeds, capture_seeds, captured_pits

    def _check_game_over(self):
        if (np.sum(self.board[0]) == 0 and self.current_player == 0) or (np.sum(self.board[1]) == 0 and self.current_player == 1):
            return True
        if self.captured_seeds[0] >= 25 or self.captured_seeds[1] >= 25:
            return True
        return False

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.fill(self.bg_color)
        if not hasattr(self, 'bg_surface'):
            self.bg_surface = pygame.Surface((self.screen_width, self.screen_height))
            self.bg_surface.fill(self.bg_color)
            for _ in range(20):
                x1, y1 = random.randint(0, self.screen_width), random.randint(0, self.screen_height)
                x2, y2 = x1 + random.randint(-200, 200), y1 + random.randint(-200, 200)
                pygame.draw.line(self.bg_surface, (40, 30, 20), (x1, y1), (x2, y2), 1)
        self.screen.blit(self.bg_surface, (0, 0))

        board_width = self.screen_width * 0.8
        board_height = self.screen_height * 0.4
        board_x = (self.screen_width - board_width) // 2
        board_y = (self.screen_height - board_height) // 2

        board_surface = pygame.Surface((board_width, board_height), pygame.SRCALPHA)
        pygame.draw.rect(board_surface, (100, 70, 40), (0, 0, board_width, board_height), border_radius=10)
        pygame.draw.rect(board_surface, (80, 50, 30), (0, 0, board_width, board_height), 5, border_radius=10)
        self.screen.blit(board_surface, (board_x, board_y))

        pit_spacing = board_width // 7
        self.pit_width = 60
        self.pit_height = 50
        self.pit_centers = [
            [(board_x + pit_spacing + i * pit_spacing, board_y + board_height * 0.25) for i in range(6)],
            [(board_x + pit_spacing + i * pit_spacing, board_y + board_height * 0.75) for i in range(6)]
        ]

        if self.animation_states and self.animation_frame < len(self.animation_states) * 20:
            index = self.animation_frame // 20
            board, captured_seeds, action_type, pos = self.animation_states[index]
            progress = (self.animation_frame % 20) / 20

            if action_type == "pickup":
                start_x, start_y = self.pit_centers[pos[0]][pos[1]]
                seed_count = self.animation_states[index - 1][0][pos[0]][pos[1]]
                if progress == 0:
                    self.seed_positions = [(start_x + random.randint(-20, 20),
                                           start_y + random.randint(-20, 20),
                                           start_x, start_y, 0) for _ in range(seed_count)]
                for i, (sx, sy, tx, ty, p) in enumerate(self.seed_positions):
                    new_p = min(1, progress)
                    self.seed_positions[i] = (sx + (tx - sx) * new_p, sy + (ty - sy) * new_p, tx, ty, new_p)

            elif action_type == "sow" and index > 1:
                prev_pos = self.animation_states[index - 1][3]
                start_x, start_y = self.pit_centers[prev_pos[0]][prev_pos[1]]
                end_x, end_y = self.pit_centers[pos[0]][pos[1]]
                for i, (sx, sy, tx, ty, p) in enumerate(self.seed_positions):
                    new_p = min(1, p + progress / len(self.seed_positions))
                    self.seed_positions[i] = (sx + (end_x - sx) * new_p,
                                             sy + (end_y - sy) * new_p, end_x, end_y, new_p)

            elif action_type == "capture":
                prev_board = self.animation_states[index - 1][0]
                curr_board = board
                captured_seeds_count = int(np.sum(prev_board - curr_board))
                target_x = 120 if pos[0] == 0 else self.screen_width - 120
                target_y = 60
                start_x, start_y = self.pit_centers[pos[0]][pos[1]]
                self.particles.extend([(start_x + random.randint(-30, 30),
                                       start_y + random.randint(-30, 30),
                                       random.randint(5, 15), 30, random.uniform(-1, 1), -2)
                                      for _ in range(captured_seeds_count * 3)])

            self.animation_frame += 1
            if index == len(self.animation_states) - 1 and progress > 0.9:
                self.seed_positions = []
        else:
            board = self.board
            captured_seeds = self.captured_seeds

        for row in range(2):
            for col in range(6):
                x, y = self.pit_centers[row][col]
                seed_count = board[row, col]

                shadow = pygame.Surface((self.pit_width * 2 + 20, self.pit_height * 2 + 20), pygame.SRCALPHA)
                pygame.draw.ellipse(shadow, (0, 0, 0, 80), (0, 0, self.pit_width * 2 + 20, self.pit_height * 2 + 20))
                self.screen.blit(shadow, (x - self.pit_width - 10, y - self.pit_height + 10))

                pygame.draw.ellipse(self.screen, self.pit_base,
                                    (x - self.pit_width, y - self.pit_height, self.pit_width * 2, self.pit_height * 2))
                pygame.draw.ellipse(self.screen, self.pit_inner,
                                    (x - self.pit_width * 0.85, y - self.pit_height * 0.85,
                                     self.pit_width * 1.7, self.pit_height * 1.7))
                pygame.draw.ellipse(self.screen, (120, 90, 60),
                                    (x - self.pit_width * 0.95, y - self.pit_height * 0.95,
                                     self.pit_width * 1.9, self.pit_height * 1.9), 3)

                if row == self.current_player and seed_count > 0:
                    glow = pygame.Surface((self.pit_width * 2 + 10, self.pit_height * 2 + 10), pygame.SRCALPHA)
                    pygame.draw.ellipse(glow, (*self.glow_color, 60),
                                        (0, 0, self.pit_width * 2 + 10, self.pit_height * 2 + 10), 6)
                    self.screen.blit(glow, (x - self.pit_width - 5, y - self.pit_height - 5))

                if not hasattr(self, 'static_seed_positions'):
                    self.static_seed_positions = {}
                if len(self.static_seed_positions.get((row, col), [])) != seed_count:
                    self.static_seed_positions[(row, col)] = []
                    for i in range(seed_count):
                        offset_x = random.randint(-int(self.pit_width * 0.6), int(self.pit_width * 0.6))
                        offset_y = random.randint(-int(self.pit_height * 0.6), int(self.pit_height * 0.6))
                        scale = max(0.3, 1 - seed_count * 0.05)
                        seed_x = x + int(offset_x * scale)
                        seed_y = y + int(offset_y * scale)
                        self.static_seed_positions[(row, col)].append((seed_x, seed_y))
                for seed_x, seed_y in self.static_seed_positions.get((row, col), []):
                    pygame.draw.circle(self.screen, self.seed_color, (seed_x, seed_y), 8)
                    pygame.draw.circle(self.screen, (255, 255, 255, 100), (seed_x - 2, seed_y - 2), 3)

                text = self.font.render(str(seed_count), True, (255, 255, 255))
                text_x = x - text.get_width() // 2
                text_y = y + self.pit_height + 5
                self.screen.blit(text, (text_x, text_y))

        for sx, sy, tx, ty, p in self.seed_positions:
            pygame.draw.circle(self.screen, self.seed_color, (int(sx), int(sy)), 8)
            pygame.draw.circle(self.screen, (255, 255, 255, 100), (int(sx) - 2, int(sy) - 2), 3)

        new_particles = []
        for px, py, size, life, vx, vy in self.particles:
            if life > 0:
                pygame.draw.circle(self.screen, (255, 255, 150), (int(px), int(py)), int(size))
                new_particles.append((px + vx, py + vy, size * 0.95, life - 1, vx, vy + 0.1))
        self.particles = new_particles

        score_surface = pygame.Surface((150, 50), pygame.SRCALPHA)
        pygame.draw.rect(score_surface, (80, 60, 40, 200), (0, 0, 150, 50), border_radius=5)
        color = self.glow_color if self.current_player == 0 else (255, 255, 255)
        text = self.ui_font.render(f"P1: {captured_seeds[0]}", True, color)
        score_surface.blit(text, (75 - text.get_width() // 2, 25 - text.get_height() // 2))
        self.screen.blit(score_surface, (20, 20))

        score_surface = pygame.Surface((150, 50), pygame.SRCALPHA)
        pygame.draw.rect(score_surface, (80, 60, 40, 200), (0, 0, 150, 50), border_radius=5)
        color = self.glow_color if self.current_player == 1 else (255, 255, 255)
        text = self.ui_font.render(f"P2: {captured_seeds[1]}", True, color)
        score_surface.blit(text, (75 - text.get_width() // 2, 25 - text.get_height() // 2))
        self.screen.blit(score_surface, (self.screen_width - 170, 20))

        turn_surface = pygame.Surface((200, 50), pygame.SRCALPHA)
        pygame.draw.rect(turn_surface, (80, 60, 40, 200), (0, 0, 200, 50), border_radius=5)
        text = self.ui_font.render(f"Player {self.current_player + 1}'s Turn", True, (255, 255, 255))
        turn_surface.blit(text, (100 - text.get_width() // 2, 25 - text.get_height() // 2))
        self.screen.blit(turn_surface, (self.screen_width // 2 - 100, self.screen_height - 100))

        move_surface = pygame.Surface((400, 50), pygame.SRCALPHA)
        pygame.draw.rect(move_surface, (200, 180, 150, 200), (0, 0, 400, 50), border_radius=5)
        text = self.ui_font.render(self.last_move, True, (50, 30, 20))
        move_surface.blit(text, (200 - text.get_width() // 2, 25 - text.get_height() // 2))
        self.screen.blit(move_surface, (self.screen_width // 2 - 200, self.screen_height - 50))

        quit_rect = pygame.Rect(self.screen_width - 120, self.screen_height - 50, 100, 40)
        mouse_pos = pygame.mouse.get_pos()
        button_color = self.quit_button_hover if quit_rect.collidepoint(mouse_pos) else self.quit_button_color
        pygame.draw.rect(self.screen, button_color, quit_rect, border_radius=5)
        pygame.draw.rect(self.screen, (255, 255, 255), quit_rect, 2, border_radius=5)
        quit_text = self.ui_font.render("Quit", True, (255, 255, 255))
        self.screen.blit(quit_text, (quit_rect.centerx - quit_text.get_width() // 2, quit_rect.centery - quit_text.get_height() // 2))

        if self._check_game_over():
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            winner = "P1" if self.captured_seeds[0] > self.captured_seeds[1] else "P2" if self.captured_seeds[1] > self.captured_seeds[0] else "Tie"
            banner = pygame.Surface((400, 200), pygame.SRCALPHA)
            pygame.draw.rect(banner, (80, 60, 40, 220), (0, 0, 400, 200), border_radius=10)
            pygame.draw.rect(banner, (120, 90, 60), (0, 0, 400, 200), 5, border_radius=10)
            texts = [
                self.font.render("Game Over", True, (255, 255, 255)),
                self.ui_font.render(f"Winner: {winner}", True, self.glow_color),
                self.ui_font.render(f"P1: {captured_seeds[0]}  P2: {captured_seeds[1]}", True, (255, 255, 255))
            ]
            y_offset = 50
            for text in texts:
                banner.blit(text, (200 - text.get_width() // 2, y_offset))
                y_offset += 50
            self.screen.blit(banner, (self.screen_width // 2 - 200, self.screen_height // 2 - 100))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human" and self.screen is not None:
            if self.bg_channel:
                self.bg_channel.stop()
            pygame.quit()
        self.screen = None
        self.bg_channel = None