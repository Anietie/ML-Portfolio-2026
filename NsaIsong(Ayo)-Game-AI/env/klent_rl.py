"""
Reinforcement learning trainer with population-based training
Uses PPO with KL divergence penalty to the teacher model
Trains multiple agents simultaneously against an opponent pool
Includes NaN recovery for stability during high-entropy phases
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, defaultdict
from dataclasses import dataclass, field
import time
from tqdm.auto import tqdm

# Path configuration
TEACHER_PATH   = "ayo_best_model.pth"
OUTPUT_DIR     = "./klent"
ENV_MODULE     = "Phase_5_GameEnv" 

# Training settings
POPULATION_SIZE   = 4        
TOTAL_GAMES       = 50_000
EVAL_INTERVAL     = 100      
SAVE_INTERVAL     = 2000      

PPO_EPOCHS        = 4        
PPO_CLIP          = 0.2      
GAE_LAMBDA        = 0.95     
GAMMA             = 1.0      
VALUE_COEF        = 0.5      

BETA_START        = 0.3      
ALPHA_START       = 0.01     
LR_START          = 3e-5     

BETA_FINAL        = 0.05
ALPHA_FINAL       = 0.001
ANNEAL_GAMES      = 15_000   

RECENT_POOL_SIZE  = 5        
HISTORY_POOL_SIZE = 20       
SNAPSHOT_INTERVAL = 50       

P_RECENT_SELF     = 0.50
P_HISTORICAL      = 0.30
P_TEACHER         = 0.20

PBT_EXPLOIT_FRAC  = 0.25     
PERTURB_FACTOR    = 0.2      

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from play_game import GameNet, GameConfig

class EnvWrapper:
    """Wraps the game environment to produce model-ready tensors"""
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs, info = self.env.reset()
        return self._parse_obs(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        parsed = self._parse_obs(obs, info)
        return parsed, reward, done, info

    def _parse_obs(self, obs, info):
        if isinstance(obs, np.ndarray) and obs.shape == (12,):
            board = obs.astype(np.int64)
        elif isinstance(obs, np.ndarray) and obs.shape == (2, 6):
            board = obs.flatten().astype(np.int64)
        else:
            board_raw = info.get("board", np.zeros(12, dtype=np.int64))
            board = np.array(board_raw).flatten().astype(np.int64)

        turn     = int(info.get("current_player", 0))
        score_0  = float(info.get("score_0", 0)) / 48.0
        score_1  = float(info.get("score_1", 0)) / 48.0
        ply      = float(info.get("ply", 0)) / 200.0

        return {
            "board"  : torch.tensor(board, dtype=torch.long),
            "turn"   : torch.tensor(turn,  dtype=torch.long),
            "context": torch.tensor([score_0, score_1, ply], dtype=torch.float32),
        }

    def legal_moves(self):
        return self.env.valid_moves() if hasattr(self.env, 'valid_moves') \
               else list(range(6))

    def clone(self):
        return EnvWrapper(copy.deepcopy(self.env))

@torch.no_grad()
def model_act(model, obs, legal_moves, device):
    board   = obs["board"].unsqueeze(0).to(device)
    turn    = obs["turn"].unsqueeze(0).to(device)
    context = obs["context"].unsqueeze(0).to(device)

    out = model(board, turn, context)
    log_probs = out["policy"][0]   
    value     = out["value"][0, 0].item()

    if torch.isnan(log_probs).any():
        log_probs = torch.zeros_like(log_probs)

    mask = torch.full((6,), float('-inf'), device=device)
    for m in legal_moves:
        mask[m] = 0.0
        
    masked_log_probs = log_probs + mask
    
    valid_max = masked_log_probs[legal_moves].max()
    probs = torch.exp(masked_log_probs - valid_max)
    probs = probs / (probs.sum() + 1e-10)    

    if torch.isnan(probs).any() or probs.sum() <= 0:
        probs = torch.zeros(6, device=device)
        for m in legal_moves:
            probs[m] = 1.0 / len(legal_moves)

    action   = torch.multinomial(probs, 1).item()
    log_prob = masked_log_probs[action].item()

    return action, log_prob, value, log_probs.cpu()

class OpponentPool:
    def __init__(self, teacher_model, device):
        self.device          = device
        self.teacher         = teacher_model   
        self.recent          = deque(maxlen=RECENT_POOL_SIZE)
        self.historical      = deque(maxlen=HISTORY_POOL_SIZE)
        self._snapshot_count = 0

    def add_snapshot(self, model):
        snap = copy.deepcopy(model)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        snap = snap.to(self.device)

        self._snapshot_count += 1
        if self._snapshot_count % 5 == 0:
            self.historical.append(snap)
        self.recent.append(snap)

    def sample_opponent(self):
        r = random.random()
        if r < P_RECENT_SELF and self.recent:
            weights = [2**(i) for i in range(len(self.recent))]
            total   = sum(weights)
            weights = [w/total for w in weights]
            return random.choices(list(self.recent), weights=weights, k=1)[0]
        elif r < P_RECENT_SELF + P_HISTORICAL and self.historical:
            return random.choice(list(self.historical))
        else:
            return self.teacher

@dataclass
class Trajectory:
    observations : list = field(default_factory=list)  
    actions      : list = field(default_factory=list)  
    log_probs    : list = field(default_factory=list)  
    values       : list = field(default_factory=list)  
    rewards      : list = field(default_factory=list)  
    log_probs_all: list = field(default_factory=list)  

    def add(self, obs, action, log_prob, value, reward, log_probs_all):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.log_probs_all.append(log_probs_all)

    def __len__(self):
        return len(self.actions)

def compute_gae(rewards, values, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae        = 0.0
    values_ext = values + [last_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t+1] - values_ext[t]
        gae   = delta + gamma * lam * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return returns, advantages

class ELOTracker:
    def __init__(self, n_agents, k=32, initial=1500):
        self.ratings = {i: float(initial) for i in range(n_agents)}
        self.k       = k

    def update(self, agent_id, opponent_id, agent_won, draw=False):
        ra = self.ratings[agent_id]
        rb = self.ratings.get(opponent_id, 1500.0)

        ea = 1.0 / (1.0 + 10**((rb - ra) / 400.0))
        eb = 1.0 - ea

        if draw:
            sa, sb = 0.5, 0.5
        elif agent_won:
            sa, sb = 1.0, 0.0
        else:
            sa, sb = 0.0, 1.0

        self.ratings[agent_id]   = ra + self.k * (sa - ea)
        if opponent_id in self.ratings:
            self.ratings[opponent_id] = rb + self.k * (sb - eb)

    def ranking(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])

class KLENTAgent:
    def __init__(self, agent_id, teacher_model, device):
        self.id      = agent_id
        self.device  = device

        self.teacher_model = copy.deepcopy(teacher_model).to(device)
        self.teacher_model.eval()
        
        self.model = copy.deepcopy(self.teacher_model)
        self.model.train()

        # Save a clean copy to revert to if NaN corruption occurs
        self.safe_state = copy.deepcopy(self.model.state_dict())

        for p in self.model.parameters():
            p.requires_grad_(True)

        self.lr      = LR_START * (0.5 + random.random())  
        self.beta    = BETA_START
        self.alpha   = ALPHA_START

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )

        self.opponent_pool = OpponentPool(self.teacher_model, device)
        self.games_played  = 0
        self.wins = self.losses = self.draws = 0

    def anneal(self, games_done):
        frac      = min(1.0, games_done / ANNEAL_GAMES)
        self.beta = BETA_START  + frac * (BETA_FINAL  - BETA_START)
        self.alpha= ALPHA_START + frac * (ALPHA_FINAL - ALPHA_START)

    def win_rate(self):
        total = self.wins + self.losses + self.draws
        return self.wins / max(1, total)

    def snapshot(self):
        self.opponent_pool.add_snapshot(self.model)

    def copy_from(self, other: "KLENTAgent"):
        self.model.load_state_dict(copy.deepcopy(other.model.state_dict()))
        self.safe_state = copy.deepcopy(self.model.state_dict()) # Update safe state
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )

    def perturb_hyperparams(self):
        factor      = 1.0 + random.uniform(-PERTURB_FACTOR, PERTURB_FACTOR)
        self.lr     = float(np.clip(self.lr * factor, 1e-6, 1e-3))
        factor      = 1.0 + random.uniform(-PERTURB_FACTOR, PERTURB_FACTOR)
        self.beta   = float(np.clip(self.beta * factor, 0.01, 1.0))
        factor      = 1.0 + random.uniform(-PERTURB_FACTOR, PERTURB_FACTOR)
        self.alpha  = float(np.clip(self.alpha * factor, 0.0001, 0.1))
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.lr

    def update(self, trajectory, teacher_model):
        if len(trajectory) == 0:
            return {}

        T = len(trajectory)

        boards   = torch.stack([o["board"]   for o in trajectory.observations]).to(self.device)
        turns    = torch.stack([o["turn"]    for o in trajectory.observations]).to(self.device)
        contexts = torch.stack([o["context"] for o in trajectory.observations]).to(self.device)
        actions  = torch.tensor(trajectory.actions,   dtype=torch.long).to(self.device)
        old_lps  = torch.tensor(trajectory.log_probs, dtype=torch.float32).to(self.device)

        returns, advantages = compute_gae(
            trajectory.rewards, trajectory.values, last_value=0.0
        )
        returns     = torch.tensor(returns,     dtype=torch.float32).to(self.device)
        advantages  = torch.tensor(advantages,  dtype=torch.float32).to(self.device)
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            teacher_out = teacher_model(boards, turns, contexts)
            teacher_probs = torch.exp(teacher_out["policy"])

        indices     = list(range(T))
        loss_stats  = defaultdict(float)

        for _ in range(PPO_EPOCHS):
            random.shuffle(indices)
            idx = torch.tensor(indices, device=self.device)

            out        = self.model(boards[idx], turns[idx], contexts[idx])
            log_probs_all = out["policy"]          
            values_pred   = out["value"].squeeze(-1)  
            
            # Check for NaN values and recover if needed
            if torch.isnan(log_probs_all).any():
                tqdm.write(f"WARNING: NaN detected in Agent {self.id}. Reverting to last safe state and lowering LR.")
                self.model.load_state_dict(self.safe_state)
                self.lr *= 0.5 
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr
                return {} # Abort update for this step

            new_lps    = log_probs_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
            old_lps_b  = old_lps[idx]

            ratio      = torch.exp(new_lps - old_lps_b)
            adv_b      = advantages[idx]
            surr1      = ratio * adv_b
            surr2      = torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP) * adv_b
            policy_loss= -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_pred, returns[idx])

            dist = torch.distributions.Categorical(logits=log_probs_all)
            entropy = dist.entropy().mean()

            kl_loss    = F.kl_div(
                log_probs_all,               
                teacher_probs[idx],          
                reduction="batchmean"
            )

            total_loss = (
                policy_loss
                + VALUE_COEF * value_loss
                - self.alpha  * entropy
                + self.beta   * kl_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # If update succeeded without NaNs, save this as the new safe state
            self.safe_state = copy.deepcopy(self.model.state_dict())

            loss_stats["policy"] += policy_loss.item()
            loss_stats["value"]  += value_loss.item()
            loss_stats["entropy"]+= entropy.item()
            loss_stats["kl"]     += kl_loss.item()
            loss_stats["total"]  += total_loss.item()

        for k in loss_stats:
            loss_stats[k] /= PPO_EPOCHS

        return dict(loss_stats)

def run_game(agent, opponent_model, env_factory, device):
    env  = EnvWrapper(env_factory())
    obs, info = env.env.reset(), {}
    obs  = env._parse_obs(obs if isinstance(obs, np.ndarray) else obs[0],
                          info if info else {})
    traj = Trajectory()
    done = False

    while not done:
        legal = env.legal_moves()
        if not legal:
            break

        current_player = obs["turn"].item()

        if current_player == 0:
            action, log_prob, value, lp_all = model_act(
                agent.model, obs, legal, device
            )
            next_obs_data, env_reward, done, info = env.step(action)
            traj.add(obs, action, log_prob, value, 0.0, lp_all)
        else:
            action, _, _, _ = model_act(opponent_model, obs, legal, device)
            next_obs_data, env_reward, done, info = env.step(action)

        obs = next_obs_data if not done else obs

    score_0 = float(info.get("score_0", 0))
    score_1 = float(info.get("score_1", 0))
    terminal_reward = (score_0 - score_1) / 48.0

    if len(traj) > 0:
        traj.rewards[-1] = terminal_reward

    if score_0 > score_1:   outcome = 1.0
    elif score_0 < score_1: outcome = -1.0
    else:                    outcome = 0.0

    return traj, outcome

class PBTController:
    def __init__(self, agents, elo_tracker):
        self.agents  = agents
        self.elo     = elo_tracker

    def step(self):
            ranking = self.elo.ranking()   
            n       = len(ranking)
            n_copy  = max(1, int(n * PBT_EXPLOIT_FRAC))

            top_ids    = [r[0] for r in ranking[:n_copy]]
            bottom_ids = [r[0] for r in ranking[-n_copy:]]

            elo_dict = dict(ranking)

            for b_id in bottom_ids:
                donor_id = random.choice(top_ids)
                self.agents[b_id].copy_from(self.agents[donor_id])
                tqdm.write(f"  PBT: Agent {b_id} (ELO {elo_dict[b_id]:.0f}) "
                        f"copies Agent {donor_id} (ELO {elo_dict[donor_id]:.0f})")

            for agent in self.agents.values():
                agent.perturb_hyperparams()

            tqdm.write(f"  PBT: ELO ranking: "
                    + " | ".join(f"A{i}={e:.0f}" for i,e in ranking))

def train():
    import importlib

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading teacher model from {TEACHER_PATH}...")
    teacher = GameNet(GameConfig()).to(DEVICE)
    state   = torch.load(TEACHER_PATH, map_location=DEVICE)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("  Teacher loaded and frozen.")

    try:
        env_mod = importlib.import_module(ENV_MODULE)
        GameEnvClass = env_mod.GameEnv
    except ImportError:
        print(f"WARNING: Could not import {ENV_MODULE}.GameEnv")
        print("  Using placeholder. Replace with your actual environment.")
        from gymnasium import Env
        class GameEnvClass(Env):
            def reset(self): return np.zeros(12), {}
            def step(self, a):
                return np.zeros(12), 0, True, False, {"score_0":0,"score_1":0,"ply":0}
            def valid_moves(self): return list(range(6))

    env_factory = GameEnvClass

    agents = {
        i: KLENTAgent(i, teacher, DEVICE)
        for i in range(POPULATION_SIZE)
    }
    elo     = ELOTracker(POPULATION_SIZE)
    pbt     = PBTController(agents, elo)

    best_elo      = -float("inf")
    best_agent_id = 0

    print(f"\nStarting KLENT RL | Population: {POPULATION_SIZE} | "
          f"Target games/agent: {TOTAL_GAMES} | Device: {DEVICE}\n")

    t0 = time.time()
    
    # Main training loop - sequential processing for stability
    for game_num in tqdm(range(1, TOTAL_GAMES + 1), desc="Training agents"):
        
        for agent_id, agent in agents.items():
            agent.anneal(agent.games_played)
            opponent = agent.opponent_pool.sample_opponent()
            
            traj, outcome = run_game(agent, opponent, env_factory, agent.device)

            agent.model.train()
            loss_info = agent.update(traj, agent.teacher_model)
            agent.model.eval()

            agent.games_played += 1
            if outcome > 0:    agent.wins   += 1
            elif outcome < 0:  agent.losses += 1
            else:              agent.draws  += 1

            elo.update(agent_id, -1, outcome > 0, outcome == 0)
            
            if agent.games_played % SNAPSHOT_INTERVAL == 0:
                agent.snapshot()

        ranking = elo.ranking()
        top_aid, top_elo_val = ranking[0]

        if game_num % 50 == 0:
            elapsed = time.time() - t0
            tqdm.write(f"\n[Game {game_num}] Top Agent: {top_aid} | Top ELO: {top_elo_val:.0f} | Time: {elapsed:.0f}s")
            a = agents[top_aid]
            tqdm.write(f"  -> Leader Stats | W/L/D: {a.wins}/{a.losses}/{a.draws} | WR: {a.win_rate():.1%} | LR: {a.lr:.2e}")

        if game_num % EVAL_INTERVAL == 0:
            tqdm.write(f"\n--- PBT Step triggered ---")
            pbt.step()

        if game_num % SAVE_INTERVAL == 0:
            for agent_id, agent in agents.items():
                path = os.path.join(OUTPUT_DIR, f"agent_{agent_id}_game_{game_num}.pth")
                torch.save({
                    "game":            game_num,
                    "model_state":     agent.model.state_dict(),
                    "optimizer_state": agent.optimizer.state_dict(),
                    "beta":            agent.beta,
                    "alpha":           agent.alpha,
                    "lr":              agent.lr,
                    "elo":             elo.ratings[agent_id],
                }, path)

            if top_elo_val > best_elo:
                best_elo      = top_elo_val
                best_agent_id = top_aid
                best_path     = os.path.join(OUTPUT_DIR, "klent_best.pth")
                torch.save(agents[top_aid].model.state_dict(), best_path)
                tqdm.write(f"--> New best model saved (ELO {best_elo:.0f})")

if __name__ == "__main__":
    train()