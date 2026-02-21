import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from GameEnv import GameEnv

@dataclass
class GameConfig:
    """Model configuration"""
    num_holes   : int   = 12
    holes_side  : int   = 6
    embed_vocab : int   = 60
    ctx_dim     : int   = 3
    embed_dim   : int   = 128
    num_heads   : int   = 4
    num_layers  : int   = 4
    window_size : int   = 6
    mlp_ratio   : float = 4.0
    dropout     : float = 0.1

class SwinBlock(nn.Module):
    """Shifted window attention"""
    def __init__(self, cfg: GameConfig, shift: bool):
        super().__init__()
        self.shift = shift
        self.window_size = cfg.window_size
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn  = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        mlp_hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, mlp_hidden),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(mlp_hidden, cfg.embed_dim), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        W = self.window_size
        residual = x
        x = self.norm1(x)
        if self.shift: 
            x = torch.roll(x, shifts=-(W // 2), dims=1)
        
        x_win = x.reshape(B * (L // W), W, D) 
        out, _ = self.attn(x_win, x_win, x_win)
        out = out.reshape(B, L, D) 

        if self.shift: 
            out = torch.roll(out, shifts=(W // 2), dims=1)
        x = residual + out
        return x + self.mlp(self.norm2(x))

class GameNet(nn.Module):
    """The game AI - 4 output heads"""
    def __init__(self, cfg: GameConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim
        self.hole_embed = nn.Embedding(cfg.embed_vocab, D)
        self.turn_embed = nn.Embedding(2, D)
        self.ctx_proj   = nn.Sequential(nn.Linear(cfg.ctx_dim, D), nn.GELU())
        self.pos = nn.Parameter(torch.randn(1, cfg.num_holes, D) * 0.02)
        self.blocks = nn.ModuleList([SwinBlock(cfg, shift=(i % 2 == 1)) for i in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(D)

        self.policy = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, cfg.holes_side))
        self.value = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1), nn.Tanh())
        self.capture = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 1))
        self.next_state = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Linear(128, cfg.num_holes), nn.Sigmoid())

    def forward(self, board, turn, context):
        x = self.hole_embed(board) + self.pos
        x = x + self.turn_embed(turn).unsqueeze(1)
        x = x + self.ctx_proj(context).unsqueeze(1)
        for block in self.blocks: 
            x = block(x)
        x_ctx = self.norm(x).mean(dim=1)
        return {
            "policy": F.log_softmax(self.policy(x_ctx), dim=-1),
            "value": self.value(x_ctx)
        }

def get_player_move(env):
    """Ask human player for their move"""
    valid_moves = [i for i in range(6) if env.get_valid_moves()[i] == 1]
    if not valid_moves:
        return 0
    print(f"Valid moves: {valid_moves}")
    while True:
        try:
            move = int(input(f"Enter your move (0-5): "))
            if move in valid_moves:
                return move
            else:
                print(f"Pit {move} is not valid (empty or would starve)")
        except Exception as e:
            print("Please enter a number between 0 and 5")

def play_game(model_path, human_first=True, render=True):
    """Play a game of Ayo against the AI"""
    DEVICE = torch.device("cpu")
    
    # init environment
    env = GameEnv(render_mode="human" if render else None)
    
    # load model
    print(f"Loading model from {model_path}...")
    model = GameNet(GameConfig()).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    env.reset()
    done = False
    truncated = False
    
    human_player = 0 if human_first else 1
    agent_player = 1 - human_player
    
    while not (done or truncated):
        if render:
            env.render()
            
        print("\n" + "="*40)
        print("CURRENT BOARD:")
        print(env.board)
        print(f"Captured: You ({env.captured_seeds[human_player]}) | AI ({env.captured_seeds[agent_player]})")
        print("="*40)
        
        current_p = env.current_player
        valid_moves = [i for i in range(6) if env.get_valid_moves()[i] == 1]

        if not valid_moves:
            print(f"Player {current_p + 1} has no valid moves!")
            break

        if current_p == human_player:
            print(f"--- YOUR TURN ---")
            action = get_player_move(env)
        else:
            print(f"--- AI IS THINKING ---")
            time.sleep(1.0)
            
            # format inputs for the model
            board_flat = env.board.flatten().astype(np.int64)
            b_tensor = torch.tensor([board_flat], dtype=torch.long).to(DEVICE)
            t_tensor = torch.tensor([current_p], dtype=torch.long).to(DEVICE)
            
            c_tensor = torch.tensor([[
                env.captured_seeds[0] / 48.0, 
                env.captured_seeds[1] / 48.0, 
                env.current_step / 200.0
            ]], dtype=torch.float32).to(DEVICE)
            
            # get AI prediction
            with torch.no_grad():
                preds = model(b_tensor, t_tensor, c_tensor)
                probs = torch.exp(preds['policy']).squeeze().numpy()
                value = preds['value'].item()
            
            # pick the best legal move
            best_p = -1.0
            best_move = valid_moves[0]
            for m in valid_moves:
                if probs[m] > best_p:
                    best_p = probs[m]
                    best_move = m
                    
            action = best_move
            
            print(f"-> Win probability: {value:+.3f}")
            print(f"-> Confidence: {best_p * 100:.1f}%")
            print(f"-> AI chose pit {action}")

        # Execute move in Pygame Env
        obs, reward, done, truncated, info = env.step(action)
        
        if "seeds_captured" in info and info["seeds_captured"] > 0:
            print(f"*** {info['seeds_captured']} seeds captured! ***")

    # Game Over Sequence
    if render:
        env.render()
        
    print("\n" + "="*40)
    print("GAME OVER")
    print(f"Your pits: {env.board[0].tolist()}")
    print(f"AI pits: {env.board[1].tolist()}")
    print(f"Final Score - You: {env.captured_seeds[human_player]} | AI: {env.captured_seeds[agent_player]}")
    print("="*40)
    
    if env.captured_seeds[human_player] > env.captured_seeds[agent_player]:
        print("You won!")
    elif env.captured_seeds[human_player] < env.captured_seeds[agent_player]:
        print("AI won.")
    else:
        print("Draw!")
        
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against the game AI")
    parser.add_argument("--model-path", type=str, default="ayo_best_model.pth", help="Path to model file")
    parser.add_argument("--human-second", action="store_true", help="Let AI play first")
    parser.add_argument("--no-render", action="store_true", help="Disable graphics")
    args = parser.parse_args()
    
    play_game(
        model_path=args.model_path, 
        human_first=not args.human_second, 
        render=not args.no_render
    )