"""
Model architecture for the game AI
4 output heads: policy, value, capture prediction, and next state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class GameConfig:
    """Model and training configuration"""
    num_holes   : int   = 12
    holes_side  : int   = 6
    embed_vocab : int   = 60  # max seeds per hole is 52, so 60 is safe
    ctx_dim     : int   = 3   # context: [score_0/48, score_1/48, ply/200]
    # Transformer settings
    embed_dim   : int   = 128
    num_heads   : int   = 4
    num_layers  : int   = 4
    window_size : int   = 6
    mlp_ratio   : float = 4.0
    dropout     : float = 0.1


class SwinBlock(nn.Module):
    """Shifted window attention block with circular board wraparound"""
    def __init__(self, cfg: GameConfig, shift: bool):
        super().__init__()
        self.shift       = shift
        self.window_size = cfg.window_size

        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn  = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads,
            dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

        mlp_hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(mlp_hidden, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        W = self.window_size

        # apply shift if this is an odd layer
        residual = x
        x = self.norm1(x)
        if self.shift:
            x = torch.roll(x, shifts=-(W // 2), dims=1)

        # split into windows and apply attention
        num_windows = L // W
        x_win = x.reshape(B * num_windows, W, D) 
        
        out, _ = self.attn(x_win, x_win, x_win)
        out = out.reshape(B, L, D)

        if self.shift:
            out = torch.roll(out, shifts=(W // 2), dims=1)

        x = residual + out
        x = x + self.mlp(self.norm2(x))
        return x


class Backbone(nn.Module):
    """Main encoder using stacked swin blocks"""
    def __init__(self, cfg: GameConfig):
        super().__init__()
        D = cfg.embed_dim

        # input embeddings
        self.hole_embed = nn.Embedding(cfg.embed_vocab, D)
        self.turn_embed = nn.Embedding(2, D)
        self.ctx_proj   = nn.Sequential(
            nn.Linear(cfg.ctx_dim, D),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.randn(1, cfg.num_holes, D) * 0.02)

        # stack of swin blocks
        self.blocks = nn.ModuleList([
            SwinBlock(cfg, shift=(i % 2 == 1))
            for i in range(cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(D)

    def forward(self,
                board  : torch.Tensor,
                turn   : torch.Tensor,
                context: torch.Tensor,
                ) -> torch.Tensor:
        # embed inputs and fuse with position/turn/context
        x = self.hole_embed(board) + self.pos
        x = x + self.turn_embed(turn).unsqueeze(1)
        x = x + self.ctx_proj(context).unsqueeze(1)

        # pass through transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x.mean(dim=1)   # global average to single vector


class GameNet(nn.Module):
    """Full network with 4 output heads"""
    def __init__(self, cfg: GameConfig = None):
        super().__init__()
        self.cfg      = cfg or GameConfig()
        self.backbone = Backbone(self.cfg)
        D = self.cfg.embed_dim

        # policy head: which move is best (6 possible actions)
        self.policy = nn.Sequential(
            nn.Linear(D, 64), nn.GELU(),
            nn.Linear(64, self.cfg.holes_side),
        )

        # value head: who's winning (-1 to 1)
        self.value = nn.Sequential(
            nn.Linear(D, 64), nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # capture predictor: does the best move capture?
        self.capture = nn.Sequential(
            nn.Linear(D, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

        # next state: what's the board after the best move?
        self.next_state = nn.Sequential(
            nn.Linear(D, 128), nn.GELU(),
            nn.Linear(128, self.cfg.num_holes),
            nn.Sigmoid(),
        )

    def forward(self,
                board  : torch.Tensor,
                turn   : torch.Tensor,
                context: torch.Tensor,
                ) -> dict[str, torch.Tensor]:
        x = self.backbone(board, turn, context)
        return {
            "policy"    : F.log_softmax(self.policy(x), dim=-1),
            "value"     : self.value(x),
            "capture"   : self.capture(x),
            "next_state": self.next_state(x),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GameLoss(nn.Module):
    """Combined loss for all 4 heads"""
    def __init__(self,
                 policy_w    : float = 1.0,
                 value_w     : float = 1.0,
                 capture_w   : float = 0.3,
                 next_state_w: float = 0.2):
        super().__init__()
        self.w = dict(
            policy=policy_w, value=value_w,
            capture=capture_w, next_state=next_state_w,
        )

    def forward(self,
                preds  : dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor],
                ) -> tuple[torch.Tensor, dict[str, float]]:

        # policy: KL divergence
        policy_loss = F.kl_div(
            preds["policy"],
            targets["policy"].clamp(min=1e-8),
            reduction="batchmean",
        )

        # value: MSE
        value_loss = F.mse_loss(
            preds["value"].squeeze(-1),
            targets["value"],
        )

        # capture: binary cross entropy
        capture_loss = F.binary_cross_entropy_with_logits(
            preds["capture"].squeeze(-1),
            targets["capture"],
        )

        # next state: MSE on board reconstruction
        next_loss = F.mse_loss(
            preds["next_state"],
            targets["next_state"],
        )

        total = (
            self.w["policy"]     * policy_loss  +
            self.w["value"]      * value_loss   +
            self.w["capture"]    * capture_loss +
            self.w["next_state"] * next_loss
        )

        components = {
            "policy"    : policy_loss.item(),
            "value"     : value_loss.item(),
            "capture"   : capture_loss.item(),
            "next_state": next_loss.item(),
            "total"     : total.item(),
        }

        return total, components


if __name__ == "__main__":
    cfg     = GameConfig()
    model   = GameNet(cfg)
    loss_fn = GameLoss()

    print(f"GameNet with {model.count_parameters():,} parameters")
    print()

    B = 4
    board   = torch.randint(0, 10, (B, 12))
    turn    = torch.randint(0, 2,  (B,))
    context = torch.rand(B, 3)

    preds = model(board, turn, context)
    print("Output shapes:")
    for k, v in preds.items():
        print(f"  {k:12s}: {tuple(v.shape)}")

    targets = {
        "policy"    : F.softmax(torch.randn(B, 6), dim=-1),
        "value"     : torch.rand(B) * 2 - 1,
        "capture"   : torch.randint(0, 2, (B,)).float(),
        "next_state": torch.randint(0, 10, (B, 12)).float() / 48.0,
    }

    total, components = loss_fn(preds, targets)
    print("\nLoss breakdown:")
    for k, v in components.items():
        print(f"  {k:12s}: {v:.4f}")

