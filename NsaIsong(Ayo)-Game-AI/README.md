# Nsa Isong (Ayo): Reinforcement Learning via Circular Swin Transformers

This repository implements a high-performance Deep Reinforcement Learning (DRL) pipeline for **Nsa Isong**, a traditional West African strategy game. The project centers on **Andifiok**, an autonomous agent developed using Proximal Policy Optimization (PPO) and Population-Based Training (PBT).

## Project Overview

Nsa Isong presents a unique challenge for standard neural architectures. The board's circular topology and the "sowing" mechanic break the spatial assumptions of traditional Convolutional Neural Networks (CNNs). **Andifiok** overcomes this through a **Circular Swin Transformer** that natively understands the wrapping nature of the game state.

## The Agent: Andifiok

**Andifiok** (meaning "The one who knows" in Ibibio) is a zero-search agent. Unlike traditional engines that rely on computationally expensive look-ahead trees (Minimax/MCTS), Andifiok utilizes a single-pass policy network. Strategic depth is trained into the weights of the Transformer, allowing for master-level decision-making with sub-millisecond latency.

## Technical Stack

1. **Performance Core (C++):** A high-speed engine used for both expert-level Supervised Learning (SL) data generation and Phase 5 RL training.
2. **Neural Architecture:** An 842k parameter Transformer that natively encodes circular board states.
3. **Training (PBT):** An approach that asynchronously optimizes a population of 4 agents across dual NVIDIA T4 GPUs.
4. **Deployment:** Model size is reduced from 9MB (training) to 3.5MB (ONNX) for sub-10ms browser-based inference via ONNX Runtime Web.

## Repository Structure

- `nsa_isong_engine.cpp`: C++ game logic and state transition engine.
- `klent_rl.py`: PBT and PPO orchestration for population evolution.
- `export_onnx.py`: Model distillation and optimization script.
- `index.html`: Client-side web interface for the Andifiok agent.

---

**Andifiok** represents a fusion of traditional Akwa Ibom culture and modern deep learning research.
