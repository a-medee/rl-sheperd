# ShepherdRL – Multi-Level Robotic Shepherding with Reinforcement Learning

This project implements a **custom reinforcement learning environment** for robotic shepherding, inspired by biologically grounded flocking dynamics (Strombom model).  
The goal is to coordinate one or more **differential-drive robotic shepherds** to guide a group of sheep-like agents toward designated goal regions.

The environment is implemented in **Gym style**, includes **real-time visualization using Pygame**, and supports **rule-based and RL-based agents** across **four increasingly complex levels**.

---

## Project Overview

### Core Features
- Custom Gym-style environment
- Differential-drive shepherd robots (left/right wheels)
- Sheep with shepherd-avoidance behavior
- Pygame visualization
- Rule-based heuristic agent
- Reinforcement Learning (PPO / Stable-Baselines3)
- Multi-agent cooperative and competitive settings

---

## Environment Levels

### Level 1 – Basic Shepherding
- 1 shepherd
- 2–10 sheep
- Random goal (circular area)
- No obstacles
- Deterministic dynamics

### Level 2 – Obstacle Avoidance
- Same as Level 1
- One random square obstacle
- Sheep and shepherd cannot start inside obstacle

### Level 3 – Cooperative Multi-Shepherd
- 2 shepherds
- Shared goal
- Shared reward = number of sheep in goal
- Turn-based action execution

### Level 4 – Competitive Multi-Shepherd
- 2 shepherds
- Separate goal regions (random corners)
- Competitive reward: reward = sheep_in_own_goal − sheep_in_opponent_goal
- Turn-based multi-agent interaction

---

## Sheep Behavior Model

- If a shepherd is **close**, the sheep moves **directly away** from the shepherd (goal ignored)
- Otherwise:
- Levels 1–3: random movement
- Level 4: moves toward nearest goal
- Once a sheep enters a goal area, it **cannot exit**

---

### Rule-Based Shepherd
- Finds **furthest sheep from goal**
- Computes a **driving point behind the sheep** along the goal→sheep line
- Moves toward that point to push sheep toward the goal

Used for:
- Levels 1–2
- Optional warm-start for RL training

---

## Installation: Create a Python environment (recommended)

```bash
conda create -n shepherding python=3.12
conda activate shepherding
pip install -r requirements.txt
```

## Project Structure

```bash
shepherd_rl/
│
├─ envs/
│  └─ shepherd_env.py        # Gym environment
│
├─ agents/
│  ├─ rule_based_agent.py   # Heuristic agent
│  └─ rl_agent.py           # RL utilities
│
├─ main.py                  # Run simulation (rule-based or RL)
├─ train.py                 # Train RL agents (Levels 3 & 4)
├─ models/                  # Saved RL models (.zip)
└─ README.md
```