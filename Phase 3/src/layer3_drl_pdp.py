"""
Layer 3: Deep Reinforcement Learning Trust Agent (Policy Decision Point - PDP)
Implements DQN-based adaptive Zero-Trust mitigation over state S_t,
action space {Allow, Re-auth, Throttle, Quarantine}, parameterized reward function,
experience replay, target network, and comparison with Static Threshold Policy (Ablation d).
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import DRL_CONFIG
except (ImportError, ValueError):
    from config import DRL_CONFIG

class QNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=4, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return self.fc(state)

class DRLTrustAgent:
    def __init__(self, state_dim=5, action_dim=4, lr=0.001, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = DRL_CONFIG["epsilon_start"]
        self.epsilon_min = DRL_CONFIG["epsilon_min"]
        self.epsilon_decay = DRL_CONFIG["epsilon_decay"]
        
        self.q_net = QNetwork(state_dim, action_dim, DRL_CONFIG["hidden_dim"])
        self.target_net = QNetwork(state_dim, action_dim, DRL_CONFIG["hidden_dim"])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=DRL_CONFIG["memory_capacity"])
        self.step_count = 0
        
        # Action cost mapping L_t (allow is free, quarantine is high cost)
        self.action_costs = {0: 0.0, 1: 0.25, 2: 0.50, 3: 1.00}

    def select_action(self, state, evaluate=False):
        """Epsilon-greedy action selection."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(s_tensor)
            return int(torch.argmax(q_values).item())

    def compute_reward(self, state, action, true_label):
        """
        R_t = lambda_1 * D_t - lambda_2 * FPR_t - lambda_3 * L_t + lambda_4 * DeltaThreat_t
        """
        anomaly_prob = state[0]
        is_attack = (true_label == 1)
        action_cost = self.action_costs[action]
        
        # Detection Utility D_t
        if is_attack:
            if action in [2, 3]: # Throttle or Quarantine correctly applied to attack
                d_t = 1.0
                fpr_t = 0.0
                threat_delta = 1.0
            elif action == 1: # Re-auth on attack
                d_t = 0.4
                fpr_t = 0.0
                threat_delta = 0.4
            else: # Allowed attack (breach!)
                d_t = -1.5
                fpr_t = 0.0
                threat_delta = -1.0
        else: # Benign traffic
            threat_delta = 0.0
            if action == 0: # Correctly allowed benign flow
                d_t = 1.0
                fpr_t = 0.0
            elif action == 1: # Minor disruption
                d_t = 0.0
                fpr_t = 0.5
            else: # False positive (throttled/quarantined benign user)
                d_t = -0.5
                fpr_t = 2.0
                
        r = (DRL_CONFIG["lambda_detection"] * d_t - 
             DRL_CONFIG["lambda_fpr"] * fpr_t - 
             DRL_CONFIG["lambda_latency"] * action_cost + 
             DRL_CONFIG["lambda_threat_delta"] * threat_delta)
        return float(r)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        s_t = torch.FloatTensor(np.array(states))
        a_t = torch.LongTensor(actions).unsqueeze(1)
        r_t = torch.FloatTensor(rewards).unsqueeze(1)
        s_next = torch.FloatTensor(np.array(next_states))
        d_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # Current Q
        current_q = self.q_net(s_t).gather(1, a_t)
        
        # Target Q
        with torch.no_grad():
            max_next_q = self.target_net(s_next).max(1)[0].unsqueeze(1)
            target_q = r_t + (1.0 - d_t) * self.gamma * max_next_q
            
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % DRL_CONFIG["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(loss.item())

class StaticThresholdPolicy:
    """
    Ablation Baseline (d): Static threshold rule-based zero-trust policy.
    Replaces the dynamic DRL agent with static deterministic thresholding.
    """
    def __init__(self, allow_th=0.35, reauth_th=0.65, throttle_th=0.85):
        self.allow_th = allow_th
        self.reauth_th = reauth_th
        self.throttle_th = throttle_th

    def select_action(self, state, evaluate=True):
        anomaly_prob = state[0]
        if anomaly_prob < self.allow_th:
            return 0  # Allow
        elif anomaly_prob < self.reauth_th:
            return 1  # Re-auth
        elif anomaly_prob < self.throttle_th:
            return 2  # Throttle
        else:
            return 3  # Quarantine
