# utils/rl_agent.py (fixed)
import os
import json
import random
import math
import numpy as np
from typing import List, Tuple

class SimpleRLAgent:
    """
    Q-table style lightweight agent that indexes states by:
      (sim_bucket, sentiment, degree_match_bool, skill_bucket)
    """

    def __init__(self, actions: List[str], learning_rate=0.1, discount=0.9, epsilon=0.15):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = {}
        self.rewards = []

    # ----------------------
    # Helpers
    # ----------------------
    def _is_missing_numeric(self, x) -> bool:
        """Return True for None or NaN-like numeric values."""
        if x is None:
            return True
        try:
            # works for Python float, numpy.float64, etc.
            return math.isnan(float(x))
        except Exception:
            return False

    # ----------------------
    # State discretizers
    # ----------------------
    def _sim_bucket(self, sim: float) -> str:
        # buckets: low (<0.2), med (0.2-0.6), high (>0.6)
        if self._is_missing_numeric(sim):
            return "sim_unknown"
        sim = float(sim)
        if sim >= 0.6:
            return "high"
        if sim >= 0.2:
            return "medium"
        return "low"

    def _skill_bucket(self, skill_pct: float) -> str:
        # buckets: low (<0.3), mid (0.3-0.6), high (>0.6)
        if self._is_missing_numeric(skill_pct):
            return "skill_unknown"
        skill_pct = float(skill_pct)
        if skill_pct >= 0.6:
            return "high"
        if skill_pct >= 0.3:
            return "mid"
        return "low"

    def _normalize_sentiment_label(self, sentiment) -> str:
        """
        Ensure sentiment is a lowercase string label.
        If sentiment is numeric, map it to a label.
        If missing/NaN, return 'neutral'.
        """
        if sentiment is None:
            return "neutral"

        # If it's a numeric score, map to label thresholds (adjust as needed)
        try:
            s_val = float(sentiment)
            # Example mapping: >0.2 positive, < -0.2 negative, else neutral
            if math.isnan(s_val):
                return "neutral"
            if s_val > 0.2:
                return "positive"
            if s_val < -0.2:
                return "negative"
            return "neutral"
        except Exception:
            # Not numeric; coerce to string and lower
            try:
                return str(sentiment).strip().lower()
            except Exception:
                return "neutral"

    def _state_key(self, sim: float, sentiment, degree_match: bool, skill_pct: float) -> Tuple:
        s = self._sim_bucket(sim)
        sk = self._skill_bucket(skill_pct)
        sent = self._normalize_sentiment_label(sentiment)
        deg = bool(degree_match)
        return (s, sent, deg, sk)

    # ----------------------
    # Q-table helpers
    # ----------------------
    def _ensure_state(self, key: Tuple):
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.actions}

    # ----------------------
    # Policy
    # ----------------------
    def choose_action(self, sim: float, sentiment, degree_match: bool, skill_pct: float) -> str:
        key = self._state_key(sim, sentiment, degree_match, skill_pct)
        self._ensure_state(key)

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        qvals = self.q_table[key]
        best_action = max(qvals, key=qvals.get)
        return best_action

    # ----------------------
    # Q update
    # ----------------------
    def update(self, sim: float, sentiment, degree_match: bool, skill_pct: float, action: str, reward: float):
        key = self._state_key(sim, sentiment, degree_match, skill_pct)
        self._ensure_state(key)

        if action not in self.actions:
            raise ValueError(f"Unknown action: {action}")

        old = self.q_table[key][action]
        new = old + self.lr * (float(reward) - old)
        self.q_table[key][action] = new

        self.rewards.append(float(reward))

    # ----------------------
    # Utilities
    # ----------------------
    def get_q_values(self, sim: float, sentiment, degree_match: bool, skill_pct: float):
        key = self._state_key(sim, sentiment, degree_match, skill_pct)
        return self.q_table.get(key, {a: 0.0 for a in self.actions})

    def print_q_table(self):
        print("=== Q TABLE (state -> {action: value}) ===")
        for k, v in self.q_table.items():
            print(k, "->", v)

    def get_reward_history(self):
        return list(self.rewards)

    def save_q_table(self, path="models/q_table.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {json.dumps(k): v for k, v in self.q_table.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"q_table": serializable, "rewards": self.rewards}, f, indent=2)

    def load_q_table(self, path="models/q_table.json"):
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        serializable = data.get("q_table", {})
        reconstructed = {}
        for kstr, v in serializable.items():
            k = tuple(json.loads(kstr))
            reconstructed[k] = v
        self.q_table = reconstructed
        self.rewards = data.get("rewards", [])
        return True
