# hope/hope_protocol.py

from dataclasses import dataclass
import random
from typing import Optional


@dataclass
class HopeConfig:
    """
    Configuration for the Hope Protocol.

    p_hope_base: base probability of reward amplification
    p_hope_boost: boosted probability after stagnation
    stagnation_threshold: steps without positive reward before boost
    min_multiplier: lower bound for reward amplification
    max_multiplier: upper bound for reward amplification
    mode:
        - "positive_only"
        - "negative_only"
        - "symmetric"
    """
    p_hope_base: float = 0.01
    p_hope_boost: float = 0.1
    stagnation_threshold: int = 500

    min_multiplier: float = 2.0
    max_multiplier: float = 3.0
    mode: str = "positive_only"


class HopeProtocol:
    def __init__(self, config: HopeConfig):
        self.config = config
        self.steps_since_positive = 0

    def reset(self):
        """Reset internal counters (call on env reset if desired)."""
        self.steps_since_positive = 0

    def transform(
        self,
        reward: float,
        *,
        step: Optional[int] = None,
    ) -> float:
        # Track stagnation
        if reward > 0:
            self.steps_since_positive = 0
        else:
            self.steps_since_positive += 1

        # Select probability based on stagnation
        if self.steps_since_positive >= self.config.stagnation_threshold:
            p_hope = self.config.p_hope_boost
        else:
            p_hope = self.config.p_hope_base

        # Decide whether hope fires
        if random.random() > p_hope:
            return reward

        # Mode gating
        if self.config.mode == "positive_only" and reward <= 0:
            return reward
        if self.config.mode == "negative_only" and reward >= 0:
            return reward

        multiplier = random.uniform(
            self.config.min_multiplier,
            self.config.max_multiplier
        )

        return reward * multiplier
