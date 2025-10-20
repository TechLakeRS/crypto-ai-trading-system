"""
Simple signal types for Model 2 and Model 3 engines
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class SignalType(Enum):
    """Signal types for trading"""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


@dataclass
class Signal:
    """
    Simple signal class for Model 2 (DP) and Model 3 (Hybrid) engines
    """
    symbol: str
    signal_type: SignalType
    confidence: float
    strength: float
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    risk_reward_ratio: float
    position_size_pct: float
    primary_reason: str
    supporting_factors: List[str]
    risk_factors: List[str]
    timestamp: datetime
    priority: int
    sources: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
