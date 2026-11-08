from typing import Literal

import numpy as np

from market_sim.exchange import OrderType, Side
from market_sim.simulation import MarketObservation, OrderRequest


class NoiseTrader:
    """Uninformed flow. No view on value, no objective, no memory.

    passive: rests a limit order near its own side's best price (or
    `reference_price` when that side is empty), never touching the far side.
    aggressive: sends market orders and takes whatever's there.

    `reference_price` is a fixed model parameter, not the fundamental. This
    agent never sees the fundamental.
    """

    def __init__(
        self,
        agent_id: int,
        seed: int,
        mode: Literal["passive", "aggressive"],
        reference_price: int,
        act_prob: float = 1.0,
        buy_prob: float = 0.5,
        qty_range: tuple[int, int] = (1, 10),
        max_offset: int = 3,
    ) -> None:
        if mode not in ("passive", "aggressive"):
            raise ValueError(f"mode must be 'passive' or 'aggressive', got {mode!r}")
        if not (0 <= act_prob <= 1 and 0 <= buy_prob <= 1):
            raise ValueError("probabilities must be in [0, 1]")
        if not 1 <= qty_range[0] <= qty_range[1]:
            raise ValueError(f"bad qty_range {qty_range}")
        if max_offset < 0 or reference_price < 1:
            raise ValueError("max_offset must be >= 0 and reference_price >= 1")
        self.agent_id = agent_id
        self.mode = mode
        self.reference_price = reference_price
        self.act_prob = act_prob
        self.buy_prob = buy_prob
        self.qty_range = qty_range
        self.max_offset = max_offset
        self._rng = np.random.default_rng(seed)

    def on_wake(self, obs: MarketObservation) -> OrderRequest | None:
        # Same four draws every wake, used or not. The RNG stream then depends
        # only on how often we woke, not on what the book looked like, which
        # keeps seeds comparable across runs with different market activity.
        act = self._rng.random() < self.act_prob
        side = Side.BUY if self._rng.random() < self.buy_prob else Side.SELL
        # numpy ints aren't ints as far as Order validation is concerned.
        qty = int(self._rng.integers(self.qty_range[0], self.qty_range[1] + 1))
        offset = int(self._rng.integers(-self.max_offset, self.max_offset + 1))

        if not act:
            return None
        if self.mode == "aggressive":
            return OrderRequest(side, OrderType.MARKET, qty)

        if side is Side.BUY:
            anchor = obs.best_bid if obs.best_bid is not None else self.reference_price
            price = anchor + offset
            if obs.best_ask is not None:
                price = min(price, obs.best_ask - 1)
        else:
            anchor = obs.best_ask if obs.best_ask is not None else self.reference_price
            price = anchor + offset
            if obs.best_bid is not None:
                price = max(price, obs.best_bid + 1)

        if price < 1:
            # Only happens with an ask at 1 tick. Nowhere passive to go.
            return None
        return OrderRequest(side, OrderType.LIMIT, qty, price)
