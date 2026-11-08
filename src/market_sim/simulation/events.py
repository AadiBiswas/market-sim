"""Scheduled inputs and the records the simulator writes about them.

Inputs are specs, not Orders: the simulator builds the Order when the event
fires, so nothing can poke at it between scheduling and arrival, and its
timestamp can only come from the clock.

Records are frozen snapshots. The live Order keeps changing after it's logged
(fills, cancels), and a log that held the object would rewrite its own past.
"""

from dataclasses import dataclass
from typing import Protocol

from market_sim.exchange import OrderType, Side, Trade


@dataclass(frozen=True, slots=True)
class SubmitOrder:
    order_id: int
    agent_id: int
    side: Side
    order_type: OrderType
    qty: int
    price: int | None = None


@dataclass(frozen=True, slots=True)
class CancelOrder:
    order_id: int


@dataclass(frozen=True, slots=True)
class UpdateFundamental:
    pass


@dataclass(frozen=True, slots=True)
class WakeAgent:
    agent_id: int


Event = SubmitOrder | CancelOrder | UpdateFundamental | WakeAgent


# What agents see and what they can ask for. Agents get plain values only:
# no book, no exchange, no simulator, no live Orders, and nothing about the
# fundamental. Anything an agent is allowed to know has to be put here on purpose.

@dataclass(frozen=True, slots=True)
class MarketObservation:
    timestamp: int
    best_bid: int | None
    best_ask: int | None
    bid_qty: int
    ask_qty: int


@dataclass(frozen=True, slots=True)
class OrderRequest:
    side: Side
    order_type: OrderType
    qty: int
    price: int | None = None


class Agent(Protocol):
    agent_id: int

    def on_wake(self, obs: MarketObservation) -> OrderRequest | None: ...


@dataclass(frozen=True, slots=True)
class Submitted:
    timestamp: int
    order_id: int
    agent_id: int
    side: Side
    order_type: OrderType
    qty: int
    price: int | None


@dataclass(frozen=True, slots=True)
class Cancelled:
    """Order left the book unexecuted: an explicit cancel, or a market order's
    leftover once liquidity ran out (same timestamp as its Submitted)."""
    timestamp: int
    order_id: int
    remaining: int


@dataclass(frozen=True, slots=True)
class CancelRejected:
    """Cancel for an order that wasn't live -- already filled, already gone, or never existed."""
    timestamp: int
    order_id: int


Activity = Submitted | Trade | Cancelled | CancelRejected
