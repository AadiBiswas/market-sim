"""Order domain types.

Conventions:
- Prices are integer ticks. Float prices make price-level lookups and equality
  checks a minefield; conversion to currency happens at the edges, not here.
- Timestamps are integer nanoseconds of simulation time, not wall clock.
- Order IDs are assigned by the caller (eventually the exchange), so tests can
  pick IDs explicitly and runs stay deterministic.
"""

from dataclasses import dataclass, field
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(Enum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


# Everything else on an order is fixed at submission. Changing price or size
# means cancel/replace, which loses queue priority -- that's the point.
_MUTABLE_FIELDS = frozenset({"remaining", "status"})


@dataclass(slots=True)
class Order:
    order_id: int
    agent_id: int
    side: Side
    order_type: OrderType
    qty: int
    timestamp: int
    price: int | None = None
    remaining: int = field(init=False)
    status: OrderStatus = field(init=False, default=OrderStatus.OPEN)

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        if self.order_type is OrderType.LIMIT:
            if self.price is None:
                raise ValueError("limit order needs a price")
            if not isinstance(self.price, int) or self.price <= 0:
                raise ValueError(f"limit price must be a positive int (ticks), got {self.price!r}")
        elif self.price is not None:
            # A market order with a price is really a marketable limit order.
            # Make the caller say which one they mean.
            raise ValueError("market order can't have a price")
        self.remaining = self.qty

    def __setattr__(self, name: str, value) -> None:
        # Unset slots aren't attributes yet, so this only blocks reassignment
        # after __init__ has run.
        if name not in _MUTABLE_FIELDS and hasattr(self, name):
            raise AttributeError(f"Order.{name} is immutable")
        object.__setattr__(self, name, value)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    def fill(self, qty: int) -> None:
        if not self.is_active:
            raise ValueError(f"can't fill order {self.order_id} with status {self.status.name}")
        if not 0 < qty <= self.remaining:
            raise ValueError(f"fill qty {qty} out of range for remaining {self.remaining}")
        self.remaining -= qty
        self.status = OrderStatus.FILLED if self.remaining == 0 else OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        # remaining is left alone on purpose: after cancel it means "never
        # executed", and zero is reserved for fully filled.
        if not self.is_active:
            raise ValueError(f"can't cancel order {self.order_id} with status {self.status.name}")
        self.status = OrderStatus.CANCELLED
