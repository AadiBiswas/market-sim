from bisect import insort
from collections import deque

from market_sim.exchange.order import Order, OrderType, Side


class OrderBook:
    """Resting limit orders only. No matching happens here.

    add() refuses anything that would lock or cross the book, so the resting
    book is never crossed. Whatever does matching (Day 3) should execute the
    marketable part first and only rest the leftover.
    """

    def __init__(self) -> None:
        self._levels: dict[Side, dict[int, deque[Order]]] = {Side.BUY: {}, Side.SELL: {}}
        # Both kept ascending, so best bid is the last element and best ask the first.
        # TODO: insort/pop is O(levels). Fine for a reference book; revisit if
        # profiling ever says otherwise.
        self._prices: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: dict[int, Order] = {}

    @property
    def best_bid(self) -> int | None:
        prices = self._prices[Side.BUY]
        return prices[-1] if prices else None

    @property
    def best_ask(self) -> int | None:
        prices = self._prices[Side.SELL]
        return prices[0] if prices else None

    @property
    def spread(self) -> int | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    def add(self, order: Order) -> None:
        if order.order_type is not OrderType.LIMIT:
            raise ValueError(f"only limit orders can rest, got {order.order_type.name}")
        if not order.is_active or order.remaining <= 0:
            raise ValueError(f"order {order.order_id} isn't live ({order.status.name}, remaining={order.remaining})")
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order_id {order.order_id}")

        # Equal prices count as crossing too: a locked book means a trade should
        # have happened.
        if order.side is Side.BUY:
            ask = self.best_ask
            if ask is not None and order.price >= ask:
                raise ValueError(f"bid {order.price} would cross best ask {ask}")
        else:
            bid = self.best_bid
            if bid is not None and order.price <= bid:
                raise ValueError(f"ask {order.price} would cross best bid {bid}")

        levels = self._levels[order.side]
        level = levels.get(order.price)
        if level is None:
            level = levels[order.price] = deque()
            insort(self._prices[order.side], order.price)
        # FIFO is arrival order at the book, not order.timestamp. The simulator
        # is responsible for delivering orders in time order.
        level.append(order)
        self._orders[order.order_id] = order

    def get_order(self, order_id: int) -> Order | None:
        return self._orders.get(order_id)

    def orders_at_price(self, side: Side, price: int) -> tuple[Order, ...]:
        return tuple(self._levels[side].get(price, ()))

    def quantity_at_price(self, side: Side, price: int) -> int:
        return sum(o.remaining for o in self._levels[side].get(price, ()))
