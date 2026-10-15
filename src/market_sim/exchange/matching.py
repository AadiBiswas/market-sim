from dataclasses import dataclass

from market_sim.exchange.book import OrderBook
from market_sim.exchange.order import Order, OrderType, Side


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: int
    price: int
    qty: int
    # Time the aggressor arrived; every trade it sweeps shares this stamp.
    timestamp: int
    buy_order_id: int
    sell_order_id: int
    buy_agent_id: int
    sell_agent_id: int
    aggressor: Side


class Exchange:
    def __init__(self) -> None:
        self.book = OrderBook()
        self._next_trade_id = 1

    def submit(self, order: Order) -> list[Trade]:
        """Match `order` against the opposite side, rest any leftover, return trades in execution order."""
        # Validate everything up front. Finding a problem halfway through a
        # sweep would leave fills applied with nowhere for the order to go.
        if order.order_type is not OrderType.LIMIT:
            raise ValueError(f"only limit orders are supported for now, got {order.order_type.name}")
        if not order.is_active:
            raise ValueError(f"order {order.order_id} isn't live ({order.status.name})")
        if self.book.get_order(order.order_id) is not None:
            raise ValueError(f"duplicate order_id {order.order_id}")

        contra = order.side.opposite
        trades: list[Trade] = []

        while order.remaining:
            resting = self.book._front(contra)
            if resting is None or not self._crosses(order, resting.price):
                break

            qty = min(order.remaining, resting.remaining)
            resting.fill(qty)
            order.fill(qty)
            trades.append(self._make_trade(order, resting, qty))

            if resting.remaining == 0:
                self.book._pop_front(contra)
            # A partially filled resting order stays at the front of its queue.

        if order.remaining:
            # Loop only stops early once the order no longer crosses, so this
            # can't trip add()'s crossing check. Priority starts now.
            self.book.add(order)
        return trades

    @staticmethod
    def _crosses(order: Order, contra_price: int) -> bool:
        if order.side is Side.BUY:
            return contra_price <= order.price
        return contra_price >= order.price

    def _make_trade(self, aggressor: Order, resting: Order, qty: int) -> Trade:
        buy, sell = (aggressor, resting) if aggressor.side is Side.BUY else (resting, aggressor)
        trade = Trade(
            trade_id=self._next_trade_id,
            # Resting order sets the price. An aggressive buy @105 into an ask
            # @101 trades at 101; the improvement goes to the aggressor.
            price=resting.price,
            qty=qty,
            timestamp=aggressor.timestamp,
            buy_order_id=buy.order_id,
            sell_order_id=sell.order_id,
            buy_agent_id=buy.agent_id,
            sell_agent_id=sell.agent_id,
            aggressor=aggressor.side,
        )
        self._next_trade_id += 1
        return trade
