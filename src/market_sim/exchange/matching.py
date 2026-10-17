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
        # Every ID ever accepted, live or not. The book only knows live orders,
        # and a recycled ID would make trade/fill history ambiguous later.
        self._seen_order_ids: set[int] = set()

    def submit(self, order: Order) -> list[Trade]:
        """Match `order` against the opposite side and return trades in execution order.

        Limit leftovers rest in the book. Market leftovers are cancelled.
        """
        # Validate everything up front. Finding a problem halfway through a
        # sweep would leave fills applied with nowhere for the order to go.
        # Rejected orders don't consume their ID; accepted ones always do.
        if not order.is_active:
            raise ValueError(f"order {order.order_id} isn't live ({order.status.name})")
        if order.order_id in self._seen_order_ids:
            raise ValueError(f"order_id {order.order_id} already used")
        self._seen_order_ids.add(order.order_id)

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
            if order.order_type is OrderType.MARKET:
                # Book ran dry. Cancel rather than zero it out, so the unfilled
                # size stays visible to anything measuring fill rates.
                order.cancel()
            else:
                # Loop only stops early once the order no longer crosses, so this
                # can't trip add()'s crossing check. Priority starts now.
                self.book.add(order)
        return trades

    def cancel(self, order_id: int) -> Order | None:
        """Pull a resting order. Returns it, or None if nothing live has that ID.

        Not an error: a strategy cancelling a quote that filled a moment ago is
        perfectly normal.
        """
        order = self.book.get_order(order_id)
        if order is None:
            return None
        self.book._remove(order)
        order.cancel()
        return order

    @staticmethod
    def _crosses(order: Order, contra_price: int) -> bool:
        if order.order_type is OrderType.MARKET:
            return True
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
