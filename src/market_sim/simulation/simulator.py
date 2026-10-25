import heapq

from market_sim.exchange import Exchange, Order, OrderStatus, OrderType, Trade
from market_sim.simulation.events import (
    Activity, CancelOrder, Cancelled, CancelRejected, Event, SubmitOrder, Submitted,
)


class Simulator:
    """Discrete-event loop around an Exchange.

    Time is an integer nanosecond counter that only moves when an event fires.
    It has nothing to do with the wall clock, so a run is the same run no
    matter how fast the machine is.
    """

    def __init__(self) -> None:
        self.exchange = Exchange()
        self.now = 0
        self._queue: list[tuple[int, int, Event]] = []
        # Tie-breaker for equal timestamps: first scheduled, first run. Also
        # means heapq never falls through to comparing event payloads.
        self._seq = 0
        self._log: list[Activity] = []

    @property
    def log(self) -> tuple[Activity, ...]:
        return tuple(self._log)

    @property
    def trades(self) -> list[Trade]:
        return [r for r in self._log if isinstance(r, Trade)]

    def schedule(self, timestamp: int, event: Event) -> None:
        # Scheduling at exactly `now` is fine: it runs after anything already
        # queued for this instant.
        if timestamp < self.now:
            raise ValueError(f"can't schedule at t={timestamp}, clock is already at t={self.now}")
        heapq.heappush(self._queue, (timestamp, self._seq, event))
        self._seq += 1

    def run(self) -> None:
        while self._queue:
            t, _, event = heapq.heappop(self._queue)
            self.now = t
            try:
                self._dispatch(event)
            except ValueError as e:
                # No rollback: earlier events already happened. Just say where it broke.
                e.add_note(f"while dispatching {event} at t={t}")
                raise

    def _dispatch(self, event: Event) -> None:
        match event:
            case SubmitOrder():
                order = Order(
                    order_id=event.order_id, agent_id=event.agent_id, side=event.side,
                    order_type=event.order_type, qty=event.qty, timestamp=self.now, price=event.price,
                )
                trades = self.exchange.submit(order)
                # Logged only after submit returns, so a rejected order leaves no trace.
                self._log.append(Submitted(
                    timestamp=self.now, order_id=order.order_id, agent_id=order.agent_id, side=order.side,
                    order_type=order.order_type, qty=order.qty, price=order.price,
                ))
                self._log.extend(trades)
                if order.order_type is OrderType.MARKET and order.status is OrderStatus.CANCELLED:
                    self._log.append(Cancelled(self.now, order.order_id, order.remaining))
            case CancelOrder():
                order = self.exchange.cancel(event.order_id)
                if order is None:
                    self._log.append(CancelRejected(self.now, event.order_id))
                else:
                    self._log.append(Cancelled(self.now, order.order_id, order.remaining))
            case _:
                raise TypeError(f"unknown event {event!r}")
