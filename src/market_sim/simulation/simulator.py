import heapq

from market_sim.exchange import Exchange, Order, OrderStatus, OrderType, Side, Trade
from market_sim.simulation.events import (
    Activity, Agent, CancelOrder, Cancelled, CancelRejected, Event, MarketObservation, SubmitOrder,
    Submitted, UpdateFundamental, WakeAgent,
)
from market_sim.simulation.fundamental import FundamentalProcess, FundamentalSnapshot


class Simulator:
    """Discrete-event loop around an Exchange.

    Time is an integer nanosecond counter that only moves when an event fires.
    It has nothing to do with the wall clock, so a run is the same run no
    matter how fast the machine is.
    """

    def __init__(self, fundamental: FundamentalProcess | None = None) -> None:
        if fundamental is not None and fundamental.timestamp != 0:
            raise ValueError(f"fundamental process must start at t=0, it's at t={fundamental.timestamp}")
        self.exchange = Exchange()
        self.now = 0
        self._queue: list[tuple[int, int, Event]] = []
        # Tie-breaker for equal timestamps: first scheduled, first run. Also
        # means heapq never falls through to comparing event payloads.
        self._seq = 0
        self._log: list[Activity] = []
        self._fundamental = fundamental
        self._fundamental_history: list[FundamentalSnapshot] = []
        if fundamental is not None:
            self._fundamental_history.append(FundamentalSnapshot(fundamental.timestamp, fundamental.value))
        self._agents: dict[int, Agent] = {}
        # Agent orders get IDs from this counter, skipping anything already
        # scheduled by hand. A manual ID scheduled *after* the counter handed it
        # out isn't caught here -- the exchange rejects the duplicate on arrival.
        self._next_order_id = 1
        self._scheduled_order_ids: set[int] = set()

    @property
    def log(self) -> tuple[Activity, ...]:
        return tuple(self._log)

    @property
    def trades(self) -> list[Trade]:
        return [r for r in self._log if isinstance(r, Trade)]

    @property
    def fundamental_history(self) -> tuple[FundamentalSnapshot, ...]:
        # Research-only ground truth, deliberately not in `log`. The log is what
        # agents will eventually observe, and nobody gets the true value for free.
        return tuple(self._fundamental_history)

    def register_agent(self, agent: Agent) -> None:
        if agent.agent_id in self._agents:
            raise ValueError(f"agent_id {agent.agent_id} already registered")
        self._agents[agent.agent_id] = agent

    def schedule(self, timestamp: int, event: Event) -> None:
        if isinstance(event, UpdateFundamental) and self._fundamental is None:
            raise ValueError("can't schedule UpdateFundamental on a simulator with no fundamental process")
        if isinstance(event, WakeAgent) and event.agent_id not in self._agents:
            raise ValueError(f"can't wake unregistered agent {event.agent_id}")
        if isinstance(event, SubmitOrder):
            self._scheduled_order_ids.add(event.order_id)
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
            case UpdateFundamental():
                # Only explicit updates move the process, never order flow. Otherwise
                # one extra order would consume a draw and change the whole path, and
                # two strategies could never be compared on the same fundamentals.
                value = self._fundamental.advance_to(self.now)
                # A repeat at the same instant draws nothing, so there's nothing new to record.
                if self.now > self._fundamental_history[-1].timestamp:
                    self._fundamental_history.append(FundamentalSnapshot(self.now, value))
            case WakeAgent():
                agent = self._agents[event.agent_id]
                req = agent.on_wake(self._observe())
                if req is not None:
                    # Zero latency, but still through the queue: the order lands
                    # at `now` behind anything already waiting at this instant,
                    # and the agent never touches the exchange directly.
                    self.schedule(self.now, SubmitOrder(
                        self._allocate_order_id(), agent.agent_id, req.side, req.order_type, req.qty, req.price,
                    ))
            case _:
                raise TypeError(f"unknown event {event!r}")

    def _observe(self) -> MarketObservation:
        book = self.exchange.book
        bid, ask = book.best_bid, book.best_ask
        return MarketObservation(
            timestamp=self.now,
            best_bid=bid,
            best_ask=ask,
            bid_qty=0 if bid is None else book.quantity_at_price(Side.BUY, bid),
            ask_qty=0 if ask is None else book.quantity_at_price(Side.SELL, ask),
        )

    def _allocate_order_id(self) -> int:
        while self._next_order_id in self._scheduled_order_ids:
            self._next_order_id += 1
        oid = self._next_order_id
        self._next_order_id += 1
        return oid
