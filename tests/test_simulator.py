import pytest

from helpers import check_invariants, snapshot
from market_sim.exchange import OrderStatus, OrderType, Side, Trade
from market_sim.simulation import CancelOrder, Cancelled, CancelRejected, Simulator, SubmitOrder, Submitted


def limit(oid, side, price, qty, agent=None):
    return SubmitOrder(oid, oid if agent is None else agent, side, OrderType.LIMIT, qty, price)


def market(oid, side, qty):
    return SubmitOrder(oid, oid, side, OrderType.MARKET, qty)


def kinds(log):
    """Compact view of the log for asserting on sequence."""
    out = []
    for r in log:
        match r:
            case Submitted():
                out.append((r.timestamp, "submit", r.order_id))
            case Trade():
                out.append((r.timestamp, "trade", r.buy_order_id, r.sell_order_id, r.qty, r.price))
            case Cancelled():
                out.append((r.timestamp, "cancel", r.order_id, r.remaining))
            case CancelRejected():
                out.append((r.timestamp, "cancel_rejected", r.order_id))
    return out


def reconstruct(log):
    """Final (executed, remaining, status) per order from the log alone. No live Orders."""
    state = {}
    for r in log:
        match r:
            case Submitted():
                state[r.order_id] = [0, r.qty, OrderStatus.OPEN]
            case Trade():
                for oid in (r.buy_order_id, r.sell_order_id):
                    s = state[oid]
                    s[0] += r.qty
                    s[1] -= r.qty
                    s[2] = OrderStatus.FILLED if s[1] == 0 else OrderStatus.PARTIALLY_FILLED
            case Cancelled():
                s = state[r.order_id]
                assert s[1] == r.remaining
                s[2] = OrderStatus.CANCELLED
    return {oid: tuple(s) for oid, s in state.items()}


# -- ordering and the clock --

def test_events_run_in_time_order_then_insertion_order():
    sim = Simulator()
    sim.schedule(100, limit(1, Side.BUY, 90, 5))
    sim.schedule(300, limit(2, Side.BUY, 91, 5))
    sim.schedule(100, limit(3, Side.BUY, 92, 5))
    sim.schedule(200, CancelOrder(1))
    sim.run()

    assert kinds(sim.log) == [
        (100, "submit", 1),
        (100, "submit", 3),
        (200, "cancel", 1, 5),
        (300, "submit", 2),
    ]
    assert sim.now == 300


def test_empty_run_is_a_noop():
    sim = Simulator()
    sim.run()
    assert (sim.now, sim.log, sim.trades) == (0, (), [])


def test_cannot_schedule_into_the_past():
    sim = Simulator()
    with pytest.raises(ValueError):
        sim.schedule(-1, CancelOrder(1))
    sim.schedule(500, limit(1, Side.BUY, 100, 5))
    sim.run()
    with pytest.raises(ValueError):
        sim.schedule(499, CancelOrder(1))
    assert sim.log[-1] == Submitted(500, 1, 1, Side.BUY, OrderType.LIMIT, 5, 100)


def test_can_keep_scheduling_after_run():
    sim = Simulator()
    sim.schedule(100, limit(1, Side.SELL, 101, 5))
    sim.run()
    sim.schedule(100, limit(2, Side.BUY, 101, 2))  # same instant is still allowed
    sim.schedule(250, limit(3, Side.BUY, 101, 3))
    sim.run()

    assert [(t.timestamp, t.buy_order_id, t.qty) for t in sim.trades] == [(100, 2, 2), (250, 3, 3)]
    assert sim.now == 250
    assert sim.exchange.book.best_ask is None


def test_same_time_order_decides_who_rests_and_who_aggresses():
    def run(first, second):
        sim = Simulator()
        sim.schedule(100, first)
        sim.schedule(100, second)
        sim.run()
        return sim.trades

    bid, ask = limit(1, Side.BUY, 102, 5), limit(2, Side.SELL, 100, 5)

    # Bid arrives first and rests; the ask hits it at the bid's price.
    (t,) = run(bid, ask)
    assert (t.price, t.aggressor) == (102, Side.SELL)

    (t,) = run(ask, bid)
    assert (t.price, t.aggressor) == (100, Side.BUY)


# -- submissions --

def test_passive_order_rests_with_dispatch_timestamp():
    sim = Simulator()
    sim.schedule(1500, limit(1, Side.BUY, 100, 10))
    sim.run()
    o = sim.exchange.book.get_order(1)
    assert (o.timestamp, o.status, o.remaining) == (1500, OrderStatus.OPEN, 10)
    assert sim.exchange.book.best_bid == 100


def test_crossing_and_market_orders_trade_at_their_arrival_time():
    sim = Simulator()
    sim.schedule(100, limit(1, Side.SELL, 101, 3))
    sim.schedule(100, limit(2, Side.SELL, 102, 10))
    sim.schedule(700, limit(3, Side.BUY, 102, 5))  # sweeps 3 @101, 2 @102
    sim.schedule(900, market(4, Side.BUY, 4))
    sim.run()

    assert [(t.timestamp, t.price, t.qty, t.buy_order_id) for t in sim.trades] == [
        (700, 101, 3, 3), (700, 102, 2, 3), (900, 102, 4, 4),
    ]
    assert sim.exchange.book.quantity_at_price(Side.SELL, 102) == 4
    check_invariants(sim.exchange.book)


def test_market_order_leftover_logged_as_cancel_at_submit_time():
    sim = Simulator()
    sim.schedule(100, limit(1, Side.BUY, 99, 3))
    sim.schedule(200, market(2, Side.SELL, 10))
    sim.run()
    assert kinds(sim.log)[1:] == [
        (200, "submit", 2),
        (200, "trade", 1, 2, 3, 99),
        (200, "cancel", 2, 7),
    ]


# -- cancellations --

def test_cancel_after_partial_fill_keeps_unexecuted_qty():
    sim = Simulator()
    sim.schedule(100, limit(1, Side.BUY, 100, 10))
    sim.schedule(200, market(2, Side.SELL, 4))
    sim.schedule(300, CancelOrder(1))
    sim.run()

    assert sim.log[-1] == Cancelled(300, 1, 6)
    assert sim.exchange.book.best_bid is None


@pytest.mark.parametrize("setup", ["filled", "unknown", "before_arrival", "twice"])
def test_cancel_of_non_live_order_is_harmless(setup):
    sim = Simulator()
    match setup:
        case "filled":
            sim.schedule(100, limit(1, Side.BUY, 100, 5))
            sim.schedule(200, market(2, Side.SELL, 5))
            sim.schedule(300, CancelOrder(1))
        case "unknown":
            sim.schedule(300, CancelOrder(1))
        case "before_arrival":
            sim.schedule(300, CancelOrder(1))
            sim.schedule(400, limit(1, Side.BUY, 100, 5))
        case "twice":
            sim.schedule(100, limit(1, Side.BUY, 100, 5))
            sim.schedule(300, CancelOrder(1))
            sim.schedule(300, CancelOrder(1))
    sim.run()

    assert CancelRejected(300, 1) in sim.log
    assert sum(isinstance(r, Cancelled) for r in sim.log) == (1 if setup == "twice" else 0)
    if setup == "before_arrival":
        # The early cancel didn't poison the ID or the later order.
        o = sim.exchange.book.get_order(1)
        assert (o.status, o.timestamp) == (OrderStatus.OPEN, 400)
    check_invariants(sim.exchange.book)


# -- failures --

def test_invalid_event_stops_the_run_with_context():
    sim = Simulator()
    sim.schedule(100, limit(1, Side.BUY, 100, 5))
    sim.schedule(200, limit(1, Side.SELL, 120, 5))  # duplicate ID
    sim.schedule(300, limit(2, Side.SELL, 120, 5))
    with pytest.raises(ValueError, match="already used") as exc:
        sim.run()

    assert any("t=200" in n for n in exc.value.__notes__)
    assert sim.now == 200
    # The bad submission left no trace; the later event is still queued.
    assert kinds(sim.log) == [(100, "submit", 1)]
    sim.run()
    assert kinds(sim.log)[-1] == (300, "submit", 2)


def test_malformed_spec_fails_at_dispatch():
    sim = Simulator()
    sim.schedule(100, SubmitOrder(1, 1, Side.BUY, OrderType.LIMIT, 0, 100))
    with pytest.raises(ValueError, match="qty must be positive"):
        sim.run()
    assert sim.log == ()


# -- the log --

def test_log_records_are_snapshots():
    sim = Simulator()
    sim.schedule(100, limit(1, Side.BUY, 100, 10))
    sim.run()
    first = sim.log[0]

    sim.schedule(200, market(2, Side.SELL, 10))
    sim.run()
    # The live order is now FILLED; the record of it arriving hasn't moved.
    assert sim.log[0] is first
    assert first == Submitted(100, 1, 1, Side.BUY, OrderType.LIMIT, 10, 100)
    with pytest.raises(AttributeError):
        first.qty = 3  # type: ignore[misc]
    assert isinstance(sim.log, tuple)


def test_brief_sequence_reconstructs_from_log():
    sim = Simulator()
    sim.schedule(100, limit(1, Side.BUY, 100, 10))
    sim.schedule(200, market(2, Side.SELL, 4))
    sim.schedule(300, CancelOrder(1))
    sim.schedule(400, limit(3, Side.BUY, 99, 5))
    sim.run()

    assert kinds(sim.log) == [
        (100, "submit", 1),
        (200, "submit", 2),
        (200, "trade", 1, 2, 4, 100),
        (300, "cancel", 1, 6),
        (400, "submit", 3),
    ]
    assert reconstruct(sim.log) == {
        1: (4, 6, OrderStatus.CANCELLED),
        2: (4, 0, OrderStatus.FILLED),
        3: (0, 5, OrderStatus.OPEN),
    }
    assert snapshot(sim.exchange.book) == {
        Side.BUY: [(99, [(3, 5, OrderStatus.OPEN)])],
        Side.SELL: [],
    }
    assert sim.now == 400


def busy_schedule(sim):
    """A fixed, tie-heavy script that touches every path."""
    events = [
        (100, limit(1, Side.SELL, 103, 5)),
        (100, limit(2, Side.SELL, 101, 4)),
        (100, limit(3, Side.SELL, 101, 6)),
        (100, limit(4, Side.BUY, 98, 7)),
        (150, CancelOrder(9)),
        (200, limit(5, Side.BUY, 102, 8)),
        (200, market(6, Side.SELL, 3)),
        (250, CancelOrder(3)),
        (250, limit(7, Side.SELL, 98, 20)),
        (300, market(8, Side.BUY, 30)),
        (300, limit(9, Side.BUY, 97, 2)),
        (350, CancelOrder(5)),
        (400, limit(10, Side.BUY, 99, 4, agent=1)),
        (400, limit(11, Side.SELL, 99, 1, agent=1)),
    ]
    for t, e in events:
        sim.schedule(t, e)


def test_replay_is_identical():
    a, b = Simulator(), Simulator()
    busy_schedule(a)
    busy_schedule(b)
    a.run()
    b.run()

    # Value equality on frozen records, not identity.
    assert a.log == b.log
    assert len(a.trades) >= 5
    assert snapshot(a.exchange.book) == snapshot(b.exchange.book)
    assert a.now == b.now == 400


def test_busy_log_agrees_with_live_state():
    sim = Simulator()
    busy_schedule(sim)
    sim.run()
    rebuilt = reconstruct(sim.log)
    book = sim.exchange.book
    for oid, (executed, remaining, status) in rebuilt.items():
        live = book.get_order(oid)
        if live is not None:
            assert (live.remaining, live.status) == (remaining, status)
        else:
            assert status in (OrderStatus.FILLED, OrderStatus.CANCELLED)
    # Everything still resting got a Submitted record.
    assert set(book._orders) <= set(rebuilt)
    check_invariants(book)
