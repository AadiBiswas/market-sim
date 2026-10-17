import random

import pytest

from helpers import buy, check_invariants, ids_at, lim, mkt, sell
from market_sim.exchange import Exchange, OrderStatus, Side


def test_aggressive_buy_hits_best_ask_at_resting_price():
    ex = Exchange()
    ex.submit(sell(1, 101, 5))
    trades = ex.submit(buy(2, 105, 5, ts=42))

    assert len(trades) == 1
    t = trades[0]
    # 101, not 105: resting price wins.
    assert (t.price, t.qty, t.timestamp) == (101, 5, 42)
    assert (t.buy_order_id, t.sell_order_id) == (2, 1)
    assert (t.buy_agent_id, t.sell_agent_id) == (2, 1)
    assert t.aggressor is Side.BUY
    assert ex.book.best_ask is None and ex.book.best_bid is None
    check_invariants(ex.book)


def test_aggressive_sell_hits_best_bid_at_resting_price():
    ex = Exchange()
    ex.submit(buy(1, 99, 5))
    ex.submit(buy(2, 98, 5))
    trades = ex.submit(sell(3, 95, 5))

    assert [(t.price, t.qty) for t in trades] == [(99, 5)]
    assert (trades[0].buy_order_id, trades[0].sell_order_id) == (1, 3)
    assert trades[0].aggressor is Side.SELL
    assert ex.book.best_bid == 98
    check_invariants(ex.book)


def test_small_aggressor_partially_fills_resting_which_keeps_its_place():
    ex = Exchange()
    ex.submit(sell(1, 101, 10))
    ex.submit(sell(2, 101, 10))
    incoming = buy(3, 101, 4)
    trades = ex.submit(incoming)

    assert [(t.sell_order_id, t.qty) for t in trades] == [(1, 4)]
    assert incoming.status is OrderStatus.FILLED
    assert ex.book.get_order(3) is None

    resting = ex.book.get_order(1)
    assert (resting.remaining, resting.status) == (6, OrderStatus.PARTIALLY_FILLED)
    assert ids_at(ex, Side.SELL, 101) == [1, 2]
    check_invariants(ex.book)


def test_exact_fill_removes_order_and_level():
    ex = Exchange()
    ex.submit(sell(1, 101, 5))
    ex.submit(sell(2, 103, 5))
    resting = ex.book.get_order(1)
    trades = ex.submit(buy(3, 101, 5))

    assert [(t.price, t.qty) for t in trades] == [(101, 5)]
    assert resting.status is OrderStatus.FILLED
    assert ex.book.get_order(1) is None
    assert ex.book.orders_at_price(Side.SELL, 101) == ()
    assert ex.book.best_ask == 103
    check_invariants(ex.book)


def test_large_aggressor_fills_resting_and_rests_remainder():
    ex = Exchange()
    ex.submit(sell(1, 101, 4))
    incoming = buy(2, 101, 10)
    trades = ex.submit(incoming)

    assert [(t.price, t.qty) for t in trades] == [(101, 4)]
    assert ex.book.get_order(1) is None
    # Same object, same ID, now resting on the bid at its own limit.
    assert ex.book.get_order(2) is incoming
    assert (incoming.remaining, incoming.status) == (6, OrderStatus.PARTIALLY_FILLED)
    assert (ex.book.best_bid, ex.book.best_ask) == (101, None)
    check_invariants(ex.book)


def test_fifo_within_level_ignores_ids_and_timestamps():
    ex = Exchange()
    # A arrives first but has the larger ID and later timestamp.
    ex.submit(sell(50, 101, 3, ts=900))
    ex.submit(sell(7, 101, 3, ts=100))
    trades = ex.submit(buy(99, 101, 4))

    assert [(t.sell_order_id, t.qty) for t in trades] == [(50, 3), (7, 1)]
    assert ids_at(ex, Side.SELL, 101) == [7]
    check_invariants(ex.book)


def test_better_price_beats_older_worse_price():
    ex = Exchange()
    ex.submit(sell(1, 102, 5))
    ex.submit(sell(2, 101, 5))
    trades = ex.submit(buy(3, 102, 5))

    assert [(t.sell_order_id, t.price) for t in trades] == [(2, 101)]
    assert ex.book.best_ask == 102
    check_invariants(ex.book)


def test_walks_levels_and_stops_at_limit():
    ex = Exchange()
    for o in (sell(1, 101, 3), sell(2, 101, 5), sell(3, 102, 4), sell(4, 103, 10)):
        ex.submit(o)
    incoming = buy(5, 102, 10)
    trades = ex.submit(incoming)

    assert [(t.sell_order_id, t.price, t.qty) for t in trades] == [(1, 101, 3), (2, 101, 5), (3, 102, 2)]
    assert [t.trade_id for t in trades] == [1, 2, 3]
    assert incoming.status is OrderStatus.FILLED
    assert ex.book.best_ask == 102
    assert ex.book.quantity_at_price(Side.SELL, 102) == 2
    assert ex.book.quantity_at_price(Side.SELL, 103) == 10
    check_invariants(ex.book)


def test_sweep_then_remainder_rests_at_its_limit():
    ex = Exchange()
    ex.submit(sell(1, 101, 2))
    ex.submit(sell(2, 102, 4))
    ex.submit(sell(3, 104, 5))
    ex.submit(buy(4, 99, 1))
    incoming = buy(5, 102, 10)
    trades = ex.submit(incoming)

    assert [(t.price, t.qty) for t in trades] == [(101, 2), (102, 4)]
    assert (incoming.remaining, incoming.status) == (4, OrderStatus.PARTIALLY_FILLED)
    assert (ex.book.best_bid, ex.book.best_ask, ex.book.spread) == (102, 104, 2)
    assert ids_at(ex, Side.BUY, 102) == [5]
    check_invariants(ex.book)


def test_remainder_priority_starts_when_it_rests():
    # A crossing order's remainder always opens a fresh level (anything already
    # resting at that price would have crossed the asks it just swept), so the
    # thing to check is that later arrivals queue behind it.
    ex = Exchange()
    ex.submit(sell(1, 101, 2))
    ex.submit(buy(2, 101, 6))
    ex.submit(buy(3, 101, 1))
    ex.submit(buy(4, 101, 1))
    assert ids_at(ex, Side.BUY, 101) == [2, 3, 4]

    trades = ex.submit(sell(5, 101, 5))
    assert [(t.buy_order_id, t.qty) for t in trades] == [(2, 4), (3, 1)]
    assert ids_at(ex, Side.BUY, 101) == [4]
    check_invariants(ex.book)


def test_passive_order_rests_without_trading():
    ex = Exchange()
    ex.submit(sell(1, 105, 5))
    incoming = buy(2, 103, 10)
    trades = ex.submit(incoming)

    assert trades == []
    assert incoming.status is OrderStatus.OPEN
    assert (ex.book.best_bid, ex.book.best_ask) == (103, 105)
    check_invariants(ex.book)


def test_quantity_conserved_across_sweep():
    ex = Exchange()
    asks = [sell(1, 100, 7), sell(2, 100, 3), sell(3, 101, 6), sell(4, 102, 9), sell(5, 104, 4)]
    for o in asks:
        ex.submit(o)
    before = sum(ex.book.quantity_at_price(Side.SELL, p) for p in (100, 101, 102, 104))

    incoming = buy(6, 102, 30)
    trades = ex.submit(incoming)
    traded = sum(t.qty for t in trades)
    after = sum(ex.book.quantity_at_price(Side.SELL, p) for p in (100, 101, 102, 104))

    assert traded == before - after == incoming.qty - incoming.remaining == 25
    assert sum(o.qty - o.remaining for o in asks) == traded
    check_invariants(ex.book)


def test_trade_ids_keep_counting_across_submissions():
    ex = Exchange()
    ex.submit(sell(1, 100, 1))
    ex.submit(sell(2, 100, 1))
    t1 = ex.submit(buy(3, 100, 1))
    t2 = ex.submit(buy(4, 100, 1))
    assert [t.trade_id for t in t1 + t2] == [1, 2]


def test_invalid_submissions_rejected_before_any_fill():
    ex = Exchange()
    ex.submit(sell(1, 101, 5))
    ex.submit(buy(2, 99, 5))

    dup = buy(2, 101, 5)  # same ID as the resting bid, and marketable
    filled = buy(3, 101, 1)
    filled.fill(1)
    cancelled = buy(4, 101, 1)
    cancelled.status = OrderStatus.CANCELLED

    for o in (dup, filled, cancelled):
        with pytest.raises(ValueError):
            ex.submit(o)

    resting = ex.book.get_order(1)
    assert (resting.remaining, resting.status) == (5, OrderStatus.OPEN)
    check_invariants(ex.book)


@pytest.mark.parametrize("seed", range(20))
def test_random_flow_keeps_invariants(seed):
    rng = random.Random(seed)
    ex = Exchange()
    orders, traded = [], 0
    for oid in range(400):
        roll = rng.random()
        if roll < 0.15 and orders:
            ex.cancel(rng.choice(orders).order_id)  # often dead already, which is fine
            check_invariants(ex.book)
            continue
        side, qty = rng.choice(list(Side)), rng.randint(1, 20)
        if roll < 0.25:
            o = mkt(oid, side, qty, ts=oid)
        else:
            o = lim(oid, side, rng.randint(95, 105), qty, ts=oid, agent=rng.randrange(5))
        orders.append(o)
        trades = ex.submit(o)
        assert sum(t.qty for t in trades) == o.qty - o.remaining
        if o.price is not None:
            for t in trades:
                assert t.price <= o.price if o.side is Side.BUY else t.price >= o.price
        if o.price is None:
            assert ex.book.get_order(oid) is None
            assert o.status in (OrderStatus.FILLED, OrderStatus.CANCELLED)
        traded += sum(t.qty for t in trades)
        check_invariants(ex.book)

    # Every traded share left exactly one buy and one sell.
    filled = {side: sum(o.qty - o.remaining for o in orders if o.side is side) for side in Side}
    assert filled[Side.BUY] == filled[Side.SELL] == traded


@pytest.mark.parametrize("lifecycle", ["resting", "filled", "cancelled", "market_filled", "market_cancelled"])
def test_order_id_never_reusable(lifecycle):
    ex = Exchange()
    match lifecycle:
        case "resting":
            ex.submit(buy(1, 99, 5))
        case "filled":
            ex.submit(sell(9, 101, 5))
            ex.submit(buy(1, 101, 5))
        case "cancelled":
            ex.submit(buy(1, 99, 5))
            ex.cancel(1)
        case "market_filled":
            ex.submit(sell(9, 101, 5))
            ex.submit(mkt(1, Side.BUY, 5))
        case "market_cancelled":
            ex.submit(mkt(1, Side.BUY, 5))

    ex.submit(sell(10, 120, 5))
    reuse = buy(1, 120, 5)  # marketable, so a missed check would trade
    with pytest.raises(ValueError, match="already used"):
        ex.submit(reuse)
    assert reuse.remaining == 5
    assert ex.book.quantity_at_price(Side.SELL, 120) == 5
    check_invariants(ex.book)


def test_rejected_submission_does_not_consume_id():
    ex = Exchange()
    dead = buy(1, 99, 5)
    dead.status = OrderStatus.CANCELLED
    with pytest.raises(ValueError):
        ex.submit(dead)
    ex.submit(buy(1, 99, 5))
    assert ex.book.get_order(1) is not None
