"""Hand-built sequences that mix operations. Each one is the kind of thing the
random flow hits eventually, pinned down so a failure is readable."""

import pytest

from helpers import buy, check_invariants, ids_at, mkt, reconcile, sell, snapshot
from market_sim.exchange import Exchange, OrderStatus, Side


def test_interior_level_deleted_then_recreated():
    ex = Exchange()
    for oid, px in [(1, 100), (2, 99), (3, 98), (4, 97)]:
        ex.submit(buy(oid, px, 5))

    ex.cancel(3)
    assert ex.book._prices[Side.BUY] == [97, 99, 100]
    ex.submit(buy(5, 98, 4))
    assert ex.book._prices[Side.BUY] == [97, 98, 99, 100]
    check_invariants(ex.book)

    trades = ex.submit(sell(6, 97, 16))
    assert [(t.buy_order_id, t.price, t.qty) for t in trades] == [(1, 100, 5), (2, 99, 5), (5, 98, 4), (4, 97, 2)]
    assert ex.book._prices[Side.BUY] == [97]
    assert ex.book.quantity_at_price(Side.BUY, 97) == 3
    check_invariants(ex.book)


def test_partial_fill_then_cancel():
    ex = Exchange()
    o = buy(1, 100, 20)
    ex.submit(o)
    trades = ex.submit(sell(2, 100, 7))
    assert ex.cancel(1) is o

    assert (o.remaining, o.status) == (13, OrderStatus.CANCELLED)
    assert ex.book.get_order(1) is None
    assert ex.book.best_bid is None
    assert sum(t.qty for t in trades if t.buy_order_id == 1) == 7
    with pytest.raises(ValueError, match="already used"):
        ex.submit(buy(1, 100, 20))
    check_invariants(ex.book)


def test_market_then_limit_sweep_then_cancel():
    ex = Exchange()
    asks = [sell(1, 101, 4), sell(2, 101, 4), sell(3, 102, 5), sell(4, 102, 5), sell(5, 104, 6)]
    for a in asks:
        ex.submit(a)

    m = mkt(6, Side.BUY, 6, ts=10)
    t1 = ex.submit(m)
    assert [(t.sell_order_id, t.price, t.qty) for t in t1] == [(1, 101, 4), (2, 101, 2)]
    assert ids_at(ex, Side.SELL, 101) == [2]

    lb = buy(7, 102, 9, ts=20)
    t2 = ex.submit(lb)
    assert [(t.sell_order_id, t.price, t.qty) for t in t2] == [(2, 101, 2), (3, 102, 5), (4, 102, 2)]
    assert lb.status is OrderStatus.FILLED

    assert ex.cancel(5).remaining == 6  # never touched
    assert [t.trade_id for t in t1 + t2] == [1, 2, 3, 4, 5]
    assert [t.timestamp for t in t1 + t2] == [10, 10, 20, 20, 20]
    assert (ex.book.best_bid, ex.book.best_ask) == (None, 102)
    assert ids_at(ex, Side.SELL, 102) == [4]
    reconcile(asks + [m, lb], t1 + t2)
    check_invariants(ex.book)


def test_fifo_survives_partial_fill_cancel_and_late_arrival():
    ex = Exchange()
    for oid in (1, 2, 3):
        ex.submit(sell(oid, 101, 5))
    ex.submit(buy(10, 101, 2))  # 1 now has 3 left
    ex.cancel(2)
    ex.submit(sell(4, 101, 5))
    assert ids_at(ex, Side.SELL, 101) == [1, 3, 4]

    trades = ex.submit(buy(11, 101, 10))
    assert [(t.sell_order_id, t.qty) for t in trades] == [(1, 3), (3, 5), (4, 2)]
    assert ex.book.get_order(4).remaining == 3
    check_invariants(ex.book)


def resting_book():
    ex = Exchange()
    for o in (buy(1, 99, 5), buy(2, 98, 5), sell(3, 101, 5), sell(4, 101, 5)):
        ex.submit(o)
    ex.submit(buy(5, 101, 2))  # 3 partially filled
    return ex


def _stale_partial():
    # Regression: submit used to check only is_active, so this was accepted
    # and its 3 earlier fills had no trade records behind them.
    o = buy(20, 101, 10)
    o.fill(3)
    return o


def _fiddled_remaining():
    # Status still says OPEN, but remaining was poked from outside.
    o = buy(21, 101, 10)
    o.remaining = 4
    return o


def _cancelled():
    o = buy(22, 101, 10)
    o.cancel()
    return o


@pytest.mark.parametrize("make_bad", [
    lambda: buy(3, 101, 10),       # ID resting on the other side
    lambda: sell(1, 50, 10),       # ID resting on the same side, very marketable
    lambda: mkt(5, Side.SELL, 10), # ID of a filled aggressor
    _stale_partial,
    _fiddled_remaining,
    _cancelled,
])
def test_rejected_submission_changes_nothing(make_bad):
    ex = resting_book()
    before, next_trade = snapshot(ex.book), ex._next_trade_id
    bad = make_bad()
    bad_state = (bad.remaining, bad.status)

    with pytest.raises(ValueError):
        ex.submit(bad)

    assert snapshot(ex.book) == before
    assert ex._next_trade_id == next_trade
    assert (bad.remaining, bad.status) == bad_state
    check_invariants(ex.book)


def test_rejected_for_staleness_doesnt_burn_the_id():
    ex = resting_book()
    with pytest.raises(ValueError, match="isn't fresh"):
        ex.submit(_stale_partial())
    trades = ex.submit(buy(20, 101, 10))
    assert sum(t.qty for t in trades) == 8


def test_reconcile_across_lifecycles():
    ex = Exchange()
    orders, trades = [], []

    def submit(o):
        orders.append(o)
        trades.extend(ex.submit(o))
        return o

    s2 = submit(sell(2, 101, 6))
    s1 = submit(sell(1, 102, 10))
    s3 = submit(sell(3, 104, 8))
    submit(buy(4, 102, 9))           # 6 @101 from s2, 3 @102 from s1
    assert s1.status is OrderStatus.PARTIALLY_FILLED and ex.book.get_order(1) is s1

    b5 = submit(buy(5, 101, 5))      # rests under the 102 ask
    submit(sell(6, 101, 2))          # partially fills b5...
    ex.cancel(5)                     # ...which is then cancelled with 3 unexecuted

    b7 = submit(buy(7, 103, 10))     # finishes s1, rests 3 @103
    assert (b7.status, ex.book.best_bid) == (OrderStatus.PARTIALLY_FILLED, 103)

    submit(mkt(9, Side.BUY, 3))      # nibbles s3
    ex.cancel(3)
    m8 = submit(mkt(8, Side.SELL, 5))  # takes b7's 3, then runs out of bids

    executed = reconcile(orders, trades)
    assert dict(executed) == {2: 6, 1: 10, 3: 3, 4: 9, 5: 2, 6: 2, 7: 10, 9: 3, 8: 3}
    assert [(o.order_id, o.status, o.remaining) for o in (s2, s1, s3, b5, b7, m8)] == [
        (2, OrderStatus.FILLED, 0),
        (1, OrderStatus.FILLED, 0),
        (3, OrderStatus.CANCELLED, 5),
        (5, OrderStatus.CANCELLED, 3),
        (7, OrderStatus.FILLED, 0),
        (8, OrderStatus.CANCELLED, 2),
    ]
    assert (ex.book.best_bid, ex.book.best_ask) == (None, None)
    check_invariants(ex.book)
