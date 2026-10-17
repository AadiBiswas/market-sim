import pytest

from helpers import buy, check_invariants, ids_at, mkt, sell
from market_sim.exchange import Exchange, OrderStatus, Side


def three_bids_at_100():
    ex = Exchange()
    for oid in (1, 2, 3):
        ex.submit(buy(oid, 100, 5))
    ex.submit(buy(4, 99, 5))
    return ex


@pytest.mark.parametrize("victim, left", [(1, [2, 3]), (2, [1, 3]), (3, [1, 2])])
def test_cancel_from_any_queue_position_keeps_fifo(victim, left):
    ex = three_bids_at_100()
    o = ex.cancel(victim)

    assert o is not None and o.order_id == victim
    assert o.status is OrderStatus.CANCELLED
    assert ex.book.get_order(victim) is None
    assert ids_at(ex, Side.BUY, 100) == left
    assert ex.book.quantity_at_price(Side.BUY, 100) == 10
    check_invariants(ex.book)


def test_cancel_middle_then_sweep_fills_first_then_third():
    ex = three_bids_at_100()
    ex.cancel(2)
    trades = ex.submit(sell(5, 100, 8))
    assert [(t.buy_order_id, t.qty) for t in trades] == [(1, 5), (3, 3)]
    check_invariants(ex.book)


def test_cancelling_only_order_removes_level_and_updates_best():
    ex = Exchange()
    ex.submit(sell(1, 101, 5))
    ex.submit(sell(2, 103, 5))
    ex.submit(buy(3, 99, 5))
    ex.submit(buy(4, 97, 5))

    ex.cancel(1)
    assert ex.book.best_ask == 103
    assert ex.book.orders_at_price(Side.SELL, 101) == ()
    ex.cancel(3)
    assert ex.book.best_bid == 97
    assert ex.book.spread == 6
    check_invariants(ex.book)

    ex.cancel(2)
    ex.cancel(4)
    assert (ex.book.best_bid, ex.book.best_ask) == (None, None)
    check_invariants(ex.book)


def test_cancel_non_best_level_leaves_best_alone():
    ex = Exchange()
    ex.submit(sell(1, 101, 5))
    ex.submit(sell(2, 102, 5))
    ex.submit(sell(3, 103, 5))
    ex.cancel(2)
    assert ex.book.best_ask == 101
    assert ex.book._prices[Side.SELL] == [101, 103]
    check_invariants(ex.book)


def test_cancel_after_partial_fill_keeps_unexecuted_remaining():
    ex = Exchange()
    resting = sell(1, 101, 10)
    ex.submit(resting)
    ex.submit(buy(2, 101, 4))
    assert resting.status is OrderStatus.PARTIALLY_FILLED

    assert ex.cancel(1) is resting
    # 6 never traded. Not zero -- zero would mean it filled.
    assert (resting.qty, resting.remaining, resting.status) == (10, 6, OrderStatus.CANCELLED)
    check_invariants(ex.book)


def test_cancel_missing_or_dead_orders_returns_none():
    ex = Exchange()
    ex.submit(sell(1, 101, 5))
    ex.submit(buy(2, 101, 5))  # fills 1 and 2 completely
    ex.submit(buy(3, 99, 5))
    ex.cancel(3)
    ex.submit(mkt(4, Side.BUY, 5))  # empty ask side, cancelled on arrival

    assert ex.cancel(999) is None  # never existed
    assert ex.cancel(1) is None  # filled resting order
    assert ex.cancel(2) is None  # filled aggressor
    assert ex.cancel(3) is None  # already cancelled
    assert ex.cancel(4) is None  # market leftover, never rested
    check_invariants(ex.book)
