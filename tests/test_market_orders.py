from helpers import buy, check_invariants, ids_at, mkt, sell
from market_sim.exchange import Exchange, OrderStatus, Side


def test_market_buy_hits_best_ask():
    ex = Exchange()
    ex.submit(sell(1, 101, 5))
    ex.submit(sell(2, 103, 5))
    o = mkt(3, Side.BUY, 5, ts=7)
    trades = ex.submit(o)

    assert [(t.price, t.qty, t.sell_order_id, t.buy_order_id, t.timestamp) for t in trades] == [(101, 5, 1, 3, 7)]
    assert trades[0].aggressor is Side.BUY
    assert (o.remaining, o.status) == (0, OrderStatus.FILLED)
    assert ex.book.best_ask == 103
    check_invariants(ex.book)


def test_market_sell_hits_best_bid():
    ex = Exchange()
    ex.submit(buy(1, 99, 5))
    ex.submit(buy(2, 97, 5))
    o = mkt(3, Side.SELL, 3)
    trades = ex.submit(o)

    assert [(t.price, t.qty, t.buy_order_id) for t in trades] == [(99, 3, 1)]
    assert trades[0].aggressor is Side.SELL
    assert o.status is OrderStatus.FILLED
    assert ex.book.quantity_at_price(Side.BUY, 99) == 2
    check_invariants(ex.book)


def test_market_order_walks_levels_fifo_at_resting_prices():
    ex = Exchange()
    for o in (sell(1, 101, 3), sell(2, 101, 2), sell(3, 102, 4), sell(4, 110, 6)):
        ex.submit(o)
    o = mkt(5, Side.BUY, 12)
    trades = ex.submit(o)

    # No limit, so it'll happily pay 110. That's what market orders do.
    assert [(t.sell_order_id, t.price, t.qty) for t in trades] == [(1, 101, 3), (2, 101, 2), (3, 102, 4), (4, 110, 3)]
    assert o.status is OrderStatus.FILLED
    assert ids_at(ex, Side.SELL, 110) == [4]
    assert ex.book.quantity_at_price(Side.SELL, 110) == 3
    check_invariants(ex.book)


def test_market_order_exhausts_book_and_cancels_remainder():
    ex = Exchange()
    asks = [sell(1, 101, 3), sell(2, 102, 4)]
    for a in asks:
        ex.submit(a)
    o = mkt(3, Side.BUY, 10)
    trades = ex.submit(o)

    assert [(t.price, t.qty) for t in trades] == [(101, 3), (102, 4)]
    assert (o.remaining, o.status) == (3, OrderStatus.CANCELLED)
    executed = sum(t.qty for t in trades)
    assert executed + o.remaining == o.qty
    assert all(a.status is OrderStatus.FILLED and a.remaining == 0 for a in asks)

    # Never rests, on either side.
    assert ex.book.get_order(3) is None
    assert (ex.book.best_bid, ex.book.best_ask) == (None, None)
    check_invariants(ex.book)


def test_market_order_into_empty_book_cancels_untouched():
    ex = Exchange()
    ex.submit(buy(1, 99, 5))  # same side only; nothing to hit
    o = mkt(2, Side.BUY, 10)
    trades = ex.submit(o)

    assert trades == []
    assert (o.remaining, o.status) == (10, OrderStatus.CANCELLED)
    assert ex.book.get_order(2) is None
    assert ids_at(ex, Side.BUY, 99) == [1]
    check_invariants(ex.book)


def test_market_sweep_conserves_resting_quantity():
    ex = Exchange()
    bids = [buy(1, 100, 4), buy(2, 100, 6), buy(3, 98, 5)]
    for b in bids:
        ex.submit(b)
    o = mkt(4, Side.SELL, 12)
    trades = ex.submit(o)

    executed = sum(t.qty for t in trades)
    assert executed + o.remaining == o.qty == 12
    assert sum(b.qty - b.remaining for b in bids) == executed
    assert ex.book.quantity_at_price(Side.BUY, 98) == 3
    check_invariants(ex.book)
