import pytest

from market_sim.exchange import Order, OrderBook, OrderStatus, OrderType, Side

def lim(oid, side, price, qty=10):
    return Order(order_id=oid, agent_id=1, side=side, order_type=OrderType.LIMIT,
                 qty=qty, timestamp=0, price=price)


def bid(oid, price, qty=10):
    return lim(oid, Side.BUY, price, qty)


def ask(oid, price, qty=10):
    return lim(oid, Side.SELL, price, qty)


def test_empty_book():
    book = OrderBook()
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.spread is None
    assert book.orders_at_price(Side.BUY, 100) == ()
    assert book.quantity_at_price(Side.SELL, 100) == 0


def test_best_bid_tracks_highest_price():
    book = OrderBook()
    book.add(bid(1, 99))
    assert book.best_bid == 99
    book.add(bid(2, 101))
    assert book.best_bid == 101
    book.add(bid(3, 100))
    assert book.best_bid == 101
    assert book.best_ask is None


def test_best_ask_tracks_lowest_price():
    book = OrderBook()
    book.add(ask(1, 103))
    assert book.best_ask == 103
    book.add(ask(2, 101))
    assert book.best_ask == 101
    book.add(ask(3, 102))
    assert book.best_ask == 101
    assert book.best_bid is None


def test_multiple_levels_both_sides():
    book = OrderBook()
    for o in (bid(1, 98, 5), bid(2, 99, 7), ask(3, 101, 3), ask(4, 104, 4)):
        book.add(o)
    assert (book.best_bid, book.best_ask, book.spread) == (99, 101, 2)
    assert book.quantity_at_price(Side.BUY, 98) == 5
    assert book.quantity_at_price(Side.BUY, 99) == 7
    assert book.quantity_at_price(Side.SELL, 104) == 4
    # Price exists on the other side only.
    assert book.quantity_at_price(Side.SELL, 99) == 0


def test_same_price_preserves_fifo():
    book = OrderBook()
    for oid in (14, 21, 33):
        book.add(bid(oid, 100))
    book.add(bid(5, 99))
    assert [o.order_id for o in book.orders_at_price(Side.BUY, 100)] == [14, 21, 33]


def test_fifo_is_arrival_order_not_order_id():
    book = OrderBook()
    for oid in (9, 2, 5):
        book.add(ask(oid, 105))
    assert [o.order_id for o in book.orders_at_price(Side.SELL, 105)] == [9, 2, 5]


def test_quantity_uses_remaining_not_original():
    book = OrderBook()
    partial = bid(1, 100, qty=10)
    partial.fill(4)
    book.add(partial)
    book.add(bid(2, 100, qty=3))
    assert book.quantity_at_price(Side.BUY, 100) == 9


def test_one_sided_book_has_no_spread():
    book = OrderBook()
    book.add(bid(1, 100))
    assert book.spread is None


def test_duplicate_order_id_rejected():
    book = OrderBook()
    book.add(bid(1, 100))
    with pytest.raises(ValueError):
        book.add(ask(1, 105))
    assert book.best_ask is None
    assert len(book.orders_at_price(Side.BUY, 100)) == 1


def test_market_order_cannot_rest():
    book = OrderBook()
    mkt = Order(order_id=1, agent_id=1, side=Side.BUY, order_type=OrderType.MARKET,
                qty=10, timestamp=0)
    with pytest.raises(ValueError):
        book.add(mkt)
    assert book.get_order(1) is None


def test_dead_orders_cannot_rest():
    filled = bid(1, 100)
    filled.fill(10)
    cancelled = bid(2, 100)
    cancelled.status = OrderStatus.CANCELLED

    book = OrderBook()
    for o in (filled, cancelled):
        with pytest.raises(ValueError):
            book.add(o)
    assert book.best_bid is None


@pytest.mark.parametrize("incoming", [
    bid(100, 105),  # locks the ask
    bid(101, 106),  # crosses it
    ask(102, 100),  # locks the bid
    ask(103, 99),
])
def test_marketable_orders_rejected(incoming):
    book = OrderBook()
    book.add(bid(1, 100))
    book.add(ask(2, 105))
    with pytest.raises(ValueError):
        book.add(incoming)
    assert (book.best_bid, book.best_ask) == (100, 105)
    assert book.get_order(incoming.order_id) is None


def test_get_order():
    book = OrderBook()
    o = bid(7, 100)
    book.add(o)
    assert book.get_order(7) is o
    assert book.get_order(8) is None


def test_returned_level_cannot_corrupt_book():
    book = OrderBook()
    book.add(bid(1, 100))
    book.add(bid(2, 100))
    level = book.orders_at_price(Side.BUY, 100)
    with pytest.raises((TypeError, AttributeError)):
        level.append(bid(3, 100))  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        level[0] = bid(4, 100)  # type: ignore[index]
    assert [o.order_id for o in book.orders_at_price(Side.BUY, 100)] == [1, 2]


def test_every_order_indexed_exactly_once():
    book = OrderBook()
    orders = [bid(1, 98), bid(2, 99), bid(3, 99), ask(4, 101), ask(5, 101), ask(6, 103)]
    for o in orders:
        book.add(o)
    for o in orders:
        assert book.get_order(o.order_id) is o
        level = book.orders_at_price(o.side, o.price)
        assert sum(x is o for x in level) == 1
        other = Side.SELL if o.side is Side.BUY else Side.BUY
        assert book.orders_at_price(other, o.price) == ()
