from collections import defaultdict

from market_sim.exchange import Order, OrderStatus, OrderType, Side


def lim(oid, side, price, qty, ts=0, agent=None):
    return Order(order_id=oid, agent_id=oid if agent is None else agent, side=side,
                 order_type=OrderType.LIMIT, qty=qty, timestamp=ts, price=price)


def buy(oid, price, qty, ts=0):
    return lim(oid, Side.BUY, price, qty, ts)


def sell(oid, price, qty, ts=0):
    return lim(oid, Side.SELL, price, qty, ts)


def check_invariants(book):
    """Pokes at internals on purpose -- this is the stuff that goes wrong quietly.

    Only catches structural damage. FIFO and fill correctness need an outside
    reference; see test_reference_model.py.
    """
    seen = set()
    for side in Side:
        levels, prices = book._levels[side], book._prices[side]
        assert prices == sorted(set(levels)), f"{side.name} price list out of sync with levels"
        for price, level in levels.items():
            assert level, f"empty {side.name} level at {price}"
            for o in level:
                assert o.order_type is OrderType.LIMIT
                assert o.side is side and o.price == price
                assert o.remaining > 0 and o.is_active
                assert book._orders[o.order_id] is o
                assert o.order_id not in seen
                seen.add(o.order_id)
            assert book.quantity_at_price(side, price) == sum(o.remaining for o in level) > 0
    assert seen == set(book._orders)

    bids, asks = book._levels[Side.BUY], book._levels[Side.SELL]
    assert book.best_bid == (max(bids) if bids else None)
    assert book.best_ask == (min(asks) if asks else None)
    if bids and asks:
        assert book.best_bid < book.best_ask
        assert book.spread == book.best_ask - book.best_bid
    else:
        assert book.spread is None


def snapshot(book):
    """Everything observable about the resting book, as plain values."""
    return {
        side: [(p, [(o.order_id, o.remaining, o.status) for o in book.orders_at_price(side, p)])
               for p in book._prices[side]]
        for side in Side
    }


def ids_at(ex, side, price):
    return [o.order_id for o in ex.book.orders_at_price(side, price)]


def mkt(oid, side, qty, ts=0):
    return Order(order_id=oid, agent_id=oid, side=side, order_type=OrderType.MARKET, qty=qty, timestamp=ts)


def reconcile(orders, trades):
    """Per-order and per-trade accounting from trade records alone."""
    by_id = {o.order_id: o for o in orders}
    executed = defaultdict(int)
    for i, t in enumerate(trades):
        assert t.trade_id == i + 1
        assert t.qty > 0
        b, s = by_id[t.buy_order_id], by_id[t.sell_order_id]
        assert b.side is Side.BUY and s.side is Side.SELL
        assert (t.buy_agent_id, t.sell_agent_id) == (b.agent_id, s.agent_id)
        aggressor, resting = (b, s) if t.aggressor is Side.BUY else (s, b)
        assert resting.order_type is OrderType.LIMIT
        assert t.price == resting.price
        assert t.timestamp == aggressor.timestamp
        if aggressor.price is not None:
            assert t.price <= aggressor.price if aggressor.side is Side.BUY else t.price >= aggressor.price
        executed[b.order_id] += t.qty
        executed[s.order_id] += t.qty

    for o in orders:
        done = executed[o.order_id]
        assert o.qty == done + o.remaining, f"order {o.order_id} leaks shares"
        match o.status:
            case OrderStatus.FILLED:
                assert o.remaining == 0
            case OrderStatus.CANCELLED:
                assert o.remaining > 0
            case OrderStatus.PARTIALLY_FILLED:
                assert 0 < done < o.qty
            case OrderStatus.OPEN:
                assert done == 0
        if o.order_type is OrderType.MARKET:
            assert o.status in (OrderStatus.FILLED, OrderStatus.CANCELLED)
    return executed
