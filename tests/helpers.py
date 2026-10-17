from market_sim.exchange import Order, OrderType, Side


def lim(oid, side, price, qty, ts=0, agent=None):
    return Order(order_id=oid, agent_id=oid if agent is None else agent, side=side,
                 order_type=OrderType.LIMIT, qty=qty, timestamp=ts, price=price)


def buy(oid, price, qty, ts=0):
    return lim(oid, Side.BUY, price, qty, ts)


def sell(oid, price, qty, ts=0):
    return lim(oid, Side.SELL, price, qty, ts)


def check_invariants(book):
    """Pokes at internals on purpose -- this is the stuff that goes wrong quietly."""
    seen = set()
    for side in Side:
        levels, prices = book._levels[side], book._prices[side]
        assert prices == sorted(levels), f"{side.name} price list out of sync with levels"
        for price, level in levels.items():
            assert level, f"empty {side.name} level at {price}"
            for o in level:
                assert o.order_type is OrderType.LIMIT
                assert o.side is side and o.price == price
                assert o.remaining > 0 and o.is_active
                assert book._orders[o.order_id] is o
                assert o.order_id not in seen
                seen.add(o.order_id)
    assert seen == set(book._orders)
    if book.best_bid is not None and book.best_ask is not None:
        assert book.best_bid < book.best_ask


def ids_at(ex, side, price):
    return [o.order_id for o in ex.book.orders_at_price(side, price)]


def mkt(oid, side, qty, ts=0):
    return Order(order_id=oid, agent_id=oid, side=side, order_type=OrderType.MARKET, qty=qty, timestamp=ts)
