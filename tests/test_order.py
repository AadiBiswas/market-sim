import pytest

from market_sim.exchange import Order, OrderStatus, OrderType, Side


def limit(qty=10, price=100, side=Side.BUY, **kw):
    return Order(order_id=1, agent_id=7, side=side, order_type=OrderType.LIMIT,
                 qty=qty, timestamp=0, price=price, **kw)


def test_limit_order_starts_open_and_unfilled():
    o = limit(qty=10, price=101)
    assert o.price == 101
    assert o.remaining == 10
    assert o.status is OrderStatus.OPEN
    assert o.is_active


def test_market_order_has_no_price():
    o = Order(order_id=2, agent_id=7, side=Side.SELL, order_type=OrderType.MARKET,
              qty=5, timestamp=0)
    assert o.price is None
    assert o.remaining == 5


@pytest.mark.parametrize("qty", [0, -3])
def test_nonpositive_qty_rejected(qty):
    with pytest.raises(ValueError):
        limit(qty=qty)


@pytest.mark.parametrize("price", [None, 0, -1, 100.5])
def test_bad_limit_price_rejected(price):
    with pytest.raises(ValueError):
        limit(price=price)


def test_market_order_with_price_rejected():
    with pytest.raises(ValueError):
        Order(order_id=2, agent_id=7, side=Side.BUY, order_type=OrderType.MARKET,
              qty=5, timestamp=0, price=100)


def test_remaining_and_status_not_constructor_args():
    with pytest.raises(TypeError):
        limit(remaining=3)


def test_partial_then_full_fill():
    o = limit(qty=10)
    o.fill(4)
    assert (o.remaining, o.status) == (6, OrderStatus.PARTIALLY_FILLED)
    o.fill(6)
    assert (o.remaining, o.status) == (0, OrderStatus.FILLED)
    assert not o.is_active
    assert o.qty == 10


@pytest.mark.parametrize("qty", [0, -1, 11])
def test_fill_out_of_range_rejected_and_state_unchanged(qty):
    o = limit(qty=10)
    with pytest.raises(ValueError):
        o.fill(qty)
    assert (o.remaining, o.status) == (10, OrderStatus.OPEN)


def test_cannot_fill_inactive_order():
    o = limit(qty=10)
    o.fill(10)
    with pytest.raises(ValueError):
        o.fill(1)

    o = limit(qty=10)
    o.status = OrderStatus.CANCELLED
    with pytest.raises(ValueError):
        o.fill(1)


@pytest.mark.parametrize("attr", ["qty", "price", "side", "order_id", "timestamp"])
def test_submission_fields_are_immutable(attr):
    o = limit()
    with pytest.raises(AttributeError):
        setattr(o, attr, getattr(o, attr))


def test_enums_are_distinct():
    assert Side.BUY != Side.SELL
    assert {s.name for s in OrderStatus} == {"OPEN", "PARTIALLY_FILLED", "FILLED", "CANCELLED"}
