import inspect

import numpy as np
import pytest

from helpers import check_invariants, reconcile, snapshot
from market_sim.agents import NoiseTrader
from market_sim.exchange import Order, OrderStatus, OrderType, Side
from market_sim.simulation import (
    Cancelled, FundamentalProcess, MarketObservation, Simulator, Submitted, UpdateFundamental, WakeAgent,
)

SEC = 1_000_000_000

EMPTY = MarketObservation(0, None, None, 0, 0)
BIDS_ONLY = MarketObservation(0, 9_998, None, 5, 0)
ASKS_ONLY = MarketObservation(0, None, 10_002, 0, 5)
TWO_SIDED = MarketObservation(0, 9_999, 10_001, 5, 5)
TIGHT = MarketObservation(0, 9_999, 10_000, 5, 5)
OBS = [EMPTY, BIDS_ONLY, ASKS_ONLY, TWO_SIDED, TIGHT]


def trader(seed=1, mode="passive", **kw):
    kw.setdefault("reference_price", 10_000)
    return NoiseTrader(agent_id=1, seed=seed, mode=mode, **kw)


def many(t, obs, n=400):
    return [t.on_wake(obs) for _ in range(n)]


# -- decisions --

def test_same_seed_same_observations_same_requests():
    seq = OBS * 40
    a, b = trader(seed=7), trader(seed=7)
    assert [a.on_wake(o) for o in seq] == [b.on_wake(o) for o in seq]
    assert many(trader(seed=7), TWO_SIDED) != many(trader(seed=8), TWO_SIDED)


@pytest.mark.parametrize("mode", ["passive", "aggressive"])
def test_both_sides_occur(mode):
    sides = {r.side for r in many(trader(mode=mode), TWO_SIDED)}
    assert sides == {Side.BUY, Side.SELL}


@pytest.mark.parametrize("obs", OBS)
def test_passive_orders_never_touch_the_far_side_and_are_valid(obs):
    for r in many(trader(max_offset=5, qty_range=(2, 7)), obs):
        assert r.order_type is OrderType.LIMIT
        assert type(r.price) is int and type(r.qty) is int
        assert 2 <= r.qty <= 7
        if r.side is Side.BUY and obs.best_ask is not None:
            assert r.price < obs.best_ask
        if r.side is Side.SELL and obs.best_bid is not None:
            assert r.price > obs.best_bid
        # Must survive Order's own validation.
        Order(1, 1, r.side, r.order_type, r.qty, 0, r.price)


def test_passive_prices_cluster_around_the_right_anchor():
    reqs = many(trader(max_offset=2), EMPTY)
    assert {r.price for r in reqs} == set(range(9_998, 10_003))
    bids = {r.price for r in many(trader(max_offset=2), BIDS_ONLY) if r.side is Side.BUY}
    assert bids == set(range(9_996, 10_001))
    # Tight book: everything gets pushed out to the touch or beyond.
    tight = many(trader(max_offset=2), TIGHT)
    assert max(r.price for r in tight if r.side is Side.BUY) == 9_999
    assert min(r.price for r in tight if r.side is Side.SELL) == 10_000


def test_aggressive_orders_are_market_orders():
    for obs in OBS:
        for r in many(trader(mode="aggressive"), obs, n=50):
            assert (r.order_type, r.price) == (OrderType.MARKET, None)
            Order(1, 1, r.side, r.order_type, r.qty, 0, r.price)


def test_act_prob_and_buy_prob():
    assert many(trader(act_prob=0.0), TWO_SIDED, n=50) == [None] * 50
    assert {r.side for r in many(trader(buy_prob=1.0), TWO_SIDED)} == {Side.BUY}
    reqs = many(trader(act_prob=0.3), TWO_SIDED, n=2_000)
    assert 0.25 < sum(r is not None for r in reqs) / 2_000 < 0.35


def test_no_passive_price_below_one_tick():
    assert many(trader(buy_prob=1.0, reference_price=1), MarketObservation(0, None, 1, 0, 3), n=20) == [None] * 20


def test_bad_config_rejected():
    for kw in [dict(mode="sneaky"), dict(act_prob=1.5), dict(qty_range=(0, 3)), dict(qty_range=(5, 2)),
               dict(max_offset=-1), dict(reference_price=0)]:
        with pytest.raises(ValueError):
            trader(**kw)


def test_rng_use_is_fixed_per_wake():
    # Draws don't depend on what the agent saw or whether it acted.
    a, b = trader(seed=3), trader(seed=3, mode="aggressive", act_prob=0.0)
    for o in OBS * 10:
        a.on_wake(o)
        b.on_wake(EMPTY)
    assert a._rng.bit_generator.state == b._rng.bit_generator.state


def test_trader_has_no_route_to_the_fundamental():
    params = inspect.signature(NoiseTrader).parameters
    assert not any("fundamental" in p for p in params)
    assert inspect.signature(NoiseTrader.on_wake).parameters.keys() == {"self", "obs"}


# -- in the simulator --

def build(seeds=(11, 12, 21, 22), extra_wakes_for=None, fundamental_seed=99, fundamental_updates=True):
    """Two passive makers (1, 2) and two aggressive takers (3, 4). Separate
    agent IDs for each role, so nobody trades against themselves."""
    sim = Simulator(fundamental=FundamentalProcess(10_000.0, 1.5, seed=fundamental_seed))
    agents = [
        NoiseTrader(1, seeds[0], "passive", 10_000, qty_range=(1, 8), max_offset=3),
        NoiseTrader(2, seeds[1], "passive", 10_000, qty_range=(1, 8), max_offset=3),
        NoiseTrader(3, seeds[2], "aggressive", 10_000, act_prob=0.8, qty_range=(1, 12)),
        NoiseTrader(4, seeds[3], "aggressive", 10_000, act_prob=0.8, qty_range=(1, 12)),
    ]
    for a in agents:
        sim.register_agent(a)
    if fundamental_updates:
        for k in range(1, 11):
            sim.schedule(k * SEC, UpdateFundamental())
    # Takers wake once before anyone has posted, to exercise the empty book.
    sim.schedule(0, WakeAgent(3))
    for k in range(40):
        t = k * SEC // 4 + 1
        sim.schedule(t, WakeAgent(1 + k % 2))
        sim.schedule(t, WakeAgent(1 + (k + 1) % 2))
        sim.schedule(t + 10, WakeAgent(3 + k % 2))
    for k in range(25 if extra_wakes_for else 0):
        sim.schedule(k * SEC // 3 + 5, WakeAgent(extra_wakes_for))
    return sim, {a.agent_id: a for a in agents}


def test_multi_agent_session_is_consistent():
    sim, agents = build()
    sim.run()

    subs = [r for r in sim.log if isinstance(r, Submitted)]
    ids = [s.order_id for s in subs]
    assert len(ids) == len(set(ids))
    by_type = {t: [s for s in subs if s.order_type is t] for t in OrderType}
    assert by_type[OrderType.LIMIT] and by_type[OrderType.MARKET]
    assert {s.agent_id for s in by_type[OrderType.LIMIT]} == {1, 2}
    assert {s.agent_id for s in by_type[OrderType.MARKET]} == {3, 4}

    trades = sim.trades
    assert len(trades) > 20
    for t in trades:
        assert t.sell_agent_id != t.buy_agent_id
    # Some takers came up short and had the remainder cancelled.
    assert any(isinstance(r, Cancelled) for r in sim.log)

    # Rebuild every order from its Submitted record and replay the log through
    # the shared reconciliation check. Uses no live Order objects.
    orders = {s.order_id: Order(s.order_id, s.agent_id, s.side, s.order_type, s.qty, s.timestamp, s.price)
              for s in subs}
    for r in sim.log:
        if isinstance(r, Cancelled):
            orders[r.order_id].cancel()
        elif not isinstance(r, Submitted):
            orders[r.buy_order_id].fill(r.qty)
            orders[r.sell_order_id].fill(r.qty)
    reconcile(list(orders.values()), trades)
    for oid, o in orders.items():
        live = sim.exchange.book.get_order(oid)
        assert (live is not None) == o.is_active
        if live is not None:
            assert (live.remaining, live.status) == (o.remaining, o.status)
    check_invariants(sim.exchange.book)


def test_replay_with_fresh_agents_is_identical():
    a, _ = build()
    b, _ = build()
    a.run()
    b.run()
    assert a.log == b.log
    assert a.fundamental_history == b.fundamental_history
    assert snapshot(a.exchange.book) == snapshot(b.exchange.book)
    assert a.now == b.now


def test_changing_one_seed_changes_the_market():
    a, _ = build()
    b, _ = build(seeds=(11, 12, 21, 23))
    a.run()
    b.run()
    assert a.log != b.log
    assert a.fundamental_history == b.fundamental_history


def test_extra_wakes_for_one_agent_dont_touch_anothers_rng():
    plain, plain_agents = build()
    busy, busy_agents = build(extra_wakes_for=3)
    plain.run()
    busy.run()
    # Agent 4's draws are identical; its decisions may differ because the book did.
    assert plain_agents[4]._rng.bit_generator.state == busy_agents[4]._rng.bit_generator.state
    assert plain_agents[3]._rng.bit_generator.state != busy_agents[3]._rng.bit_generator.state
    assert plain.fundamental_history == busy.fundamental_history


def test_fundamental_updates_dont_touch_agent_rngs_or_decisions():
    with_f, a1 = build()
    without_f, a2 = build(fundamental_updates=False)
    with_f.run()
    without_f.run()
    for aid in a1:
        assert a1[aid]._rng.bit_generator.state == a2[aid]._rng.bit_generator.state
    # Agents can't see the fundamental, so the market is literally the same.
    assert with_f.log == without_f.log


def test_agents_never_see_the_numpy_global_rng():
    np.random.seed(0)
    a, _ = build()
    a.run()
    np.random.seed(12345)
    np.random.standard_normal(100)
    b, _ = build()
    b.run()
    assert a.log == b.log


def test_empty_book_market_order_just_cancels():
    sim, _ = build()
    sim.run()
    first = sim.log[:2]
    assert first[0] == Submitted(0, 1, 3, first[0].side, OrderType.MARKET, first[0].qty, None)
    assert isinstance(first[1], Cancelled) and first[1].remaining == first[0].qty
