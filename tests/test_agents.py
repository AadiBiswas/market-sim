"""Simulator <-> agent plumbing, using a scripted agent so nothing is random."""

import dataclasses

import pytest

from helpers import check_invariants
from market_sim.exchange import OrderStatus, OrderType, Side, Trade
from market_sim.simulation import (
    CancelOrder, FundamentalProcess, MarketObservation, OrderRequest, Simulator, SubmitOrder,
    Submitted, UpdateFundamental, WakeAgent,
)


class Scripted:
    """Returns pre-baked requests in order and remembers what it saw."""

    def __init__(self, agent_id, *requests):
        self.agent_id = agent_id
        self.requests = list(requests)
        self.seen = []

    def on_wake(self, obs):
        self.seen.append(obs)
        return self.requests.pop(0) if self.requests else None


def limit(oid, side, price, qty, agent=99):
    return SubmitOrder(oid, agent, side, OrderType.LIMIT, qty, price)


def req(side, qty, price=None):
    return OrderRequest(side, OrderType.MARKET if price is None else OrderType.LIMIT, qty, price)


def submitted(sim):
    return [r for r in sim.log if isinstance(r, Submitted)]


# -- observations --

def test_empty_one_sided_and_two_sided_observations():
    sim = Simulator()
    a = Scripted(1)
    sim.register_agent(a)
    sim.schedule(10, WakeAgent(1))
    sim.schedule(20, limit(101, Side.BUY, 99, 4))
    sim.schedule(20, limit(102, Side.BUY, 99, 6))
    sim.schedule(20, limit(103, Side.BUY, 98, 50))
    sim.schedule(30, WakeAgent(1))
    sim.schedule(40, limit(104, Side.SELL, 102, 3))
    sim.schedule(40, limit(105, Side.SELL, 103, 9))
    sim.schedule(50, WakeAgent(1))
    sim.run()

    assert a.seen == [
        MarketObservation(10, None, None, 0, 0),
        MarketObservation(30, 99, None, 10, 0),
        MarketObservation(50, 99, 102, 10, 3),
    ]


def test_observation_is_a_frozen_snapshot_of_plain_values():
    sim = Simulator(fundamental=FundamentalProcess(10_000.0, 2.0, seed=1))
    a = Scripted(1)
    sim.register_agent(a)
    sim.schedule(10, limit(101, Side.SELL, 102, 5))
    sim.schedule(20, UpdateFundamental())
    sim.schedule(20, WakeAgent(1))
    sim.schedule(30, SubmitOrder(102, 98, Side.BUY, OrderType.MARKET, 5))
    sim.schedule(40, WakeAgent(1))
    sim.run()

    first, second = a.seen
    # Earlier snapshot unchanged by the trade that emptied the ask side.
    assert first == MarketObservation(20, None, 102, 0, 5)
    assert second == MarketObservation(40, None, None, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.best_ask = 1  # type: ignore[misc]
    # Nothing but ints/None in there: no book, no orders, no fundamental.
    assert {f.name for f in dataclasses.fields(first)} == {"timestamp", "best_bid", "best_ask", "bid_qty", "ask_qty"}
    assert all(v is None or type(v) is int for v in dataclasses.astuple(first))


def test_agent_sees_the_past_but_not_the_future():
    sim = Simulator()
    a = Scripted(1)
    sim.register_agent(a)
    sim.schedule(100, limit(101, Side.BUY, 99, 5))
    sim.schedule(200, WakeAgent(1))
    sim.schedule(300, limit(102, Side.SELL, 101, 5))
    sim.run()
    (obs,) = a.seen
    assert (obs.best_bid, obs.best_ask) == (99, None)


def test_same_time_visibility_follows_scheduling_order():
    sim = Simulator()
    a = Scripted(1)
    sim.register_agent(a)
    sim.schedule(100, limit(101, Side.BUY, 99, 5))
    sim.schedule(100, WakeAgent(1))
    sim.schedule(100, limit(102, Side.SELL, 101, 5))  # same instant, queued after the wake
    sim.run()
    (obs,) = a.seen
    assert (obs.best_bid, obs.best_ask) == (99, None)


# -- registration and wake-ups --

def test_duplicate_registration_rejected():
    sim = Simulator()
    sim.register_agent(Scripted(1))
    with pytest.raises(ValueError, match="already registered"):
        sim.register_agent(Scripted(1))


def test_waking_unknown_agent_fails_at_schedule_time():
    sim = Simulator()
    sim.register_agent(Scripted(1))
    with pytest.raises(ValueError, match="unregistered agent 2"):
        sim.schedule(100, WakeAgent(2))


def test_wake_with_no_request_does_nothing_public():
    sim = Simulator()
    a = Scripted(1)
    sim.register_agent(a)
    sim.schedule(100, WakeAgent(1))
    sim.schedule(250, WakeAgent(1))
    sim.run()
    assert [o.timestamp for o in a.seen] == [100, 250]
    assert sim.log == ()
    assert sim.now == 250


# -- agent submissions --

def test_agent_order_goes_through_normal_submit_path_at_wake_time():
    sim = Simulator()
    maker, taker = Scripted(1, req(Side.SELL, 5, 101)), Scripted(2, req(Side.BUY, 3))
    sim.register_agent(maker)
    sim.register_agent(taker)
    sim.schedule(100, WakeAgent(1))
    sim.schedule(700, WakeAgent(2))
    sim.run()

    assert submitted(sim) == [
        Submitted(100, 1, 1, Side.SELL, OrderType.LIMIT, 5, 101),
        Submitted(700, 2, 2, Side.BUY, OrderType.MARKET, 3, None),
    ]
    (t,) = sim.trades
    assert (t.timestamp, t.price, t.qty, t.sell_agent_id, t.buy_agent_id) == (700, 101, 3, 1, 2)
    assert sim.exchange.book.get_order(1).timestamp == 100


def test_zero_delay_submission_runs_after_events_already_queued_for_that_instant():
    sim = Simulator()
    a = Scripted(1, req(Side.BUY, 5, 101))
    sim.register_agent(a)
    sim.schedule(100, WakeAgent(1))
    sim.schedule(100, limit(500, Side.SELL, 101, 5))  # queued before the agent's order exists
    sim.run()

    # The manual ask rested first, so the agent's bid is the aggressor.
    assert [r.order_id for r in submitted(sim)] == [500, 1]
    (t,) = sim.trades
    assert (t.aggressor, t.buy_agent_id, t.timestamp) == (Side.BUY, 1, 100)


def test_bad_request_fails_at_dispatch_with_context():
    sim = Simulator()
    sim.register_agent(Scripted(1, OrderRequest(Side.BUY, OrderType.LIMIT, 0, 100)))
    sim.schedule(100, WakeAgent(1))
    with pytest.raises(ValueError, match="qty must be positive") as exc:
        sim.run()
    assert any("t=100" in n for n in exc.value.__notes__)


# -- order IDs --

def test_ids_unique_across_agents_and_not_reused_after_fill_or_cancel():
    sim = Simulator()
    a = Scripted(1, req(Side.SELL, 5, 101), req(Side.SELL, 5, 103))
    b = Scripted(2, req(Side.BUY, 5), req(Side.BUY, 2, 90))
    sim.register_agent(a)
    sim.register_agent(b)
    sim.schedule(100, WakeAgent(1))   # id 1, rests
    sim.schedule(200, WakeAgent(2))   # id 2, fills 1 completely
    sim.schedule(300, WakeAgent(2))   # id 3, rests
    sim.schedule(400, CancelOrder(3))
    sim.schedule(500, WakeAgent(1))   # id 4 -- not 1 or 3
    sim.run()

    subs = submitted(sim)
    assert [(s.order_id, s.agent_id) for s in subs] == [(1, 1), (2, 2), (3, 2), (4, 1)]
    (t,) = sim.trades
    assert (t.sell_order_id, t.sell_agent_id, t.buy_order_id, t.buy_agent_id) == (1, 1, 2, 2)


def test_allocator_skips_ids_already_scheduled_by_hand():
    sim = Simulator()
    a = Scripted(1, *(req(Side.BUY, 1, 90 + i) for i in range(4)))
    sim.register_agent(a)
    sim.schedule(50, limit(2, Side.SELL, 200, 1))
    sim.schedule(900, limit(3, Side.SELL, 201, 1))  # future, but already claimed
    for t in (100, 200, 300, 400):
        sim.schedule(t, WakeAgent(1))
    sim.run()
    assert [s.order_id for s in submitted(sim) if s.agent_id == 1] == [1, 4, 5, 6]
    check_invariants(sim.exchange.book)


def test_late_manual_id_collision_still_rejected_by_exchange():
    # The documented gap: allocator already handed out 1, then someone schedules 1 by hand.
    sim = Simulator()
    sim.register_agent(Scripted(1, req(Side.BUY, 1, 90)))
    sim.schedule(100, WakeAgent(1))
    sim.run()
    sim.schedule(200, limit(1, Side.SELL, 200, 1))
    with pytest.raises(ValueError, match="already used"):
        sim.run()
    assert sim.exchange.book.get_order(1).agent_id == 1


def test_agent_order_status_lives_in_the_log_not_in_the_agent():
    # Agents get nothing back -- no Order handle to poke at.
    sim = Simulator()
    a = Scripted(1, req(Side.SELL, 5, 101))
    sim.register_agent(a)
    sim.schedule(100, WakeAgent(1))
    sim.schedule(200, SubmitOrder(50, 9, Side.BUY, OrderType.MARKET, 2))
    sim.run()
    assert sim.exchange.book.get_order(1).status is OrderStatus.PARTIALLY_FILLED
    assert [type(r) for r in sim.log] == [Submitted, Submitted, Trade]
