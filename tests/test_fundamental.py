import dataclasses
import math

import numpy as np
import pytest

from helpers import check_invariants, snapshot
from market_sim.exchange import OrderType, Side
from market_sim.simulation import (
    CancelOrder, FundamentalProcess, FundamentalSnapshot, Simulator, SubmitOrder, UpdateFundamental,
)

SEC = 1_000_000_000


def proc(seed=42, sigma=2.0, v0=10_000.0):
    return FundamentalProcess(v0, sigma, seed=seed)


def path(process, times):
    return [process.advance_to(t) for t in times]


def history_values(sim):
    return [(s.timestamp, s.value) for s in sim.fundamental_history]


# -- the maths --

def test_increment_matches_formula_with_independent_draws():
    # Same generator and seed, driven by hand. Pins sigma, sqrt(dt) and ns -> s together.
    z = np.random.default_rng(7).standard_normal(2)
    p = proc(seed=7, sigma=3.0, v0=500.0)

    p.advance_to(SEC // 4)
    assert p.value == pytest.approx(500.0 + 3.0 * 0.5 * z[0])
    p.advance_to(SEC // 4 + 9 * SEC)
    assert p.value == pytest.approx(500.0 + 3.0 * 0.5 * z[0] + 3.0 * 3.0 * z[1])
    assert p.timestamp == SEC // 4 + 9 * SEC


def test_shock_scales_with_sqrt_time():
    # Same seed means same first Z, so only the time scaling differs.
    d1 = proc().advance_to(SEC) - 10_000.0
    d4 = proc().advance_to(4 * SEC) - 10_000.0
    assert d1 != 0
    assert d4 / d1 == pytest.approx(2.0)


def test_shock_scales_linearly_with_sigma():
    d1 = proc(sigma=1.0).advance_to(SEC) - 10_000.0
    d5 = proc(sigma=5.0).advance_to(SEC) - 10_000.0
    assert d5 / d1 == pytest.approx(5.0)


def test_zero_sigma_never_moves():
    p = proc(sigma=0.0)
    assert path(p, [1, SEC, 1000 * SEC]) == [10_000.0] * 3
    assert p.timestamp == 1000 * SEC


def test_value_is_not_rounded_to_ticks():
    assert not proc().advance_to(SEC).is_integer()


def test_same_timestamp_is_a_true_noop():
    p = proc()
    p.advance_to(SEC)
    value, rng_state = p.value, p._rng.bit_generator.state
    assert p.advance_to(SEC) == value
    assert p.timestamp == SEC
    assert p._rng.bit_generator.state == rng_state


def test_advancing_to_zero_at_start_draws_nothing():
    p, fresh = proc(), proc()
    assert p.advance_to(0) == 10_000.0
    assert p.advance_to(SEC) == fresh.advance_to(SEC)


def test_cannot_go_backwards():
    p = proc()
    p.advance_to(SEC)
    value = p.value
    with pytest.raises(ValueError):
        p.advance_to(SEC - 1)
    assert (p.value, p.timestamp) == (value, SEC)


@pytest.mark.parametrize("v0, sigma", [
    (math.nan, 1.0), (math.inf, 1.0), (-math.inf, 1.0),
    (100.0, -0.1), (100.0, math.nan), (100.0, math.inf),
])
def test_bad_parameters_rejected(v0, sigma):
    with pytest.raises(ValueError):
        FundamentalProcess(v0, sigma, seed=1)


@pytest.mark.parametrize("dt_seconds", [0.25, 4.0])
def test_increments_have_the_right_moments(dt_seconds):
    # Fixed seed, so this can't flake. Tolerances are 5 standard errors, which
    # is loose enough to survive any seed and tight enough to catch a units bug.
    n, sigma = 20_000, 2.0
    step = int(dt_seconds * SEC)
    p = FundamentalProcess(10_000.0, sigma, seed=123)
    values = np.array([10_000.0] + path(p, [step * i for i in range(1, n + 1)]))
    dv = np.diff(values)

    want_var = sigma**2 * dt_seconds
    assert abs(dv.mean()) < 5 * math.sqrt(want_var / n)
    assert abs(dv.var(ddof=1) - want_var) < 5 * want_var * math.sqrt(2 / n)
    # Increments independent: lag-1 autocorrelation is about 0, SE 1/sqrt(n).
    assert abs(np.corrcoef(dv[:-1], dv[1:])[0, 1]) < 5 / math.sqrt(n)


# -- reproducibility --

TIMES = [SEC, 3 * SEC, 3 * SEC + 17, 10 * SEC]


def test_same_seed_same_schedule_same_path():
    assert path(proc(seed=5), TIMES) == path(proc(seed=5), TIMES)


def test_different_seeds_differ():
    assert path(proc(seed=5), TIMES) != path(proc(seed=6), TIMES)


def test_instances_do_not_share_rng_state():
    # Driving another process (or numpy's global RNG) in between changes nothing.
    alone = path(proc(seed=5), TIMES)
    a, other = proc(seed=5), proc(seed=5)
    interleaved = []
    for t in TIMES:
        other.advance_to(t * 2)
        np.random.standard_normal(3)
        interleaved.append(a.advance_to(t))
    assert interleaved == alone


# -- simulator integration --

def limit(oid, side, price, qty):
    return SubmitOrder(oid, oid, side, OrderType.LIMIT, qty, price)


def market(oid, side, qty):
    return SubmitOrder(oid, oid, side, OrderType.MARKET, qty)


UPDATES = [100, 200, 300]


def test_history_has_initial_state_then_each_update_in_time_order():
    p = proc()
    sim = Simulator(fundamental=p)
    assert sim.fundamental_history == (FundamentalSnapshot(0, 10_000.0),)

    # Out of order on purpose; the heap sorts it out.
    for t in (300, 100, 200):
        sim.schedule(t, UpdateFundamental())
    sim.run()

    assert history_values(sim) == list(zip([0] + UPDATES, [10_000.0] + path(proc(), UPDATES)))
    assert sim.now == 300
    assert p.timestamp == 300


def test_snapshots_are_immutable_and_detached_from_the_process():
    p = proc()
    sim = Simulator(fundamental=p)
    sim.schedule(100, UpdateFundamental())
    sim.run()
    before = sim.fundamental_history

    assert isinstance(before, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        before[0].value = 1.0  # type: ignore[misc]

    sim.schedule(200, UpdateFundamental())
    sim.run()
    assert sim.fundamental_history[:2] == before


def test_trading_events_do_not_touch_the_process():
    p = proc()
    sim = Simulator(fundamental=p)
    sim.schedule(100, UpdateFundamental())
    sim.schedule(150, limit(1, Side.BUY, 9990, 10))
    sim.schedule(160, market(2, Side.SELL, 4))
    sim.schedule(170, CancelOrder(1))
    sim.schedule(180, CancelOrder(99))
    sim.schedule(190, limit(3, Side.SELL, 10_010, 5))
    sim.run()

    # Clock moved on; the process stayed at its last explicit update.
    assert sim.now == 190
    assert p.timestamp == 100
    assert len(sim.fundamental_history) == 2


def test_extra_trading_events_do_not_change_the_path():
    # The Monte Carlo assumption: strategies with different order flow see the
    # same fundamentals, given the same seed and update schedule.
    def run(with_trading):
        sim = Simulator(fundamental=proc(seed=11))
        for t in UPDATES:
            sim.schedule(t, UpdateFundamental())
        if with_trading:
            sim.schedule(50, limit(1, Side.BUY, 9990, 10))
            sim.schedule(100, limit(2, Side.SELL, 9990, 3))  # tied with an update
            sim.schedule(150, CancelOrder(1))
            sim.schedule(250, market(3, Side.BUY, 5))
            sim.schedule(400, limit(4, Side.BUY, 9000, 1))
        sim.run()
        return sim

    quiet, busy = run(False), run(True)
    assert len(busy.log) > 0
    assert quiet.fundamental_history == busy.fundamental_history


def test_exchange_ignores_the_fundamental():
    # Same orders with wildly different fundamentals: same trades, same book.
    def run(fundamental):
        sim = Simulator(fundamental=fundamental)
        sim.schedule(100, UpdateFundamental())
        sim.schedule(150, limit(1, Side.BUY, 100, 10))
        sim.schedule(200, UpdateFundamental())
        sim.schedule(250, market(2, Side.SELL, 4))
        sim.schedule(300, UpdateFundamental())
        sim.run()
        return sim

    low = run(FundamentalProcess(1.0, 0.0, seed=1))
    high = run(FundamentalProcess(1e6, 500.0, seed=2))
    plain = Simulator()
    plain.schedule(150, limit(1, Side.BUY, 100, 10))
    plain.schedule(250, market(2, Side.SELL, 4))
    plain.run()

    assert low.log == high.log == plain.log
    assert snapshot(low.exchange.book) == snapshot(high.exchange.book) == snapshot(plain.exchange.book)
    (t,) = high.trades
    assert (t.price, t.qty, t.timestamp) == (100, 4, 250)
    check_invariants(high.exchange.book)


def test_equal_time_update_follows_scheduling_order():
    class Spy(FundamentalProcess):
        def advance_to(self, timestamp):
            seen.append(len(sim.log))
            return super().advance_to(timestamp)

    seen = []
    sim = Simulator(fundamental=Spy(10_000.0, 2.0, seed=1))
    sim.schedule(100, limit(1, Side.BUY, 100, 1))
    sim.schedule(100, UpdateFundamental())
    sim.schedule(100, limit(2, Side.BUY, 100, 1))
    sim.schedule(200, UpdateFundamental())
    sim.schedule(200, limit(3, Side.BUY, 100, 1))
    sim.run()
    assert seen == [1, 2]


def test_repeat_update_at_same_time_records_once_and_draws_nothing():
    twice = Simulator(fundamental=proc())
    once = Simulator(fundamental=proc())
    for t in (0, 100, 100, 200):
        twice.schedule(t, UpdateFundamental())
    for t in (100, 200):
        once.schedule(t, UpdateFundamental())
    twice.run()
    once.run()
    assert twice.fundamental_history == once.fundamental_history
    assert [s.timestamp for s in twice.fundamental_history] == [0, 100, 200]


def test_true_value_stays_out_of_the_activity_log():
    sim = Simulator(fundamental=proc())
    sim.schedule(100, UpdateFundamental())
    sim.schedule(150, limit(1, Side.BUY, 100, 1))
    sim.run()
    assert len(sim.log) == 1
    assert not any(isinstance(r, FundamentalSnapshot) for r in sim.log)


def test_update_without_a_process_fails_when_scheduled():
    sim = Simulator()
    with pytest.raises(ValueError, match="no fundamental process"):
        sim.schedule(100, UpdateFundamental())
    sim.run()
    assert sim.now == 0


def test_plain_simulator_has_no_history():
    assert Simulator().fundamental_history == ()


def test_process_already_past_zero_rejected():
    p = proc()
    p.advance_to(SEC)
    with pytest.raises(ValueError, match="t=0"):
        Simulator(fundamental=p)


def test_process_moved_ahead_externally_stops_the_run_with_context():
    p = proc()
    sim = Simulator(fundamental=p)
    sim.schedule(100, UpdateFundamental())
    p.advance_to(SEC)
    with pytest.raises(ValueError, match="advance back") as exc:
        sim.run()
    assert any("t=100" in n for n in exc.value.__notes__)


def scripted(seed):
    sim = Simulator(fundamental=proc(seed=seed))
    for t in (SEC, 2 * SEC, 3 * SEC, 4 * SEC):
        sim.schedule(t, UpdateFundamental())
    sim.schedule(SEC, limit(1, Side.BUY, 9999, 10))
    sim.schedule(SEC + 5, limit(2, Side.SELL, 10_001, 6))
    sim.schedule(2 * SEC, market(3, Side.SELL, 4))
    sim.schedule(3 * SEC, CancelOrder(1))
    sim.run()
    return sim


def test_full_simulation_replays_exactly():
    a, b = scripted(42), scripted(42)
    assert a.log == b.log
    assert a.fundamental_history == b.fundamental_history
    assert snapshot(a.exchange.book) == snapshot(b.exchange.book)
    assert a.now == b.now == 4 * SEC


def test_different_seed_changes_only_the_fundamental():
    a, b = scripted(42), scripted(43)
    assert a.log == b.log
    assert a.fundamental_history != b.fundamental_history
