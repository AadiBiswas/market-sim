"""Exchange vs. a deliberately dumb reference book.

The reference model shares no code or data structures with the real one: no
deques, no sorted price lists, just a dict of resting orders and a min() over
everything eligible on each fill. If both agree on every trade and every
resting queue, FIFO and price priority are right for reasons other than "the
implementation agrees with itself".
"""

import random
from collections import defaultdict

import pytest

from helpers import check_invariants, lim, mkt, reconcile, snapshot
from market_sim.exchange import Exchange, OrderStatus, OrderType, Side


class RefBook:
    def __init__(self):
        self.resting = {}  # oid -> [side, price, remaining, arrival_seq]
        self.seen = set()
        self._seq = 0

    def submit(self, oid, side, price, qty):
        """price=None means market. Returns (trades, leftover) or None if rejected."""
        if oid in self.seen:
            return None
        self.seen.add(oid)
        trades, rem = [], qty
        while rem:
            eligible = [
                (k, r) for k, r in self.resting.items()
                if r[0] is not side and (
                    price is None
                    or (side is Side.BUY and r[1] <= price)
                    or (side is Side.SELL and r[1] >= price)
                )
            ]
            if not eligible:
                break
            # Cheapest ask / richest bid, then earliest arrival.
            if side is Side.BUY:
                k, r = min(eligible, key=lambda kr: (kr[1][1], kr[1][3]))
            else:
                k, r = min(eligible, key=lambda kr: (-kr[1][1], kr[1][3]))
            q = min(rem, r[2])
            rem -= q
            r[2] -= q
            buyer, seller = (oid, k) if side is Side.BUY else (k, oid)
            trades.append((buyer, seller, r[1], q))
            if r[2] == 0:
                del self.resting[k]
        if rem and price is not None:
            self._seq += 1
            self.resting[oid] = [side, price, rem, self._seq]
        return trades, rem

    def cancel(self, oid):
        r = self.resting.pop(oid, None)
        return None if r is None else r[2]

    def state(self):
        by_level = defaultdict(list)
        for k, (side, price, rem, seq) in self.resting.items():
            by_level[side, price].append((seq, k, rem))
        return {
            side: [(p, [(k, rem) for _, k, rem in sorted(by_level[side, p])])
                   for p in sorted(p for s, p in by_level if s is side)]
            for side in Side
        }


def book_state(book):
    return {side: [(p, [(oid, rem) for oid, rem, _ in lvl]) for p, lvl in levels]
            for side, levels in snapshot(book).items()}


def run_random_flow(seed, n_actions=500):
    rng = random.Random(seed)
    # Narrow bands keep hammering the same few levels into and out of existence.
    lo, hi = rng.choice([(99, 101), (97, 103), (90, 110)])
    ex, ref = Exchange(), RefBook()
    orders, trades = [], []
    next_id = 1

    for step in range(n_actions):
        roll = rng.random()

        if roll < 0.15 and orders:
            # Live or long dead -- both should be handled.
            oid = rng.choice(orders).order_id
            got = ex.cancel(oid)
            want = ref.cancel(oid)
            assert (got and got.remaining) == want, f"seed={seed} step={step} cancel {oid}"
        elif roll < 0.18:
            assert ex.cancel(10_000 + step) is None
        elif roll < 0.23 and orders:
            # ID reuse, with a marketable price so a missed check would trade.
            before = snapshot(ex.book)
            dup = lim(rng.choice(orders).order_id, rng.choice(list(Side)), rng.choice([lo, hi]), 50, ts=step)
            with pytest.raises(ValueError):
                ex.submit(dup)
            assert (dup.remaining, dup.status) == (50, OrderStatus.OPEN)
            assert snapshot(ex.book) == before
        else:
            side = rng.choice(list(Side))
            qty = rng.randint(30, 120) if rng.random() < 0.1 else rng.randint(1, 15)
            if roll < 0.35:
                o = mkt(next_id, side, qty, ts=step)
            else:
                o = lim(next_id, side, rng.randint(lo, hi), qty, ts=step, agent=rng.randrange(4))
            next_id += 1
            orders.append(o)
            got = ex.submit(o)
            trades.extend(got)
            want, leftover = ref.submit(o.order_id, o.side, o.price, o.qty)
            assert [(t.buy_order_id, t.sell_order_id, t.price, t.qty) for t in got] == want, \
                f"seed={seed} step={step} order {o.order_id}"
            assert o.remaining == leftover

        assert book_state(ex.book) == ref.state(), f"seed={seed} step={step}"
        check_invariants(ex.book)

    reconcile(orders, trades)
    return orders, trades


@pytest.mark.parametrize("seed", range(30))
def test_random_flow_matches_reference(seed):
    run_random_flow(seed)


def test_random_flow_actually_exercises_things():
    # Guard against the generator quietly degenerating into "nothing ever trades".
    statuses, n_trades, sweeps = defaultdict(int), 0, 0
    for seed in range(30):
        orders, trades = run_random_flow(seed)
        n_trades += len(trades)
        for o in orders:
            statuses[o.status, o.order_type] += 1
        per_aggressor = defaultdict(set)
        for t in trades:
            agg = t.buy_order_id if t.aggressor is Side.BUY else t.sell_order_id
            per_aggressor[agg].add(t.price)
        sweeps += sum(len(prices) > 1 for prices in per_aggressor.values())

    assert n_trades > 1000
    assert sweeps > 50
    for key in [(OrderStatus.OPEN, OrderType.LIMIT), (OrderStatus.PARTIALLY_FILLED, OrderType.LIMIT),
                (OrderStatus.FILLED, OrderType.LIMIT), (OrderStatus.CANCELLED, OrderType.LIMIT),
                (OrderStatus.FILLED, OrderType.MARKET), (OrderStatus.CANCELLED, OrderType.MARKET)]:
        assert statuses[key] > 0, key
