"""Small deterministic noise-trader market. Run: python examples/noise_market.py

Two passive and two aggressive noise traders for ten simulated seconds, with
the fundamental updated once a second in the background. Prints what happened
and checks a replay with fresh agents matches. Nothing here is meant to be an
economically sensible market yet.
"""

from collections import Counter

from market_sim.agents import NoiseTrader
from market_sim.exchange import Side
from market_sim.simulation import Cancelled, FundamentalProcess, Simulator, Submitted, UpdateFundamental, WakeAgent

SEC = 1_000_000_000
REF = 10_000


def build(seed_offset=0):
    sim = Simulator(fundamental=FundamentalProcess(float(REF), 1.5, seed=7))
    # Makers and takers get separate agent IDs: the exchange still allows
    # self-trades, and this setup simply never gives it the chance.
    for agent in [
        NoiseTrader(1, 101 + seed_offset, "passive", REF, qty_range=(1, 8), max_offset=3),
        NoiseTrader(2, 102 + seed_offset, "passive", REF, qty_range=(1, 8), max_offset=3),
        NoiseTrader(3, 201 + seed_offset, "aggressive", REF, act_prob=0.7, qty_range=(1, 15)),
        NoiseTrader(4, 202 + seed_offset, "aggressive", REF, act_prob=0.7, qty_range=(1, 15)),
    ]:
        sim.register_agent(agent)
    for k in range(1, 11):
        sim.schedule(k * SEC, UpdateFundamental())
    for k in range(80):
        t = k * SEC // 8
        sim.schedule(t, WakeAgent(1 + k % 2))
        sim.schedule(t + 1_000, WakeAgent(3 + k % 2))
    return sim


def main():
    sim = build()
    sim.run()

    subs = [r for r in sim.log if isinstance(r, Submitted)]
    trades = sim.trades
    kinds = Counter((s.agent_id, s.order_type.name) for s in subs)
    print(f"clock: {sim.now / SEC:.3f} s")
    print(f"orders: {len(subs)}  " + "  ".join(f"agent{a}/{k}={n}" for (a, k), n in sorted(kinds.items())))
    print(f"trades: {len(trades)}  volume: {sum(t.qty for t in trades)}")
    short = [r for r in sim.log if isinstance(r, Cancelled)]
    print(f"market orders cut short: {len(short)} ({sum(r.remaining for r in short)} shares unfilled)")

    ids = [s.order_id for s in subs]
    by_id = {s.order_id: s for s in subs}
    assert len(ids) == len(set(ids)), "duplicate order IDs"
    for t in trades:
        resting = by_id[t.sell_order_id if t.aggressor is Side.BUY else t.buy_order_id]
        assert t.price == resting.price, "trade not at resting price"
        assert t.buy_agent_id != t.sell_agent_id
    print("checks: unique IDs, trades at resting prices, no self-trades")

    book = sim.exchange.book
    print(f"final book: bid {book.best_bid} x {book.quantity_at_price(Side.BUY, book.best_bid) if book.best_bid else 0}"
          f" | ask {book.best_ask} x {book.quantity_at_price(Side.SELL, book.best_ask) if book.best_ask else 0}"
          f" | {len(book._prices[Side.BUY])} bid levels, {len(book._prices[Side.SELL])} ask levels")
    print("fundamental (research-only):", " ".join(f"{s.value:.2f}" for s in sim.fundamental_history))

    again = build()
    again.run()
    print("replay with fresh agents identical:", again.log == sim.log and again.fundamental_history == sim.fundamental_history)
    other = build(seed_offset=1000)
    other.run()
    print("different agent seeds -> different market, same fundamental:",
          other.log != sim.log and other.fundamental_history == sim.fundamental_history)


if __name__ == "__main__":
    main()
