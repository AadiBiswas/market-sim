# market-sim

An event-driven limit order book simulator, built to study market making.

The plan: a single-asset exchange with price-time priority matching, a latent
fundamental value that nobody gets to see directly, a mix of noise traders and
informed traders who see a noisy version of it, and one market maker trying to
earn the spread without getting run over. The research question is how the
market maker should change its quotes as order flow gets more informed, i.e.
the tradeoff between spread capture, inventory risk, adverse selection, and
getting filled at all.

Current state: a price-time priority matching engine (limit and market
orders, partial fills, cancels) and a deterministic discrete-event simulator
that schedules submissions and cancels and records what happened, plus a
latent fundamental value (arithmetic Brownian motion) that evolves on explicitly
scheduled updates, and seeded noise traders that wake on a schedule and see only
the top of the book. No market maker or informed traders yet.

## Setup

Requires Python 3.12+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python examples/noise_market.py   # small deterministic multi-agent run
```

## Conventions

- Prices are integer ticks.
- Time is integer nanoseconds of simulation time.
- Randomness will always be seeded explicitly.
