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
that schedules submissions and cancels and records what happened. No traders
or strategies yet.

## Setup

Requires Python 3.12+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Conventions

- Prices are integer ticks.
- Time is integer nanoseconds of simulation time.
- Randomness will always be seeded explicitly.
