# bluefish

A C++20 engine for solving imperfect-information extensive-form games using Counterfactual Regret Minimization. Implements Kuhn and Leduc poker with a two-tier architecture: a clean virtual `GameState` interface for prototyping and a compiled flat-array representation for performance.

## Results at a Glance

| Metric | Value |
|--------|-------|
| Algorithms | CFR, CFR+, MCCFR, fast-CFR, fast-CFR+, fast-MCCFR, parallel-MCCFR |
| Games | Kuhn poker (12 info sets), Leduc poker (288 info sets) |
| fast-CFR throughput vs virtual | **8× faster** — 11,100 vs 1,400 iters/sec on Leduc |
| fast-MCCFR throughput vs virtual | **8× faster** — 2.5M vs ~300K iters/sec |
| Exploitability evaluation speedup | **13× faster** — 0.37s vs 4.8s per 1000 evals |
| End-to-end (50k iters + 10 eval checkpoints) | **7.9× faster** — 4.5s vs 34.5s |
| Card abstraction | 3 Leduc abstractions: exact (288 IS), pair-only (198 IS), jq-merge (132 IS) |
| Test suite | 82 tests across 5 suites, all passing |

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

Requires CMake >= 3.20 and a C++20 compiler. Dependencies (doctest) are fetched automatically by CMake.

## Run

```bash
# Train vanilla CFR on Kuhn poker, 100k iterations
./build/solve kuhn cfr 100000

# Train fast-MCCFR on Leduc, 5M iterations, export convergence data
./build/solve leduc fast-mccfr 5000000 --csv data.csv --seed 42

# Parallel MCCFR (uses all hardware threads by default)
./build/solve leduc par-mccfr 1000000

# All algorithm × game combos, reproducible CSV output
./build/experiment --output results.csv --iters 100000

# Throughput benchmark — iters/sec and nodes/sec, median of 3 trials
./build/bench

# Thread-scaling benchmark — 1 to N threads, speedup table
./build/bench_parallel
```

**Supported algorithms**: `cfr`, `cfr+`, `mccfr`, `fast-cfr`, `fast-cfr+`, `fast-mccfr`, `par-mccfr`

## Architecture

The engine has two tiers that share a common `Solver` base class:

```
Tier 1 — Virtual (prototyping)          Tier 2 — Flat (performance)
─────────────────────────────           ──────────────────────────────────────
GameState (virtual interface)           FlatGame: contiguous arrays
  kuhn.h   — 3 cards, 12 info sets        FlatNode[]  — 24 bytes/node
  leduc.h  — 6 cards, 288 info sets       children[]  — packed child indices
                                          chance_probs[] — precomputed
CfrSolver        (O(|H|)/iter)          FastCfrSolver    (zero heap alloc)
CfrPlusSolver    (regret floor + lin)   FastCfrPlusSolver (no virtual dispatch)
MccfrSolver      (external sampling)    FastMccfrSolver   (stack regret matching)
                                        ParallelMccfrSolver (per-thread tables)
```

`FlatGame::compile()` takes any `GameState` tree and packs it into cache-friendly arrays with dense integer info set IDs. Fast solvers never touch the string-keyed `InfoMap` during training — only on `serialize_json()`. This lazy sync is what enables the 8× throughput gain.

### Card Abstraction

`Abstraction` is a pure integer mapping from exact info set IDs to abstract ones. Three built-in Leduc abstractions:

| Name | Info Sets | Strategy |
|------|-----------|----------|
| `exact` | 288 | Identity — no merging |
| `pair_only` | 198 | Postflop: only distinguish paired vs unpaired |
| `jq_merge` | 132 | Treat J and Q as equivalent everywhere |

`apply_abstraction()` creates a new `FlatGame` with remapped IDs. `lifted_flat_exploitability()` evaluates the abstract strategy on the *exact* game tree — the correct quality measure.

### Parallel MCCFR

`ParallelMccfrSolver` uses lock-free shared double arrays (benign data races on `+=` — standard practice following Bowling et al.). Per-thread `Stats` counters eliminate all contention on the hot path. Thread scaling is near-linear because each traversal touches O(depth) nodes with rare writes.

## File Layout

```
include/bluefish/   — all headers
  game.h            GameState virtual interface
  kuhn.h            Kuhn poker implementation
  leduc.h           Leduc poker implementation
  solver.h          Solver base: InfoNode, Stats, exploitability, validation, JSON
  cfr.h / cfr_plus.h / mccfr.h     Virtual-tier solvers
  flat_game.h       FlatGame, FlatNode, RegretTable, flat_exploitability
  fast_cfr.h        Fast-tier solvers (CFR, CFR+, MCCFR)
  abstraction.h     Abstraction, make/apply/lift, Leduc factories
  parallel_mccfr.h  Lock-free parallel MCCFR

src/                — implementations
app/                — executables (solve, bench, experiment, bench_parallel)
tests/              — doctest suites (82 tests)
```

## Known Nash Values

- Kuhn poker: −1/18 ≈ −0.0556 (analytical)
- Leduc poker: ≈ −0.0856 (converged empirically)

All solvers are validated against these values in the test suite.

## References

- Neller & Lanctot. *An Introduction to Counterfactual Regret Minimization.*
- Tammelin (2014). *Solving Large Imperfect Information Games Using CFR+.*
- Lanctot et al. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games.*
- Bowling et al. (2015). *Heads-up Limit Hold'em Poker is Solved.*
