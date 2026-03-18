# bluefish

An imperfect-information game-solving engine built on Counterfactual Regret Minimization.

## Build 

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Requires CMake >= 3.20 and a C++20 compiler.

## Run

```bash
# Train vanilla CFR on Kuhn poker, 100k iterations
./build/solve kuhn cfr 100000
 
# Train MCCFR on Leduc poker, 5M iterations, export convergence data
./build/solve leduc mccfr 5000000 --csv data.csv --seed 42
 
# Run all algorithm x game combinations with reproducible CSV output
./build/experiment --output results.csv --iters 100000
 
# Throughput benchmark (3 trials, warmup, median reporting)
./build/bench
```

## Architecture
 
```
include/bluefish/
  game.h         GameState interface: decision / chance / terminal nodes
  kuhn.h         Kuhn poker   (3 cards, 1 round, 12 info sets)
  leduc.h        Leduc poker  (6 cards, 2 rounds, 288 info sets)
  solver.h       Solver base class, InfoNode, Stats, exploitability, JSON export
  cfr.h          Vanilla CFR
  cfr_plus.h     CFR+
  mccfr.h        External-sampling MCCFR
 
src/
  solver.cpp     Shared infrastructure: regret matching, best response, validation
  cfr.cpp        Full-tree traversal with reach-probability weighting
  cfr_plus.cpp   Regret flooring + linear averaging
  mccfr.cpp      Sampled traversal with alternating traverser
 
app/
  solve.cpp      CLI solver with convergence logging, CSV/JSON export, stats
  experiment.cpp Reproducible experiment harness (all combos, multiple seeds)
  bench.cpp      Throughput benchmark with warmup and median-of-N trials
```

Every algorithm inherits from `Solver` and shares the same `InfoNode` structure, exploitability computation, validation, and serialization. They differ only in their traversal logic.

## References

- Neller & Lanctot. *An Introduction to Counterfactual Regret Minimization.*
- Tammelin (2014). *Solving Large Imperfect Information Games Using CFR+.*
- Lanctot et al. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games.*

## Roadmap

| Phase | Scope                                               | Status |
| ----- | --------------------------------------------------- | ------ |
| 1     | Kuhn poker, vanilla CFR, exploitability, tests      | DONE   |
| 2     | Leduc poker, generic engine, strategy serialization | DONE   |
| 2     | CFR+, external-sampling MCCFR, algorithm comparison | DONE   |
| 4     | Benchmarks, profiling                               | DONE   |
