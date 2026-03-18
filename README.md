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
  game.h         GameState interface (virtual, clean, for prototyping)
  kuhn.h         Kuhn poker   (3 cards, 12 info sets)
  leduc.h        Leduc poker  (6 cards, 288 info sets)
  solver.h       Solver base: InfoNode, Stats, exploitability, validation
  cfr.h          Vanilla CFR (virtual GameState traversal)
  cfr_plus.h     CFR+ (virtual)
  mccfr.h        External-sampling MCCFR (virtual)
  flat_game.h    FlatGame: compiled tree in contiguous arrays
  fast_cfr.h     Fast CFR/CFR+/MCCFR on flat trees
 
src/
  solver.cpp     Shared: regret matching, best response, validation, JSON
  cfr.cpp        Vanilla CFR traversal
  cfr_plus.cpp   CFR+ traversal
  mccfr.cpp      MCCFR traversal
  flat_game.cpp  Game tree compilation
  fast_cfr.cpp   Zero-allocation traversal (CFR, CFR+, MCCFR)
```

Every algorithm inherits from `Solver` and shares the same `InfoNode` structure, exploitability computation, validation, and serialization. They differ only in their traversal logic.

## Roadmap

| Phase | Scope                                               | Status |
| ----- | --------------------------------------------------- | ------ |
| 1     | Kuhn poker, vanilla CFR, exploitability, tests      | DONE   |
| 2     | Leduc poker, generic engine, strategy serialization | DONE   |
| 2     | CFR+, external-sampling MCCFR, algorithm comparison | DONE   |
| 4     | Benchmarks, profiling, experiment infra             | DONE   |
| 5     | Flat compiled trees, performance increase           |        |
| 6     | Custom game                                         |        |

## References

- Neller & Lanctot. *An Introduction to Counterfactual Regret Minimization.*
- Tammelin (2014). *Solving Large Imperfect Information Games Using CFR+.*
- Lanctot et al. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games.*


