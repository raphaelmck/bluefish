# bluefish

An imperfect-information game-solving engine built on Counterfactual Regret Minimization.

## Roadmap

| Phase | Scope                                               | Status |
| ----- | --------------------------------------------------- | ------ |
| 1     | Kuhn poker, vanilla CFR, exploitability, tests      | DONE   |
| 2     | Leduc poker, generic engine, strategy serialization | DONE   |
| 2     | CFR+, external-sampling MCCFR, algorithm comparison | DONE   |
| 4     | Benchmarks, profiling                               | DONE   |

## Build 

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Run

```bash
./build/solve kuhn						    # vanilla CFR, 100k iters
./build/solve leduc cfr+ 500000             # CFR+ on Leduc
./build/solve leduc mccfr 5000000 --seed 7  # MCCFR with custom seed
./build/solve kuhn cfr --json strategy.json # export strategy as JSON
```

## Test

```bash
cd build && ctest --output-on-failure
```

## References

- Neller & Lanctot. *An Introduction to Counterfactual Regret Minimization.*
- Tammelin (2014). *Solving Large Imperfect Information Games Using CFR+.*
- Lanctot et al. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games.*
