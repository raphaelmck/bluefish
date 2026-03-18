// bluefish/fast_cfr.h - High-performance CFR-familiy solvers on flat trees

#pragma once

#include "flat_game.h"
#include "solver.h"

#include <cstdint>
#include <random>

namespace bluefish {

// - Fast vanilla CFR -

class FastCfrSolver final : public Solver {
public:
    FastCfrSolver(FlatGame game, RootFn root_fn);
    double train(int iterations, RootFn root_fn) override;
    std::string name() const override { return "fast-cfr"; }

private:
    FlatGame game_;
    RootFn root_fn_;
    RegretTable table_;

    double cfr(int node_idx, double pi0, double pi1);
    void sync();
};

// - Fast CFR+ -

class FastCfrPlusSolver final : public Solver {
public:
    FastCfrPlusSolver(FlatGame game, RootFn root_fn);
    double train(int iterations, RootFn root_fn) override;
    std::string name() const override { return "fast-cfr+"; }
    std::string validate() const override;

private:
    FlatGame game_;
    RootFn root_fn_;
    RegretTable table_;

    double cfr_plus(int node_idx, double pi0, double pi1);
    void sync();
};

// - Fast external-sampling MCCFR - 

class FastMccfrSolver final : public Solver {
public:
    FastMccfrSolver(FlatGame game, RootFn root_fn, uint64_t seed = 42);
    double train(int iterations, RootFn root_fn) override;
    std::string name() const override { return "fast-mccfr"; }

private:
    FlatGame game_;
    RootFn root_fn_;
    RegretTable table_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_{0.0, 1.0};

    double traverse(int node_idx, int traverser);
    int sample(const double* probs, int n);
    void sync();
};

} // namespace bluefish
