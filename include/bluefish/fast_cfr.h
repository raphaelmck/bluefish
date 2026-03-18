// bluefish/fast_cfr.h - High-performance CFR-family solvers on flat trees

#pragma once

#include "flat_game.h"
#include "solver.h"

#include <cstdint>
#include <random>

namespace bluefish {

// ─ Fast vanilla CFR ─

class FastCfrSolver final : public Solver {
public:
    FastCfrSolver(FlatGame game, RootFn root_fn);
    double train(int iterations, RootFn root_fn) override;
    double exploitability(RootFn root_fn) const override;
    std::string name() const override { return "fast-cfr"; }
    std::string serialize_json() const override;
    const InfoMap& info_map() const override;
    std::size_t num_info_sets() const override {
        return static_cast<std::size_t>(game_.num_info_sets);
    }

private:
    FlatGame game_;
    RootFn root_fn_;
    RegretTable table_;
    mutable bool synced_ = false;

    double cfr(int node_idx, double pi0, double pi1);
    void ensure_synced() const;
};

// ─ Fast CFR+ ─

class FastCfrPlusSolver final : public Solver {
public:
    FastCfrPlusSolver(FlatGame game, RootFn root_fn);
    double train(int iterations, RootFn root_fn) override;
    double exploitability(RootFn root_fn) const override;
    std::string name() const override { return "fast-cfr+"; }
    std::string validate() const override;
    std::string serialize_json() const override;
    const InfoMap& info_map() const override;
    std::size_t num_info_sets() const override {
        return static_cast<std::size_t>(game_.num_info_sets);
    }

private:
    FlatGame game_;
    RootFn root_fn_;
    RegretTable table_;
    mutable bool synced_ = false;

    double cfr_plus(int node_idx, double pi0, double pi1);
    void ensure_synced() const;
};

// - Fast external-sampling MCCFR -

class FastMccfrSolver final : public Solver {
public:
    FastMccfrSolver(FlatGame game, RootFn root_fn, uint64_t seed = 42);
    double train(int iterations, RootFn root_fn) override;
    double exploitability(RootFn root_fn) const override;
    std::string name() const override { return "fast-mccfr"; }
    std::string serialize_json() const override;
    const InfoMap& info_map() const override;
    std::size_t num_info_sets() const override {
        return static_cast<std::size_t>(game_.num_info_sets);
    }

private:
    FlatGame game_;
    RootFn root_fn_;
    RegretTable table_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_{0.0, 1.0};
    mutable bool synced_ = false;

    double traverse(int node_idx, int traverser);
    int sample(const double* probs, int n);
    void ensure_synced() const;
};

} // namespace bluefish
