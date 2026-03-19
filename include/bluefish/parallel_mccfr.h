// bluefish/parallel_mccfr.h - Lock-free parallel external-sampling MCCFR.
//
// Each thread runs independent MCCFR traversals with its own RNG and
// Stats counters. Regret and strategy-sum updates are applied directly
// to shared double[] arrays without locks. This is a benign data race:
//   - Each update is a += of a small value
//   - Torn reads/writes on doubles may occasionally lose an update
//   - This does not affect convergence — only adds minor noise
//   - Standard practice in parallel CFR (Bowling et al.)
//
// Thread scaling is near-linear because:
//   - Each traversal touches O(depth) nodes (~20 on Leduc)
//   - The shared regret table is read-heavy (regret matching) with
//   - rare writes (one update per info set per traversal)
//   - No synchronization points during iteration batches

#pragma once

#include "flat_game.h"
#include "solver.h"

#include <cstdint>

namespace bluefish {

class ParallelMccfrSolver final : public Solver {
public:
    ParallelMccfrSolver(FlatGame game, RootFn root_fn,
                        int num_threads = 0,  // 0 = hardware_concurrency
                        uint64_t seed = 42);

    double train(int iterations, RootFn root_fn) override;
    double exploitability(RootFn root_fn) const override;
    std::string name() const override;
    std::string serialize_json() const override;
    const InfoMap& info_map() const override;
    std::size_t num_info_sets() const override {
        return static_cast<std::size_t>(game_.num_info_sets);
    }

    int num_threads() const { return num_threads_; }

private:
    FlatGame game_;
    RootFn root_fn_;
    RegretTable table_;
    int num_threads_;
    uint64_t base_seed_;
    mutable bool synced_ = false;

    void ensure_synced() const;
};

} // namespace bluefish
