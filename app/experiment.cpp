// experiment.cpp - Reproducible convergence experiment

// Runs all algorithm x game combinations at matched iteration counts
// Usage:
//   ./experiment                        # default: all combos, stdout
//   ./experiment --output results.csv   # write to file
//   ./experiment --iters 500000         # custom max iterations

// Output columns:
//   game, algorithm, seed, iteration, exploitability, game_value,
//   wall_sec, nodes_touched, terminal_visits, info_set_visits,
//   chance_visits, info_sets, nodes_per_iter

#include "bluefish/cfr.h"
#include "bluefish/cfr_plus.h"
#include "bluefish/mccfr.h"
#include "bluefish/kuhn.h"
#include "bluefish/leduc.h"

#include <fstream>
#include <iostream>

namespace {

using Clock = std::chrono::steady_clock;

struct GameDef {
    std::string name;
    bluefish::Solver::RootFn root_fn;
};

struct AlgoDef {
    std::string name;
    // Factory: seed -> solver; seed is ignored for deterministic algorithms
    std::function<std::unique_ptr<bluefish::Solver>(uint64_t seed)> make;
    bool stochastic;
};

std::vector<int> log_checkpoints(int total) {
    std::vector<int> cp;
    for (int p = 1; p <= total; p *= 10) {
        cp.push_back(p);
        if (p * 2 <= total) cp.push_back(p * 2);
        if (p * 5 <= total) cp.push_back(p * 5);
    }
    cp.push_back(total);
    // Deduplicate
    std::sort(cp.begin(), cp.end());
    cp.erase(std::unique(cp.begin(), cp.end()), cp.end());
    return cp;
}

void run_combo(const GameDef& game, const AlgoDef& algo,
               uint64_t seed, int max_iters,
               std::ostream& out) {
    auto solver = algo.make(seed);
    auto checkpoints = log_checkpoints(max_iters);

    int done = 0;
    auto t0 = Clock::now();

    for (int cp : checkpoints) {
        if (cp <= done) continue;
        int batch = cp - done;

        // Time only the training, not exploitability computation
        double gv = solver->train(batch, game.root_fn);
        double total_wall = std::chrono::duration<double>(
            Clock::now() - t0).count();
        done = cp;

        double exploit = solver->exploitability(game.root_fn);

        auto& st = solver->stats();
        double npi = static_cast<double>(st.nodes_touched)
                     / static_cast<double>(done);

        out << game.name << ","
            << algo.name << ","
            << seed << ","
            << done << ","
            << std::scientific << std::setprecision(10)
            << exploit << ","
            << gv << ","
            << std::fixed << std::setprecision(6)
            << total_wall << ","
            << st.nodes_touched << ","
            << st.terminal_visits << ","
            << st.info_set_visits << ","
            << st.chance_visits << ","
            << solver->num_info_sets() << ","
            << std::setprecision(1) << npi << "\n";
    }

    // Validate at end
    std::string err = solver->validate();
    if (!err.empty()) {
        std::cerr << "VALIDATION FAILED: " << game.name << "/"
                  << algo.name << " seed=" << seed << ": " << err << "\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    int max_iters = 200'000;
    std::string output_path;
    std::vector<uint64_t> mccfr_seeds = {42, 137, 271};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc)
            output_path = argv[++i];
        else if (arg == "--iters" && i + 1 < argc)
            max_iters = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: experiment [--output file.csv] [--iters N]\n";
            return 0;
        }
    }

    // Games
    std::vector<GameDef> games = {
        {"kuhn", bluefish::kuhn::root_states},
        {"leduc", bluefish::leduc::root_states},
    };

    // Algorithms
    std::vector<AlgoDef> algos = {
        {"cfr",
         [](uint64_t) { return std::make_unique<bluefish::CfrSolver>(); },
         false},
        {"cfr+",
         [](uint64_t) { return std::make_unique<bluefish::CfrPlusSolver>(); },
         false},
        {"mccfr",
         [](uint64_t s) { return std::make_unique<bluefish::MccfrSolver>(s); },
         true},
    };

    // Output stream
    std::ofstream file;
    std::ostream* out = &std::cout;
    if (!output_path.empty()) {
        file.open(output_path);
        out = &file;
    }

    // CSV header.
    *out << "game,algorithm,seed,iteration,exploitability,game_value,"
         << "wall_sec,nodes_touched,terminal_visits,info_set_visits,"
         << "chance_visits,info_sets,nodes_per_iter\n";

    int total_combos = 0;
    for (auto& game : games) {
        for (auto& algo : algos) {
            auto seeds = algo.stochastic
                ? mccfr_seeds
                : std::vector<uint64_t>{0};
            for (auto seed : seeds) {
                ++total_combos;
                std::cerr << "[" << total_combos << "] "
                          << game.name << " / " << algo.name;
                if (algo.stochastic)
                    std::cerr << " (seed=" << seed << ")";
                std::cerr << " ..." << std::flush;

                auto t0 = Clock::now();
                run_combo(game, algo, seed, max_iters, *out);
                double sec = std::chrono::duration<double>(
                    Clock::now() - t0).count();

                std::cerr << " done (" << std::fixed << std::setprecision(1)
                          << sec << "s)\n";
            }
        }
    }

    if (!output_path.empty())
        std::cerr << "Results written to " << output_path << "\n";

    return 0;
}
