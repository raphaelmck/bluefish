// bench.cpp - Focused throughput benchmark

// Measures iterations/sec and nodes/sec for each algorithm on each game
// Uses warmup iterations (discarded), then multiple timed trials

// Usage:
//   ./bench                  # default settings
//   ./bench --trials 5       # more trials for stability

#include "bluefish/cfr.h"
#include "bluefish/cfr_plus.h"
#include "bluefish/mccfr.h"
#include "bluefish/kuhn.h"
#include "bluefish/leduc.h"

#include <iomanip>
#include <iostream>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchResult {
    std::string game;
    std::string algo;
    double median_ips; // iterations per second
    double median_nps; // nodes per second
    double median_npi; // nodes per iteration
    double exploitability;
};

// Choose iteration count to keep each trial around 1-3 seconds
int calibrate_iters(const std::string& game, const std::string& algo) {
    if (game == "kuhn") {
        if (algo == "mccfr") return 2'000'000;
        return 200'000;
    }
    // Leduc
    if (algo == "mccfr") return 2'000'000;
    return 20'000;
}

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    auto n = v.size();
    if (n % 2 == 0)
        return (v[n / 2 - 1] + v[n / 2]) / 2.0;
    return v[n / 2];
}

} // namespace

int main(int argc, char* argv[]) {
    int num_trials = 3;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trials" && i + 1 < argc)
            num_trials = std::atoi(argv[++i]);
    }

    struct GameDef {
        std::string name;
        bluefish::Solver::RootFn root_fn;
    };
    struct AlgoDef {
        std::string name;
        std::function<std::unique_ptr<bluefish::Solver>()> make;
    };

    std::vector<GameDef> games = {
        {"kuhn", bluefish::kuhn::root_states},
        {"leduc", bluefish::leduc::root_states},
    };
    std::vector<AlgoDef> algos = {
        {"cfr",   []{ return std::make_unique<bluefish::CfrSolver>(); }},
        {"cfr+",  []{ return std::make_unique<bluefish::CfrPlusSolver>(); }},
        {"mccfr", []{ return std::make_unique<bluefish::MccfrSolver>(42); }},
    };

    std::vector<BenchResult> results;

    std::cout << "Bluefish throughput benchmark (" << num_trials
              << " trials each)\n\n";

    for (auto& game : games) {
        for (auto& algo : algos) {
            int iters = calibrate_iters(game.name, algo.name);

            std::cerr << game.name << "/" << algo.name
                      << " (" << iters << " iters × " << num_trials
                      << " trials) ..." << std::flush;

            std::vector<double> ips_samples, nps_samples;
            double npi = 0.0;
            double final_exploit = 0.0;

            for (int trial = 0; trial < num_trials; ++trial) {
                auto solver = algo.make();

                // Warmup: 10% of iterations, discarded
                int warmup = std::max(1, iters / 10);
                solver->train(warmup, game.root_fn);

                // Reset stats so they only measure the timed portion
                solver->stats(); // just access, stats are cumulativ

                auto prev_nodes = solver->stats().nodes_touched;
                auto t0 = Clock::now();
                solver->train(iters, game.root_fn);
                double sec = std::chrono::duration<double>(
                    Clock::now() - t0).count();

                auto batch_nodes = solver->stats().nodes_touched - prev_nodes;
                ips_samples.push_back(static_cast<double>(iters) / sec);
                nps_samples.push_back(static_cast<double>(batch_nodes) / sec);
                npi = static_cast<double>(batch_nodes)
                      / static_cast<double>(iters);

                if (trial == num_trials - 1)
                    final_exploit = solver->exploitability(game.root_fn);
            }

            results.push_back({
                game.name, algo.name,
                median(ips_samples), median(nps_samples),
                npi, final_exploit
            });

            std::cerr << " done\n";
        }
    }

    // Print results table
    std::cout << std::setw(8) << "game"
              << std::setw(8) << "algo"
              << std::setw(14) << "iters/sec"
              << std::setw(14) << "nodes/sec"
              << std::setw(14) << "nodes/iter"
              << std::setw(16) << "exploitability"
              << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (auto& r : results) {
        std::cout << std::setw(8) << r.game
                  << std::setw(8) << r.algo
                  << std::fixed << std::setprecision(0)
                  << std::setw(14) << r.median_ips
                  << std::setw(14) << r.median_nps
                  << std::setw(14) << r.median_npi
                  << std::scientific << std::setprecision(4)
                  << std::setw(16) << r.exploitability
                  << "\n";
    }
    std::cout << "\n";

    // Highlight the key ratios
    for (auto& game : games) {
        double cfr_npi = 0, mccfr_npi = 0;
        double cfr_ips = 0, mccfr_ips = 0;
        for (auto& r : results) {
            if (r.game != game.name) continue;
            if (r.algo == "cfr") { cfr_npi = r.median_npi; cfr_ips = r.median_ips; }
            if (r.algo == "mccfr") { mccfr_npi = r.median_npi; mccfr_ips = r.median_ips; }
        }
        if (cfr_npi > 0 && mccfr_npi > 0) {
            std::cout << game.name << ": CFR touches "
                      << std::fixed << std::setprecision(0)
                      << cfr_npi / mccfr_npi
                      << "× more nodes/iter than MCCFR; MCCFR runs "
                      << mccfr_ips / cfr_ips
                      << "× more iters/sec\n";
        }
    }

    return 0;
}
