// bench_parallel.cpp - Thread-scaling benchmark for parallel MCCFR.
//
// Measures iterations/sec and exploitability at 1, 2, 4, ... threads
// on Leduc poker using identical iteration counts.
//
// Usage:
//   ./bench_parallel                  # auto-detect thread count
//   ./bench_parallel --max-threads 8  # cap at 8 threads
//   ./bench_parallel --iters 2000000  # custom iteration count

#include "bluefish/parallel_mccfr.h"
#include "bluefish/fast_cfr.h"
#include "bluefish/leduc.h"

#include <iomanip>
#include <iostream>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
}

int main(int argc, char* argv[]) {
    int max_threads = static_cast<int>(std::thread::hardware_concurrency());
    int iters = 2'000'000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-threads" && i + 1 < argc)
            max_threads = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc)
            iters = std::atoi(argv[++i]);
    }
    if (max_threads < 1) max_threads = 1;

    auto root_fn = bluefish::leduc::root_states;
    auto exact = bluefish::FlatGame::compile(root_fn);

    std::cout << "Parallel MCCFR thread-scaling benchmark\n";
    std::cout << "Game: Leduc poker (" << exact.num_info_sets
              << " info sets, " << exact.num_nodes() << " nodes)\n";
    std::cout << "Iterations: " << iters
              << ", max threads: " << max_threads << "\n\n";

    // Single-threaded fast-mccfr baseline
    double baseline_ips = 0.0;
    double baseline_exploit = 0.0;
    {
        auto game = bluefish::FlatGame::compile(root_fn);
        bluefish::FastMccfrSolver solver(std::move(game), root_fn, 42);

        auto t0 = Clock::now();
        solver.train(iters, root_fn);
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();

        baseline_ips = static_cast<double>(iters) / sec;
        baseline_exploit = solver.exploitability(root_fn);

        std::cout << "Baseline (fast-mccfr, 1 thread):\n"
                  << "  " << std::fixed << std::setprecision(0)
                  << baseline_ips << " iters/sec, exploit="
                  << std::scientific << std::setprecision(4)
                  << baseline_exploit << "\n\n";
    }

    // Build thread counts: 1, 2, 4, 8, ... up to max_threads
    std::vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2)
        thread_counts.push_back(t);
    if (thread_counts.back() != max_threads)
        thread_counts.push_back(max_threads);

    std::cout << std::setw(8) << "threads"
              << std::setw(14) << "iters/sec"
              << std::setw(14) << "nodes/sec"
              << std::setw(10) << "speedup"
              << std::setw(16) << "exploitability"
              << std::setw(12) << "wall (s)"
              << "\n";
    std::cout << std::string(74, '-') << "\n";

    for (int nt : thread_counts) {
        auto game = bluefish::FlatGame::compile(root_fn);
        bluefish::ParallelMccfrSolver solver(
            std::move(game), root_fn, nt, 42);

        auto t0 = Clock::now();
        solver.train(iters, root_fn);
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();

        double ips = static_cast<double>(iters) / sec;
        double nps = static_cast<double>(solver.stats().nodes_touched) / sec;
        double exploit = solver.exploitability(root_fn);
        double speedup = ips / baseline_ips;

        std::cout << std::setw(8) << nt
                  << std::fixed << std::setprecision(0)
                  << std::setw(14) << ips
                  << std::setw(14) << nps
                  << std::setprecision(2)
                  << std::setw(10) << speedup << "×"
                  << std::scientific << std::setprecision(4)
                  << std::setw(16) << exploit
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << sec
                  << "\n";
    }

    std::cout << "\nEfficiency = speedup / threads. Perfect scaling = 1.0.\n";

    return 0;
}
