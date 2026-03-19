// Claude Opus 4.6
// test_parallel.cpp — Parallel MCCFR tests.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "bluefish/parallel_mccfr.h"
#include "bluefish/fast_cfr.h"
#include "bluefish/kuhn.h"
#include "bluefish/leduc.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

using namespace bluefish;

// ═══════════════════════════════════════════════════════════════════
// Basic correctness
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("par-mccfr: kuhn converges with 2 threads") {
    auto root_fn = kuhn::root_states;
    auto game = FlatGame::compile(root_fn);
    ParallelMccfrSolver solver(std::move(game), root_fn, 2, 42);
    solver.train(200'000, root_fn);

    double e = solver.exploitability(root_fn);
    CHECK(e < 0.05);
    CHECK(solver.num_info_sets() == 12);
}

TEST_CASE("par-mccfr: leduc converges with 2 threads") {
    auto root_fn = leduc::root_states;
    auto game = FlatGame::compile(root_fn);
    ParallelMccfrSolver solver(std::move(game), root_fn, 2, 42);
    solver.train(500'000, root_fn);

    double e = solver.exploitability(root_fn);
    CHECK(e < 0.5);
    CHECK(solver.num_info_sets() == 288);
}

TEST_CASE("par-mccfr: leduc converges with 4 threads") {
    auto root_fn = leduc::root_states;
    auto game = FlatGame::compile(root_fn);
    ParallelMccfrSolver solver(std::move(game), root_fn, 4, 42);
    solver.train(500'000, root_fn);

    double e = solver.exploitability(root_fn);
    CHECK(e < 0.5);
}

TEST_CASE("par-mccfr: 1 thread matches fast-mccfr quality") {
    auto root_fn = leduc::root_states;
    constexpr int iters = 500'000;

    auto game1 = FlatGame::compile(root_fn);
    FastMccfrSolver single(std::move(game1), root_fn, 42);
    single.train(iters, root_fn);
    double e_single = single.exploitability(root_fn);

    auto game2 = FlatGame::compile(root_fn);
    ParallelMccfrSolver par1(std::move(game2), root_fn, 1, 42);
    par1.train(iters, root_fn);
    double e_par1 = par1.exploitability(root_fn);

    // Both should converge to similar quality (not identical: different
    // seed derivation, but similar range).
    CHECK(e_par1 < 0.5);
    CHECK(e_single < 0.5);
}

// ═══════════════════════════════════════════════════════════════════
// Stats and naming
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("par-mccfr: stats are accumulated across threads") {
    auto root_fn = kuhn::root_states;
    auto game = FlatGame::compile(root_fn);
    ParallelMccfrSolver solver(std::move(game), root_fn, 2, 42);
    solver.train(1000, root_fn);

    auto& st = solver.stats();
    CHECK(st.nodes_touched > 0);
    CHECK(st.terminal_visits > 0);
    CHECK(st.info_set_visits > 0);
    CHECK(solver.iterations() == 1000);
}

TEST_CASE("par-mccfr: name includes thread count") {
    auto root_fn = kuhn::root_states;
    auto game = FlatGame::compile(root_fn);
    ParallelMccfrSolver solver(std::move(game), root_fn, 4, 42);
    CHECK(solver.name() == "par-mccfr-4t");
}

TEST_CASE("par-mccfr: validate passes") {
    auto root_fn = kuhn::root_states;
    auto game = FlatGame::compile(root_fn);
    ParallelMccfrSolver solver(std::move(game), root_fn, 2, 42);
    solver.train(1000, root_fn);
    CHECK(solver.validate().empty());
}

// ═══════════════════════════════════════════════════════════════════
// Thread scaling (lightweight, in-test benchmark)
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("par-mccfr: 2 threads faster than 1 thread") {
    auto root_fn = leduc::root_states;
    constexpr int iters = 2'000'000;
    using Clock = std::chrono::steady_clock;

    auto game1 = FlatGame::compile(root_fn);
    ParallelMccfrSolver s1(std::move(game1), root_fn, 1, 42);
    auto t0 = Clock::now();
    s1.train(iters, root_fn);
    double sec1 = std::chrono::duration<double>(Clock::now() - t0).count();

    auto game2 = FlatGame::compile(root_fn);
    ParallelMccfrSolver s2(std::move(game2), root_fn, 2, 42);
    auto t1 = Clock::now();
    s2.train(iters, root_fn);
    double sec2 = std::chrono::duration<double>(Clock::now() - t1).count();

    double speedup = sec1 / sec2;
    std::cout << "  1 thread: " << std::fixed << std::setprecision(3)
              << sec1 << "s, 2 threads: " << sec2
              << "s, speedup: " << std::setprecision(2) << speedup << "×\n";

    // On a 2-core machine, expect ~1× (no slowdown) with possible
    // minor speedup. On 4+ core machines, expect 1.5×+.
    // The key correctness property: it doesn't get *slower* with
    // more threads (which would indicate contention bugs).
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw >= 4) {
        CHECK(speedup > 1.3);
    } else {
        // On 2 cores, just verify no significant slowdown.
        CHECK(speedup > 0.8);
    }
}
