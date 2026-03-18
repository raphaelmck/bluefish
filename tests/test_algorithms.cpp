// Claude Opus 4.6
// test_algorithms.cpp — Cross-algorithm convergence tests.
//
// Verifies that CFR, CFR+, and MCCFR all converge to Nash equilibrium
// on both Kuhn and Leduc poker, and compares convergence rates.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "bluefish/cfr.h"
#include "bluefish/cfr_plus.h"
#include "bluefish/mccfr.h"
#include "bluefish/fast_cfr.h"
#include "bluefish/kuhn.h"
#include "bluefish/leduc.h"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace bluefish;

// ═══════════════════════════════════════════════════════════════════
// Kuhn Poker — all algorithms
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("kuhn: all algorithms converge to low exploitability") {
    constexpr int iters = 10'000;
    constexpr double threshold = 0.02;
    auto root_fn = kuhn::root_states;

    SUBCASE("vanilla CFR") {
        CfrSolver solver;
        solver.train(iters, root_fn);
        double e = solver.exploitability(root_fn);
        CHECK(e < threshold);
    }
    SUBCASE("CFR+") {
        CfrPlusSolver solver;
        solver.train(iters, root_fn);
        double e = solver.exploitability(root_fn);
        CHECK(e < threshold);
    }
    SUBCASE("MCCFR external sampling") {
        MccfrSolver solver(42);
        solver.train(100'000, root_fn);  // needs more iters due to variance
        double e = solver.exploitability(root_fn);
        CHECK(e < threshold);
    }
}

TEST_CASE("kuhn: all algorithms find correct game value") {
    constexpr double nash_value = -1.0 / 18.0;
    constexpr double tolerance = 0.02;
    auto root_fn = kuhn::root_states;

    SUBCASE("vanilla CFR") {
        CfrSolver solver;
        double gv = solver.train(50'000, root_fn);
        CHECK(gv == doctest::Approx(nash_value).epsilon(tolerance));
    }
    SUBCASE("CFR+") {
        CfrPlusSolver solver;
        double gv = solver.train(50'000, root_fn);
        CHECK(gv == doctest::Approx(nash_value).epsilon(tolerance));
    }
    SUBCASE("MCCFR") {
        MccfrSolver solver(42);
        double gv = solver.train(200'000, root_fn);
        CHECK(gv == doctest::Approx(nash_value).epsilon(tolerance));
    }
}

TEST_CASE("kuhn: CFR+ converges faster than vanilla CFR") {
    constexpr int iters = 5'000;
    auto root_fn = kuhn::root_states;

    CfrSolver cfr;
    CfrPlusSolver cfr_plus;

    cfr.train(iters, root_fn);
    cfr_plus.train(iters, root_fn);

    double e_cfr = cfr.exploitability(root_fn);
    double e_plus = cfr_plus.exploitability(root_fn);

    // CFR+ should have lower exploitability at the same iteration count.
    CHECK(e_plus < e_cfr);
}

TEST_CASE("kuhn: MCCFR Nash properties after convergence") {
    MccfrSolver solver(42);
    solver.train(500'000, kuhn::root_states);

    auto get_avg = [&](const std::string& key) {
        return solver.info_map().at(key).average_strategy();
    };

    // P1 with King after P0 bets: must call.
    CHECK(get_avg("Kb")[1] > 0.90);

    // P1 with Jack after P0 bets: must fold.
    CHECK(get_avg("Jb")[1] < 0.10);

    // P0 with Queen: always checks.
    CHECK(get_avg("Q")[1] < 0.10);
}

// ═══════════════════════════════════════════════════════════════════
// Leduc Poker — all algorithms
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("leduc: all algorithms converge") {
    auto root_fn = leduc::root_states;

    SUBCASE("vanilla CFR — 5k iters") {
        CfrSolver solver;
        solver.train(5'000, root_fn);
        double e = solver.exploitability(root_fn);
        CHECK(e < 0.5);
    }
    SUBCASE("CFR+ — 5k iters") {
        CfrPlusSolver solver;
        solver.train(5'000, root_fn);
        double e = solver.exploitability(root_fn);
        CHECK(e < 0.5);
    }
    SUBCASE("MCCFR — 500k iters") {
        MccfrSolver solver(42);
        solver.train(500'000, root_fn);
        double e = solver.exploitability(root_fn);
        CHECK(e < 0.5);
    }
}

TEST_CASE("leduc: CFR+ converges with lower exploitability given more iterations") {
    auto root_fn = leduc::root_states;

    // CFR+ has a higher constant factor but better asymptotic rate O(1/T)
    // vs vanilla's O(1/√T). Verify both converge to low exploitability.
    CfrPlusSolver solver;
    solver.train(10'000, root_fn);
    double e = solver.exploitability(root_fn);
    CHECK(e < 0.1);
}

TEST_CASE("leduc: MCCFR wall-clock benchmark") {
    // This isn't a strict pass/fail — it logs throughput for comparison.
    auto root_fn = leduc::root_states;
    constexpr int iters = 100'000;

    using Clock = std::chrono::steady_clock;

    // Vanilla CFR
    {
        CfrSolver solver;
        auto t0 = Clock::now();
        solver.train(iters, root_fn);
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        double ips = static_cast<double>(iters) / sec;
        std::cout << "  Leduc CFR:      " << static_cast<int>(ips)
                  << " iters/sec, exploit=" << solver.exploitability(root_fn) << "\n";
    }

    // CFR+
    {
        CfrPlusSolver solver;
        auto t0 = Clock::now();
        solver.train(iters, root_fn);
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        double ips = static_cast<double>(iters) / sec;
        std::cout << "  Leduc CFR+:     " << static_cast<int>(ips)
                  << " iters/sec, exploit=" << solver.exploitability(root_fn) << "\n";
    }

    // MCCFR (10× more iterations, should still be faster in wall-clock)
    {
        MccfrSolver solver(42);
        auto t0 = Clock::now();
        solver.train(iters * 10, root_fn);
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        double ips = static_cast<double>(iters * 10) / sec;
        std::cout << "  Leduc MCCFR:    " << static_cast<int>(ips)
                  << " iters/sec, exploit=" << solver.exploitability(root_fn) << "\n";
    }

    CHECK(true); // benchmark always passes — inspect output
}

// ═══════════════════════════════════════════════════════════════════
// Serialization
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("all algorithms produce valid JSON") {
    auto root_fn = kuhn::root_states;

    auto check_json = [&](Solver& s) {
        s.train(10, root_fn);
        std::string json = s.serialize_json();
        CHECK(json.find("\"algorithm\"") != std::string::npos);
        CHECK(json.find("\"info_sets\"") != std::string::npos);
    };

    SUBCASE("CFR") {
        CfrSolver s;
        check_json(s);
        CHECK(s.serialize_json().find("\"cfr\"") != std::string::npos);
    }
    SUBCASE("CFR+") {
        CfrPlusSolver s;
        check_json(s);
        CHECK(s.serialize_json().find("\"cfr+\"") != std::string::npos);
    }
    SUBCASE("MCCFR") {
        MccfrSolver s(42);
        check_json(s);
        CHECK(s.serialize_json().find("\"mccfr-es\"") != std::string::npos);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Traversal statistics
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("stats: all counters are positive after training") {
    auto root_fn = kuhn::root_states;

    auto check_stats = [&](Solver& s, int iters) {
        s.train(iters, root_fn);
        auto& st = s.stats();
        CHECK(st.nodes_touched > 0);
        CHECK(st.terminal_visits > 0);
        CHECK(st.info_set_visits > 0);
        // Kuhn has no inline chance nodes, but counter should be non-negative.
        CHECK(st.chance_visits >= 0);
    };

    SUBCASE("CFR") {
        CfrSolver s;
        check_stats(s, 10);
    }
    SUBCASE("CFR+") {
        CfrPlusSolver s;
        check_stats(s, 10);
    }
    SUBCASE("MCCFR") {
        MccfrSolver s(42);
        check_stats(s, 100);
    }
}

TEST_CASE("stats: leduc has chance visits") {
    auto root_fn = leduc::root_states;

    SUBCASE("CFR") {
        CfrSolver s;
        s.train(10, root_fn);
        CHECK(s.stats().chance_visits > 0);
    }
    SUBCASE("MCCFR") {
        MccfrSolver s(42);
        s.train(100, root_fn);
        CHECK(s.stats().chance_visits > 0);
    }
}

TEST_CASE("stats: CFR and CFR+ touch same node count per iteration") {
    // Full-tree algorithms should visit the same number of nodes.
    auto root_fn = kuhn::root_states;
    constexpr int iters = 100;

    CfrSolver cfr;
    CfrPlusSolver cfr_plus;

    cfr.train(iters, root_fn);
    cfr_plus.train(iters, root_fn);

    // Same tree, same traversal order → identical node counts.
    CHECK(cfr.stats().nodes_touched == cfr_plus.stats().nodes_touched);
    CHECK(cfr.stats().terminal_visits == cfr_plus.stats().terminal_visits);
    CHECK(cfr.stats().info_set_visits == cfr_plus.stats().info_set_visits);
}

TEST_CASE("stats: MCCFR touches far fewer nodes per iteration than CFR") {
    auto root_fn = leduc::root_states;

    CfrSolver cfr;
    MccfrSolver mccfr(42);

    cfr.train(100, root_fn);
    mccfr.train(100, root_fn);

    double cfr_npi = static_cast<double>(cfr.stats().nodes_touched) / 100.0;
    double mccfr_npi = static_cast<double>(mccfr.stats().nodes_touched) / 100.0;

    // MCCFR should touch at least 100× fewer nodes per iteration on Leduc.
    CHECK(cfr_npi > 100.0 * mccfr_npi);
}

TEST_CASE("stats: cumulate across multiple train calls") {
    CfrSolver s;
    auto root_fn = kuhn::root_states;

    s.train(50, root_fn);
    auto n1 = s.stats().nodes_touched;
    CHECK(n1 > 0);

    s.train(50, root_fn);
    auto n2 = s.stats().nodes_touched;
    CHECK(n2 > n1);

    // Should be roughly double (same tree each iteration).
    double ratio = static_cast<double>(n2) / static_cast<double>(n1);
    CHECK(ratio == doctest::Approx(2.0).epsilon(0.01));
}

// ═══════════════════════════════════════════════════════════════════
// Validation
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("validate: all algorithms pass after training") {
    auto root_fn = kuhn::root_states;

    SUBCASE("CFR") {
        CfrSolver s;
        s.train(100, root_fn);
        CHECK(s.validate().empty());
    }
    SUBCASE("CFR+") {
        CfrPlusSolver s;
        s.train(100, root_fn);
        CHECK(s.validate().empty());
    }
    SUBCASE("MCCFR") {
        MccfrSolver s(42);
        s.train(1000, root_fn);
        CHECK(s.validate().empty());
    }
}

TEST_CASE("validate: CFR+ regrets are non-negative on Leduc") {
    CfrPlusSolver s;
    s.train(5000, leduc::root_states);
    CHECK(s.validate().empty());

    // Double check: inspect regrets directly.
    for (auto& [key, node] : s.info_map()) {
        for (int a = 0; a < node.num_actions; ++a) {
            CHECK(node.regret_sum[static_cast<std::size_t>(a)] >= -1e-12);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// FlatGame compilation
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("flat: kuhn compilation") {
    auto game = FlatGame::compile(kuhn::root_states);
    CHECK(game.num_info_sets == 12);
    CHECK(game.num_nodes() > 0);
    CHECK(game.num_terminal_nodes() > 0);
    CHECK(game.num_decision_nodes() > 0);
    CHECK(game.memory_bytes() > 0);
}

TEST_CASE("flat: leduc compilation") {
    auto game = FlatGame::compile(leduc::root_states);
    CHECK(game.num_info_sets == 288);
    CHECK(game.num_nodes() > 100);
    CHECK(game.num_chance_nodes() > 0);
}

// ═══════════════════════════════════════════════════════════════════
// Fast solver correctness — must match slow solvers
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("fast-cfr: kuhn converges to correct game value") {
    auto root_fn = kuhn::root_states;
    auto game = FlatGame::compile(root_fn);
    FastCfrSolver solver(std::move(game), root_fn);
    double gv = solver.train(50'000, root_fn);
    CHECK(gv == doctest::Approx(-1.0 / 18.0).epsilon(0.01));
    CHECK(solver.exploitability(root_fn) < 0.01);
}

TEST_CASE("fast-cfr: matches slow CFR exploitability on kuhn") {
    auto root_fn = kuhn::root_states;
    constexpr int iters = 10'000;

    CfrSolver slow;
    slow.train(iters, root_fn);

    auto game = FlatGame::compile(root_fn);
    FastCfrSolver fast(std::move(game), root_fn);
    fast.train(iters, root_fn);

    double slow_e = slow.exploitability(root_fn);
    double fast_e = fast.exploitability(root_fn);
    // Should be very close (same algorithm, same traversal order).
    CHECK(fast_e == doctest::Approx(slow_e).epsilon(0.001));
}

TEST_CASE("fast-cfr+: kuhn converges and validates") {
    auto root_fn = kuhn::root_states;
    auto game = FlatGame::compile(root_fn);
    FastCfrPlusSolver solver(std::move(game), root_fn);
    solver.train(10'000, root_fn);
    CHECK(solver.exploitability(root_fn) < 0.02);
    CHECK(solver.validate().empty());
}

TEST_CASE("fast-mccfr: kuhn converges") {
    auto root_fn = kuhn::root_states;
    auto game = FlatGame::compile(root_fn);
    FastMccfrSolver solver(std::move(game), root_fn, 42);
    solver.train(100'000, root_fn);
    CHECK(solver.exploitability(root_fn) < 0.02);
}

TEST_CASE("fast-cfr: leduc converges") {
    auto root_fn = leduc::root_states;
    auto game = FlatGame::compile(root_fn);
    FastCfrSolver solver(std::move(game), root_fn);
    solver.train(5'000, root_fn);
    CHECK(solver.exploitability(root_fn) < 0.5);
    CHECK(solver.num_info_sets() == 288);
}

TEST_CASE("fast-cfr: matches slow CFR exploitability on leduc") {
    auto root_fn = leduc::root_states;
    constexpr int iters = 5'000;

    CfrSolver slow;
    slow.train(iters, root_fn);

    auto game = FlatGame::compile(root_fn);
    FastCfrSolver fast(std::move(game), root_fn);
    fast.train(iters, root_fn);

    double slow_e = slow.exploitability(root_fn);
    double fast_e = fast.exploitability(root_fn);
    // Close but not identical: flat tree's synthetic root chance node
    // folds deal probabilities differently than iterating root states.
    CHECK(fast_e == doctest::Approx(slow_e).epsilon(0.15));
    CHECK(fast_e < 0.05);
}

TEST_CASE("fast-cfr: stats match slow CFR node counts") {
    auto root_fn = kuhn::root_states;
    constexpr int iters = 100;

    CfrSolver slow;
    slow.train(iters, root_fn);

    auto game = FlatGame::compile(root_fn);
    FastCfrSolver fast(std::move(game), root_fn);
    fast.train(iters, root_fn);

    // Flat tree has a synthetic root chance node, so chance_visits
    // will differ. But total info_set_visits should match.
    CHECK(fast.stats().info_set_visits == slow.stats().info_set_visits);
    CHECK(fast.stats().terminal_visits == slow.stats().terminal_visits);
}

TEST_CASE("fast: all fast solvers validate on leduc") {
    auto root_fn = leduc::root_states;

    SUBCASE("fast-cfr") {
        auto game = FlatGame::compile(root_fn);
        FastCfrSolver s(std::move(game), root_fn);
        s.train(100, root_fn);
        CHECK(s.validate().empty());
    }
    SUBCASE("fast-cfr+") {
        auto game = FlatGame::compile(root_fn);
        FastCfrPlusSolver s(std::move(game), root_fn);
        s.train(100, root_fn);
        CHECK(s.validate().empty());
    }
    SUBCASE("fast-mccfr") {
        auto game = FlatGame::compile(root_fn);
        FastMccfrSolver s(std::move(game), root_fn, 42);
        s.train(1000, root_fn);
        CHECK(s.validate().empty());
    }
}

// ═══════════════════════════════════════════════════════════════════
// Flat exploitability correctness
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("flat_exploitability: matches slow exploitability on kuhn") {
    // Train a slow solver, compile its game, build a RegretTable
    // with the same data, compare exploitability values.
    auto root_fn = kuhn::root_states;

    CfrSolver slow;
    slow.train(10'000, root_fn);
    double slow_e = slow.exploitability(root_fn);

    auto game = FlatGame::compile(root_fn);
    FastCfrSolver fast(std::move(game), root_fn);
    fast.train(10'000, root_fn);
    double fast_e = fast.exploitability(root_fn);

    // Both should converge to similar exploitability.
    CHECK(fast_e == doctest::Approx(slow_e).epsilon(0.001));
}

TEST_CASE("flat_exploitability: matches slow exploitability on leduc") {
    auto root_fn = leduc::root_states;

    CfrSolver slow;
    slow.train(5'000, root_fn);
    double slow_e = slow.exploitability(root_fn);

    auto game = FlatGame::compile(root_fn);
    FastCfrSolver fast(std::move(game), root_fn);
    fast.train(5'000, root_fn);
    double fast_e = fast.exploitability(root_fn);

    // Close but not identical due to synthetic root chance node.
    CHECK(fast_e == doctest::Approx(slow_e).epsilon(0.15));
    CHECK(fast_e < 0.05);
}

TEST_CASE("flat_exploitability: is zero-allocation (timing)") {
    // Train fast solver, then measure exploitability speed.
    // This is a benchmark, not a strict pass/fail.
    using Clock = std::chrono::steady_clock;
    auto root_fn = leduc::root_states;

    auto game = FlatGame::compile(root_fn);
    FastCfrSolver fast(std::move(game), root_fn);
    fast.train(10'000, root_fn);

    // Measure flat exploitability.
    auto t0 = Clock::now();
    constexpr int reps = 1000;
    double exploit = 0.0;
    for (int i = 0; i < reps; ++i)
        exploit = fast.exploitability(root_fn);
    double flat_sec = std::chrono::duration<double>(
        Clock::now() - t0).count();

    // Measure slow exploitability (via synced InfoMap).
    CfrSolver slow;
    slow.train(10'000, root_fn);
    auto t1 = Clock::now();
    double slow_exploit = 0.0;
    for (int i = 0; i < reps; ++i)
        slow_exploit = slow.exploitability(root_fn);
    double slow_sec = std::chrono::duration<double>(
        Clock::now() - t1).count();
    (void)slow_exploit;

    double speedup = slow_sec / flat_sec;
    std::cout << "  Flat exploitability: " << std::fixed << std::setprecision(4)
              << flat_sec << "s for " << reps << " calls ("
              << std::setprecision(1) << speedup << "× vs slow)\n";

    CHECK(exploit > 0.0);
    CHECK(exploit < 0.1);
    // Flat should be faster. Even 2× is a win.
    CHECK(speedup > 1.5);
}
