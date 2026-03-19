// Claude Opus 4.6
// test_abstraction.cpp — Card abstraction layer tests.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "bluefish/abstraction.h"
#include "bluefish/fast_cfr.h"
#include "bluefish/leduc.h"

#include <iostream>
#include <iomanip>

using namespace bluefish;

// ═══════════════════════════════════════════════════════════════════
// Abstraction construction
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("abstraction: exact is identity") {
    auto game = FlatGame::compile(leduc::root_states);
    auto abs = leduc_abs::exact(game);
    CHECK(abs.num_abstract == game.num_info_sets);
    CHECK(abs.num_abstract == 288);
    for (int i = 0; i < game.num_info_sets; ++i)
        CHECK(abs.mapping[static_cast<std::size_t>(i)] == i);
}

TEST_CASE("abstraction: jq-merge reduces info sets") {
    auto game = FlatGame::compile(leduc::root_states);
    auto abs = leduc_abs::jq_merge(game);
    CHECK(abs.num_abstract < 288);
    CHECK(abs.num_abstract > 50);
    // Multiple exact IDs should map to same abstract ID.
    bool found_merge = false;
    for (int i = 0; i < game.num_info_sets && !found_merge; ++i) {
        for (int j = i + 1; j < game.num_info_sets; ++j) {
            if (abs.mapping[static_cast<std::size_t>(i)] ==
                abs.mapping[static_cast<std::size_t>(j)]) {
                found_merge = true;
                break;
            }
        }
    }
    CHECK(found_merge);
}

TEST_CASE("abstraction: pair-only reduces info sets") {
    auto game = FlatGame::compile(leduc::root_states);
    auto abs = leduc_abs::pair_only(game);
    CHECK(abs.num_abstract < 288);
    CHECK(abs.num_abstract > abs.num_abstract / 2);  // moderate reduction
    // Preflop info sets should be unchanged (no board card).
    // Check a preflop key: exact and abstract should be the same.
    for (int i = 0; i < game.num_info_sets; ++i) {
        auto ui = static_cast<std::size_t>(i);
        auto& key = game.is_keys[ui];
        if (key.find('/') == std::string::npos) {
            // Preflop: mapping should be unique for this key.
            CHECK(abs.abstract_keys[static_cast<std::size_t>(abs.mapping[ui])]
                  == key);
        }
    }
}

TEST_CASE("abstraction: mapping values are in range") {
    auto game = FlatGame::compile(leduc::root_states);

    auto check = [&](const Abstraction& abs) {
        for (int i = 0; i < game.num_info_sets; ++i) {
            int m = abs.mapping[static_cast<std::size_t>(i)];
            CHECK(m >= 0);
            CHECK(m < abs.num_abstract);
        }
    };

    check(leduc_abs::exact(game));
    check(leduc_abs::jq_merge(game));
    check(leduc_abs::pair_only(game));
}

// ═══════════════════════════════════════════════════════════════════
// apply_abstraction
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("apply_abstraction: preserves tree structure") {
    auto exact = FlatGame::compile(leduc::root_states);
    auto abs = leduc_abs::jq_merge(exact);
    auto ag = apply_abstraction(exact, abs);

    // Same node count, same children, same terminal utilities.
    CHECK(ag.num_nodes() == exact.num_nodes());
    CHECK(ag.children.size() == exact.children.size());
    CHECK(ag.num_info_sets == abs.num_abstract);

    // Terminal utilities unchanged.
    for (int i = 0; i < exact.num_nodes(); ++i) {
        auto ui = static_cast<std::size_t>(i);
        if (exact.nodes[ui].type == FlatNode::kTerminal)
            CHECK(ag.nodes[ui].utility_p0 == exact.nodes[ui].utility_p0);
    }
}

TEST_CASE("apply_abstraction: exact abstraction preserves info sets") {
    auto exact = FlatGame::compile(leduc::root_states);
    auto abs = leduc_abs::exact(exact);
    auto ag = apply_abstraction(exact, abs);
    CHECK(ag.num_info_sets == exact.num_info_sets);
}

// ═══════════════════════════════════════════════════════════════════
// Training on abstract games
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("abstract: fast-cfr on abstract game converges") {
    auto exact = FlatGame::compile(leduc::root_states);
    auto abs = leduc_abs::jq_merge(exact);
    auto ag = apply_abstraction(exact, abs);

    auto root_fn = leduc::root_states;
    FastCfrSolver solver(std::move(ag), root_fn);
    solver.train(5000, root_fn);

    // Should converge in the abstract game.
    double e = solver.exploitability(root_fn);
    CHECK(e < 0.5);
    CHECK(solver.num_info_sets() == static_cast<std::size_t>(abs.num_abstract));
}

TEST_CASE("abstract: fewer info sets = faster training") {
    auto exact_game = FlatGame::compile(leduc::root_states);
    auto abs = leduc_abs::jq_merge(exact_game);

    // Abstract game has fewer info sets, so less memory per regret table.
    CHECK(abs.num_abstract < exact_game.num_info_sets);
}

// ═══════════════════════════════════════════════════════════════════
// Lifted exploitability
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("lifted: exact abstraction matches flat_exploitability") {
    auto exact = FlatGame::compile(leduc::root_states);
    auto abs = leduc_abs::exact(exact);

    // Train on exact game (through abstraction pipeline for consistency).
    auto ag = apply_abstraction(exact, abs);
    auto root_fn = leduc::root_states;
    FastCfrSolver solver(std::move(ag), root_fn);
    solver.train(5000, root_fn);

    // flat_exploitability on the abstract game (which is exact).
    double abstract_e = solver.exploitability(root_fn);

    // Lifted exploitability: evaluate abstract strategy on exact tree.
    // Since abstraction is identity, these should match.
    RegretTable lifted_table;
    lifted_table.init(abs.num_abstract, exact.is_num_actions);
    // Copy from solver's info map.
    for (auto& [key, node] : solver.info_map()) {
        // Find the abstract ID for this key.
        for (int i = 0; i < exact.num_info_sets; ++i) {
            if (exact.is_keys[static_cast<std::size_t>(i)] == key) {
                int off = lifted_table.offset[static_cast<std::size_t>(i)];
                for (int a = 0; a < node.num_actions; ++a) {
                    lifted_table.regret[static_cast<std::size_t>(off + a)] =
                        node.regret_sum[static_cast<std::size_t>(a)];
                    lifted_table.strategy_sum[static_cast<std::size_t>(off + a)] =
                        node.strategy_sum[static_cast<std::size_t>(a)];
                }
                break;
            }
        }
    }

    double lifted_e = lifted_flat_exploitability(exact, lifted_table, abs);
    CHECK(lifted_e == doctest::Approx(abstract_e).epsilon(0.001));
}

TEST_CASE("lifted: coarser abstraction has higher exploitability") {
    auto exact = FlatGame::compile(leduc::root_states);
    auto root_fn = leduc::root_states;
    constexpr int iters = 10'000;

    // Train exact.
    auto abs_exact = leduc_abs::exact(exact);
    auto g_exact = apply_abstraction(exact, abs_exact);
    FastCfrSolver s_exact(std::move(g_exact), root_fn);
    s_exact.train(iters, root_fn);
    double e_exact = s_exact.exploitability(root_fn);

    // Train jq-merge.
    auto abs_jq = leduc_abs::jq_merge(exact);
    auto g_jq = apply_abstraction(exact, abs_jq);
    // Save is_num_actions before move.
    auto jq_is_na = g_jq.is_num_actions;
    FastCfrSolver s_jq(std::move(g_jq), root_fn);
    s_jq.train(iters, root_fn);

    // Rebuild abstract RegretTable from solver's info_map.
    RegretTable tab_jq;
    tab_jq.init(abs_jq.num_abstract, jq_is_na);
    for (auto& [key, node] : s_jq.info_map()) {
        for (int aid = 0; aid < abs_jq.num_abstract; ++aid) {
            if (abs_jq.abstract_keys[static_cast<std::size_t>(aid)] == key) {
                int off = tab_jq.offset[static_cast<std::size_t>(aid)];
                for (int a = 0; a < node.num_actions; ++a) {
                    auto ua = static_cast<std::size_t>(a);
                    auto uoa = static_cast<std::size_t>(off + a);
                    tab_jq.regret[uoa] = node.regret_sum[ua];
                    tab_jq.strategy_sum[uoa] = node.strategy_sum[ua];
                }
                break;
            }
        }
    }
    double e_jq = lifted_flat_exploitability(exact, tab_jq, abs_jq);

    std::cout << "  Exact exploitability:    " << std::scientific
              << std::setprecision(4) << e_exact << " (" << exact.num_info_sets << " IS)\n";
    std::cout << "  JQ-merge lifted exploit: " << e_jq
              << " (" << abs_jq.num_abstract << " IS)\n";

    // The abstract strategy should have HIGHER exploitability when
    // evaluated on the exact game, because it can't distinguish J from Q.
    CHECK(e_jq > e_exact);
}

// ═══════════════════════════════════════════════════════════════════
// Info set count report
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("abstraction: info set counts") {
    auto game = FlatGame::compile(leduc::root_states);
    auto exact = leduc_abs::exact(game);
    auto jq = leduc_abs::jq_merge(game);
    auto po = leduc_abs::pair_only(game);

    std::cout << "  Leduc info set counts:\n"
              << "    exact:     " << exact.num_abstract << "\n"
              << "    pair-only: " << po.num_abstract << "\n"
              << "    jq-merge:  " << jq.num_abstract << "\n";

    CHECK(exact.num_abstract == 288);
    CHECK(jq.num_abstract < po.num_abstract);  // jq is coarser
    CHECK(po.num_abstract < exact.num_abstract);
}
