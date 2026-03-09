// Claude Opus 4.6
// test_kuhn_cfr.cpp — Tests for Kuhn poker and vanilla CFR.
//
// Verifies:
//   • Kuhn poker terminal utilities are correct.
//   • Information set keys are well-formed.
//   • CFR converges: exploitability decreases over iterations.
//   • Average strategy approaches known Nash equilibrium properties.
//   • Game value converges to −1/18.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "bluefish/cfr.h"
#include "bluefish/kuhn.h"

using namespace bluefish;
using namespace bluefish::kuhn;

// ═══════════════════════════════════════════════════════════════════
// Kuhn Poker State Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("kuhn: initial state is not terminal") {
    KuhnState s(kJack, kQueen);
    CHECK_FALSE(s.is_terminal());
    CHECK_FALSE(s.is_chance());
    CHECK(s.current_player() == 0);
}

TEST_CASE("kuhn: two actions available at every decision node") {
    KuhnState s(kKing, kJack);
    auto actions = s.legal_actions();
    CHECK(actions.size() == 2);
    CHECK(actions[0] == kPass);
    CHECK(actions[1] == kBet);
}

TEST_CASE("kuhn: terminal utilities for check-check") {
    // P0=Q, P1=K → P1 wins → u(P0) = -1, u(P1) = +1
    KuhnState s(kQueen, kKing);
    auto s1 = s.act(kPass);                // P0 checks
    auto s2 = s1->act(kPass);              // P1 checks → showdown
    CHECK(s2->is_terminal());
    CHECK(s2->utility(0) == doctest::Approx(-1.0));
    CHECK(s2->utility(1) == doctest::Approx(1.0));
}

TEST_CASE("kuhn: terminal utilities for bet-fold") {
    // P0 bets, P1 folds → P0 wins ante
    KuhnState s(kJack, kKing);
    auto s1 = s.act(kBet);
    auto s2 = s1->act(kPass);
    CHECK(s2->is_terminal());
    CHECK(s2->utility(0) == doctest::Approx(1.0));
    CHECK(s2->utility(1) == doctest::Approx(-1.0));
}

TEST_CASE("kuhn: terminal utilities for bet-call showdown") {
    // P0=K, P1=J → bet-call → P0 wins pot of 4
    KuhnState s(kKing, kJack);
    auto s1 = s.act(kBet);
    auto s2 = s1->act(kBet);              // call
    CHECK(s2->is_terminal());
    CHECK(s2->utility(0) == doctest::Approx(2.0));
    CHECK(s2->utility(1) == doctest::Approx(-2.0));
}

TEST_CASE("kuhn: check-bet-fold terminal") {
    // P0 checks, P1 bets, P0 folds → P1 wins
    KuhnState s(kJack, kQueen);
    auto s1 = s.act(kPass);
    auto s2 = s1->act(kBet);
    CHECK_FALSE(s2->is_terminal());       // P0 must respond
    CHECK(s2->current_player() == 0);

    auto s3 = s2->act(kPass);             // P0 folds
    CHECK(s3->is_terminal());
    CHECK(s3->utility(0) == doctest::Approx(-1.0));
}

TEST_CASE("kuhn: check-bet-call showdown") {
    // P0=K, P1=J → check, bet, call → showdown
    KuhnState s(kKing, kJack);
    auto s1 = s.act(kPass);
    auto s2 = s1->act(kBet);
    auto s3 = s2->act(kBet);              // call
    CHECK(s3->is_terminal());
    CHECK(s3->utility(0) == doctest::Approx(2.0));
}

TEST_CASE("kuhn: info set keys encode card and history") {
    KuhnState s(kQueen, kKing);
    CHECK(s.info_set_key() == "Q");        // P0 sees Q, no history

    auto s1 = s.act(kPass);
    CHECK(s1->info_set_key() == "Kp");     // P1 sees K, history "p"

    auto s2 = s1->act(kBet);
    CHECK(s2->info_set_key() == "Qpb");    // P0 sees Q, history "pb"
}

TEST_CASE("kuhn: root states enumerate 6 deals") {
    auto roots = root_states();
    CHECK(roots.size() == 6);

    double total_prob = 0.0;
    for (auto& r : roots) {
        total_prob += r.probability;
        CHECK_FALSE(r.state->is_terminal());
    }
    CHECK(total_prob == doctest::Approx(1.0));
}

// ═══════════════════════════════════════════════════════════════════
// InfoNode Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("info node: uniform strategy when all regrets non-positive") {
    InfoNode node;
    node.init(3);
    node.regret_sum = {-1.0, -2.0, 0.0};

    auto s = node.current_strategy();
    CHECK(s[0] == doctest::Approx(1.0 / 3.0));
    CHECK(s[1] == doctest::Approx(1.0 / 3.0));
    CHECK(s[2] == doctest::Approx(1.0 / 3.0));
}

TEST_CASE("info node: regret matching with positive regrets") {
    InfoNode node;
    node.init(2);
    node.regret_sum = {3.0, 1.0};

    auto s = node.current_strategy();
    CHECK(s[0] == doctest::Approx(0.75));
    CHECK(s[1] == doctest::Approx(0.25));
}

TEST_CASE("info node: average strategy normalization") {
    InfoNode node;
    node.init(2);
    node.strategy_sum = {100.0, 300.0};

    auto avg = node.average_strategy();
    CHECK(avg[0] == doctest::Approx(0.25));
    CHECK(avg[1] == doctest::Approx(0.75));
}

// ═══════════════════════════════════════════════════════════════════
// CFR Convergence Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("cfr: exploitability decreases with iterations") {
    CfrTrainer trainer;
    auto root_fn = root_states;

    trainer.train(100, root_fn);
    double exploit_100 = trainer.exploitability(root_fn);

    trainer.train(900, root_fn);
    double exploit_1k = trainer.exploitability(root_fn);

    trainer.train(9'000, root_fn);
    double exploit_10k = trainer.exploitability(root_fn);

    // Exploitability should decrease monotonically (approximately).
    CHECK(exploit_1k < exploit_100);
    CHECK(exploit_10k < exploit_1k);

    // After 10k iterations, exploitability should be very small.
    CHECK(exploit_10k < 0.01);
}

TEST_CASE("cfr: discovers all 12 information sets") {
    CfrTrainer trainer;
    trainer.train(1, root_states);
    // Kuhn poker has 12 info sets: 3 cards × {P0 has 2 info sets, P1 has 2}
    // Actually: P0 at root (3), P1 after "p" (3), P1 after "b" (3),
    // P0 after "pb" (3) = 12.
    CHECK(trainer.num_info_sets() == 12);
}

TEST_CASE("cfr: game value converges to -1/18") {
    CfrTrainer trainer;
    double gv = trainer.train(50'000, root_states);

    // Game value for P0 in Kuhn poker = −1/18 ≈ −0.0556.
    CHECK(gv == doctest::Approx(-1.0 / 18.0).epsilon(0.01));
}

TEST_CASE("cfr: known Nash equilibrium properties after convergence") {
    CfrTrainer trainer;
    trainer.train(100'000, root_states);

    auto get_avg = [&](const std::string& key) {
        return trainer.info_map().at(key).average_strategy();
    };

    // ── Player 1 (P1) has tight equilibrium constraints ────────

    // P1 with Jack after P0 bets: must fold (pass). bet_prob ≈ 0.
    {
        auto avg = get_avg("Jb");
        CHECK(avg[1] < 0.05);   // bet (call) frequency near 0
    }

    // P1 with King after P0 bets: must call. bet_prob ≈ 1.
    {
        auto avg = get_avg("Kb");
        CHECK(avg[1] > 0.95);   // call frequency near 1
    }

    // P1 with King after P0 checks: must bet. bet_prob ≈ 1.
    {
        auto avg = get_avg("Kp");
        CHECK(avg[1] > 0.95);
    }

    // P1 with Queen after P0 bets: call with prob 1/3.
    {
        auto avg = get_avg("Qb");
        CHECK(avg[1] == doctest::Approx(1.0 / 3.0).epsilon(0.05));
    }

    // P1 with Jack after P0 checks: bet with prob 1/3.
    {
        auto avg = get_avg("Jp");
        CHECK(avg[1] == doctest::Approx(1.0 / 3.0).epsilon(0.05));
    }

    // ── Player 0 (P0) ──────────────────────────────────────────

    // P0 with Queen: always checks at root. bet_prob ≈ 0.
    {
        auto avg = get_avg("Q");
        CHECK(avg[1] < 0.05);
    }

    // P0 with Queen facing bet (Qpb): indifferent — any call probability
    // is Nash (EV of calling = EV of folding = -1).
    {
        auto avg = get_avg("Qpb");
        CHECK(avg[0] + avg[1] == doctest::Approx(1.0));
    }

    // P0 with King facing bet (Kpb): always calls.
    {
        auto avg = get_avg("Kpb");
        CHECK(avg[1] > 0.95);
    }

    // P0 with Jack: bluffs with some α ∈ [0, 1/3].
    // P0 with King: bets with prob 3α (correlated with J's bluff freq).
    {
        auto avg_j = get_avg("J");
        auto avg_k = get_avg("K");
        double alpha = avg_j[1];
        CHECK(alpha >= -0.01);
        CHECK(alpha <= 0.40);  // ≤ 1/3 + tolerance
        CHECK(avg_k[1] == doctest::Approx(3.0 * alpha).epsilon(0.05));
    }
}
