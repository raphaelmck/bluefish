// bluefish/kuhn.h - Kuhn poker, the simplest nontrivial poker game

#pragma once

#include "game.h"

#include <array>
#include <cassert>
#include <algorithm>

namespace bluefish {
namespace kuhn {

// Actions
inline constexpr Action kPass = 0;
inline constexpr Action kBet = 1;

// Cards
inline constexpr int kJack = 0;
inline constexpr int kQueen = 1;
inline constexpr int kKing = 2;

inline constexpr char kCardName[] = "JQK";

class KuhnState final : public GameState {
public:
	KuhnState(int card0, int card1)
		: cards_{card0, card1} {
			assert(card0 != card1);
			assert(card0 >= 0 && card0 <= 2);
			assert(card1 >= 0 && card1 <= 2);
		}
	
	// Node type
	bool is_terminal() const override {
        // Terminal sequences: "pp", "bp", "bb", "pbp", "pbb"
		if (len_ < 2) return false;
		if (len_ == 2) return !(history_[0] == 'p' && history_[1] == 'b');
		return len_ == 3; // "pbp" or "pbb"
	}

	bool is_chance() const override { return false; }

	// Players
	int current_player() const override {
		// Alternative play: P0 at even lengths, P1 at odd lengths
		return len_ % 2;
	}

	// Terminal queries
	double utility(int player) const override {
		assert(is_terminal());
		
		bool p0_higher = cards_[0] > cards_[1];

		double u0 = 0.0;
		if (len_ == 2) {
			if (history_[0] == 'p' && history_[1] == 'p') {
				// Check-check -> showdown, pot = 2
				u0 = p0_higher ? 1.0 : -1.0;
			} else if (history_[0] == 'b' && history_[1] == 'p') {
				// Bet-fold -> P0 winds ante
				u0 = 1.0;
			} else {
				// Bet-call -> showdown, pot = 4
				u0 = p0_higher ? 2.0 : -2.0;
			}
		} else {
			// len_ == 3, history is "pb?"
			if (history_[2] == 'p') {
				// Pass-bet-fold -> P1 wins ante
				u0 = -1.0;
			} else {
				// Pass-bet-call -> showdown, pot = 4
				u0 = p0_higher ? 2.0 : -2.0;
			}
		}

		return player == 0 ? u0 : -u0;
	}

	// Information sets
	std::string info_set_key() const override {
		std::string key(1, kCardName[cards_[static_cast<std::size_t>(current_player())]]);
		key.append(history_, static_cast<std::size_t>(len_));
		return key;
	}

	// Actions
	std::vector<Action> legal_actions() const override { return {kPass, kBet }; }

	std::unique_ptr<GameState> act(Action a) const override {
		assert(a == kPass || a == kBet);
		assert(!is_terminal());

		auto next = std::make_unique<KuhnState>(cards_[0], cards_[1]);
		std::copy_n(history_, len_, next->history_);
		next->history_[len_] = (a == kPass) ? 'p' : 'b';
		next->len_ = len_ + 1;
		return next;
	}

private:
	std::array<int, 2> cards_;
	char history_[4] = {};
	int len_ = 0;
};

inline std::vector<RootState> root_states() {
	std::vector<RootState> roots;
	roots.reserve(6);
	for (int c0 = 0; c0 < 3; ++c0) {
		for (int c1 = 0; c1 < 3; ++c1) {
			if (c0 != c1) {
				roots.push_back({
					std::make_unique<KuhnState>(c0, c1),
					1.0 / 6.0
				});
			}
		}
	}
	return roots;
}

} // namespace kuhn
} // namespace bluefish
