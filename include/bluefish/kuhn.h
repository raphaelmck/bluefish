// bluefish/kuhn.h - Kuhn poker, the simplest nontrivial poker game

#pragma once

#include "game.h"

#include <array>
#include <cassert>

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

private:
	std::array<int, 2> cards_;
	char history_[4] = {};
	int len_ = 0;
};

}
}
