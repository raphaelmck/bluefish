// bluefish/leduc.h - Ledux poker: the standard mid-complexity poker testbed

#pragma once

#include "game.h"

#include <cassert>

namespace bluefish {
namespace leduc {

// - Constants -

inline constexpr int kNumCards = 6;
inline constexpr int kNumRanks = 3;
inline constexpr char kRankName[] = "JQK";

inline int rank(int card) { return card / 2; }

// Actions
inline constexpr Action kFold = 0;
inline constexpr Action kCall = 1;
inline constexpr Action kRaise = 2;

// Betting parameters
inline constexpr int kAnte = 1;
inline constexpr int kMaxRaise = 2;
inline constexpr int kRaise0 = 2;
inline constexpr int kRaise1 = 4;

// - State -

class LeducState final : public GameState {
public:
	LeducState(int card0, int card1)
		: cards_{card0, card1}, stakes_{kAnte, kAnte} {
			assert(card0 != card1);
			assert(card0 >= 0 && card0 < kNumCards);
			assert(card1 >= 0 && card1 < kNumCards);
		}

		// - Node Type -

		bool is_terminal() const override {
			return status_ == Status::kFolded || status_ == Status::kShowdown;
		}

		bool is_chance() const override {
			return status_ == Status::kChance;
		}

		// - Players -

		int  current_player() const override { return acting_; }

		// - Terminal _

		double utility(int player) const override {
			assert(is_terminal());

			int winner;
			if (status_ == Status::kFolded) {
				winner = 1 - folder_;
			} else {
				winner = showdown_winner();
			}

			if (winner == -1) return 0.0;

			double u0 = (winner == 0) ? static_cast<double>(stakes_[1]) 
									  : -static_cast<double>(stakes_[0]);
			return player == 0 ? u0 : -u0;
		}

private:
	enum class Status { kPlaying, kChance, kFolded, kShowdown };
	std::array<int, 2> cards_;
	int board_ = -1;
	int round_ = 0;
	int acting_ = 0;
	std::array<int, 2> stakes_;
	int bets_ = 0;
	bool has_bets_= false;
	int round_actions = false;
	Status status_ = Status::kPlaying;
	int folder_ = -1;
	char history_[24] = {};
	int hist_len_ = 0;

	int raise_size() const { return round_ == 0 ? kRaise0 : kRaise1; }

	int showdown_winner() const {
		int r0 = rank(cards_[0]);
		int r1 = rank(cards_[1]);
		int rb = rank(board_);

		bool pair0 = (r0 == rb);
		bool pair1 = (r1 == rb);

		if (pair0 && !pair1) return 0;
		if (pair1 && !pair0) return 1;
		if (r0 > r1) return 0;
		if (r1 > r0) return 1;
		return -1; // tie, same rank no pair
	}
};

}
}
