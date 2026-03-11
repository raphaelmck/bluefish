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
inline constexpr int kMaxRaises = 2;
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

		// - Information sets -

		std::string info_set_key() const override {
			std::string key;
			key += kRankName[rank(cards_[acting_])];
			if (board_ >= 0) {
				key += kRankName[rank(board_)];
			}
			key += ':';
			key.append(history_, static_cast<std::size_t>(hist_len_));
			return key;
		}

		// - Actions -
		
		std::vector<Action> legal_actions() const override {
			assert(status_ == Status::kPlaying);
			if (has_bet_) {
				if (bets_ < kMaxRaises) {
					return {kFold, kCall, kRaise};
				}
				return {kFold, kCall};
			}
			return {kCall, kRaise};
		}

		std::unique_ptr<GameState> act(Action a) const override {
			assert(status_ == Status::kPlaying);
			auto next = std::make_unique<LeducState>(*this);

			char hist_char = '?';

			switch (a) {
			case kFold:
				assert(has_bet_);
				next->status_ = Status::kFolded;
				next->folder_ = acting_;
				hist_char = 'f';
				break;

			case kCall:
				if (has_bet_) {
					next->stakes_[acting_] = stakes_[1 - acting_];
					hist_char = 'c';
					next->end_round();
				} else {
					// Check
					hist_char = 'k';
					next->round_actions_++;
					if (next->round_actions_ >= 2) {
						next->end_round();
					} else {
						next->acting_ = 1 - acting_;
					}
				}
				break;

			case kRaise:
				assert(bets_ < kMaxRaises);
				hist_char = has_bet_ ? 'r' : 'b';
				next->stakes_[acting_] = stakes_[1 - acting_] + raise_size();
				next->bets_++;
				next->round_actions_++;
				next->acting_ = 1 - acting_;
				break;
			
				default:
					assert(false && "invalid action");
			}

			next->history_[hist_len_] = hist_char;
			next->hist_len_ = hist_len_ + 1;
			return next;
		}
	
		// - Chance - 

		std::vector<ChanceOutcome> chance_outcomes() const override {
			assert(status_ == Status::kChance);
			std::vector<ChanceOutcome> outcomes;
			int remaining = kNumCards - 2;
			
			for (int c = 0; c < kNumCards; ++c) {
				if (c == cards_[0] || c == cards_[1]) continue;

				auto next = std::make_unique<LeducState>(*this);
				next->board_ = c;
				next->round_ = 1;
				next->status_ = Status::kPlaying;
				next->bets_ = 0;
				next->has_bet_ = false;
				next->history_[hist_len_] = '/'; // round separator
				next->hist_len_ = hist_len_ + 1;

				outcomes.push_back({std::move(next), 1.0 / static_cast<double>(remaining)});
			}
			return outcomes;
		}

private:
	enum class Status { kPlaying, kChance, kFolded, kShowdown };
	std::array<int, 2> cards_;
	int board_ = -1;
	int round_ = 0;
	int acting_ = 0;
	std::array<int, 2> stakes_;
	int bets_ = 0;
	bool has_bet_= false;
	int round_actions_ = false;
	Status status_ = Status::kPlaying;
	int folder_ = -1;
	char history_[24] = {};
	int hist_len_ = 0;

	int raise_size() const { return round_ == 0 ? kRaise0 : kRaise1; }

	void end_round() {
		if (round_ == 0) {
			status_ = Status::kChance;
		} else {
			status_ = Status::kShowdown;
		}
	}

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

// - Root state enumeration -
// All 30 ordered deals (6 cards, choose 2)

inline std::vector<RootState> root_states() {
	std::vector<RootState> roots;
	roots.reserve(30);
	constexpr double prob = 1.0 / 30.0;
	for (int c0 = 0; c0 < kNumCards; ++c0) {
		for (int c1 = 0; c1 < kNumCards; ++c1) {
			if (c0 != c1) {
				roots.push_back({
					std::make_unique<LeducState>(c0, c1), prob
				});
			}
		}
	}
	return roots;
}

} // namespace leduc
} // namespce bluefish
