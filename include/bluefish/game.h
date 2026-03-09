// Every game in Bluefish implements this interface. A GameState is an immutable
// snapshot of a game at one point in the game tree. Calling act() returns a new
// state; the original is unchanged.

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace bluefish {
	
// Actions are small non-negative integers. Each game defines its own mapping
using Action = int;

class GameState {
public:
	virtual ~GameState() = default;

	// Node type
	virtual bool is_terminal() const = 0;
	virtual bool is_chance() const = 0;

	// Players
	// Returns 0 or 1 for a two-player game. Undefined at terminal/chance
	virtual int current_player() const = 0;

	// Terminal queries
	virtual double utility(int player) const = 0;

	// Information sets
	virtual std::string info_set_key() const = 0;

	// Actions
	virtual std::vector<Action> legal_actions() const = 0;

	virtual std::unique_ptr<GameState> act(Action a) const = 0;

	// Chance
	struct ChanceOutcome {
		std::unique_ptr<GameState> state;
		double probability;
	};

	virtual std:: vector<ChanceOutcome> chance_outcomes() const { return {}; }
};

struct RootState {
	std::unique_ptr<GameState> state;
	double probability;
};

}
