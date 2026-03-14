// bluefish/mccfr.h - External-sampling Monte Carlo CFR

#pragma once

#include "solver.h"

#include <random>

namespace bluefish {

class MccfrSolver final : public Solver {
public:
	explicit MccfrSolver(uint64_t seed = 42);

	double train(int iterations, RootFn root_fn) override;
	std::string name() const override { return "mccfr-es"; }

private:
	std::mt19937_64 rng_;

	double traverse(const GameState& state, int traverser,
	double pi_traverser, double pi_opponent);

	int sample_action(const std::vector<double>& probs);
};

} // namespace bluefish
