// bluefish/cfr.h - Vanilla Counterfactual Regeret Minimization

#pragma once

#include "solver.h"

namespace bluefish {

class CfrSolver final : public Solver {
public:
	double train(int iterations, RootFn root_fr) override;
	std::string name() const override { return "cfr"; }

private:
	double cfr(const GameState& state, double pi0, double pi1);
};

} // namespace bluefish
