// bluefish/cfr_plus.h - CFR+ (Tammelin 2014)

#pragma once

#include "solver.h"

namespace bluefish {

class CfrPlusSolver final : public Solver {
public:
	double train(int iterations, RootFn root_fn) override;
	std::string name() const override { return "cfr+"; }
	std::string validate() const override;

private:
	double cfr_plus(const GameState& state, double pi0, double pi1);
};

} // namespace bluefish
