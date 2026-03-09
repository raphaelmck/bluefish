// solve.cpp - Train vanilla CFR on kuhn poker and inspect results

#include "bluefish/cfr.h"
#include "bluefish/kuhn.h"

#include <map>
#include <iostream>
#include <iomanip>

namespace {

constexpr int kDefaultIterations = 100'000;

// Display the average stratey at every information set, grouped by player
void print_strategy(const bluefish::CfrTrainer& trainer) {
	std::map<std::string, std::vector<double>> strategies;
	for (auto& [key, node] : trainer.info_map()) {
		strategies[key] = node.average_srategy();
	}

	auto player_of = [](const std::string& key) {
		// History length determines the player for Kuhn
		return static_cast<int>((key.size() - 1) % 2);
	};

	for (int p = 0; p < 2; ++p) {
		std::cout << p << ":\n";
		std::cout << std::setw(10) << "info set"
				  << std::setw(12) << "pass"
				  << std::setw(12) << "bet" << "\n";
		std::cout << std::string(34, '-') << "\n";

		for (auto& [key, avg] : strategies) {
			if (player_of(key) != p) continue;
			std::cout << std::setw(10) << key
					  << std::fixed << std::setprecision(4)
					  << std::setw(12) << avg[0]
					  << std::setw(12) << avg[1]
					  << "\n";
		}
		std::cout << "\n";
	}
}

// Print convergence samples: exploitability at logarithmic checkpoints
void train_with_logging(bluefish::CfrTrainer& trainer, int iterations) {
	auto root_fn = bluefish::kuhn::root_states;

	std::cout << std::setw(12) << "iteration"
              << std::setw(16) << "exploitability"
              << std::setw(16) << "game value" << "\n";
    std::cout << std::string(44, '-') << "\n";

    // Checkpoints at powers of 10 and a few intermediate points.
    std::vector<int> checkpoints;
    for (int p = 1; p <= iterations; p *= 10) {
        checkpoints.push_back(p);
        if (p * 5 <= iterations) checkpoints.push_back(p * 5);
    }
    checkpoints.push_back(iterations);

    int done = 0;
	for (int cp : checkpoints) {
		if (cp <= done) continue;
		int batch = cp - done;
		double gv = trainer.train(batch, root_fn);
		done = cp;

		double exploit = trainer.exploitability(root_fn);
		std::cout << std::setw(12) << done
                  << std::fixed << std::setprecision(8)
                  << std::setw(16) << exploit
                  << std::setw(16) << gv
                  << "\n";
    }
    std::cout << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
	int iterations = kDefaultIterations;
	if (argc > 1) {
		iterations = std::atoi(argv[1]);
		if (iterations <= 0)  {
			std::cerr << "Usage: solve[iterations]\n";
			return 1;
		}
	}

	std::cout << "Bluefish — Kuhn Poker CFR Solver\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    bluefish::CfrTrainer trainer;

    // Train with convergence logging.
    train_with_logging(trainer, iterations);

    // Print learned strategies.
    std::cout << "Learned average strategy:\n\n";
    print_strategy(trainer);

    // Final exploitability.
    double exploit = trainer.exploitability(bluefish::kuhn::root_states);
    std::cout << "Final exploitability: " << std::scientific << exploit << "\n";
    std::cout << "Information sets:     " << trainer.num_info_sets() << "\n";
    std::cout << "Nash game value:      -0.0556 (-1/18)\n";

    return 0;
}
