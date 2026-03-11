// solve.cpp - Train vanilla CFR on kuhn poker and inspect results

// Usage:
//	 ./solve kuhn  [iterations] [--json output.json]
//	 ./solve leduc [iterations] [--json output.json]

#include "bluefish/cfr.h"
#include "bluefish/kuhn.h"
#include "bluefish/leduc.h"

#include <map>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kDefaultIterations = 100'000;

// - Strategy display -

struct GameActions {
	std::vector<std::string> names;
};

GameActions kuhn_actions() { return {{"pass", "bet"}}; }
GameActions leduc_actions() { return {{"fold", "call", "raise"}}; }

// Display the average stratey at every information set, grouped by player
void print_strategy(const bluefish::CfrTrainer& trainer, const GameActions& ga) {
	std::map<std::string, std::vector<double>> strategies;
	for (auto& [key, node] : trainer.info_map()) {
		strategies[key] = node.average_strategy();
	}

	std::size_t max_key = 8;
	for (auto& [key, _] : strategies) {
		max_key = std::max(max_key, key.size());
	}

	std::cout << std::setw(static_cast<int>(max_key) + 2) << "info set";
	for (auto& name : ga.names) 
		std::cout << std::setw(10) << name;
	std::cout << "\n";
	std::cout << std::string(max_key + 2 + ga.names.size() * 10, '-') << "\n";

	for (auto& [key, avg] : strategies) {
		std::cout << std::setw(static_cast<int>(max_key) + 2) << key;
		for (auto v : avg)
			std::cout << std::fixed << std::setprecision(4)
					  << std::setw(10) << v;	
		std::cout << "\n";
	}

	std::cout << "\n";
}

// - Training with convergence logging -

void train_with_logging(bluefish::CfrTrainer& trainer,
                        bluefish::CfrTrainer::RootFn root_fn,
                        int iterations) {
    std::cout << std::setw(12) << "iteration"
              << std::setw(16) << "exploitability"
              << std::setw(16) << "game value"
              << std::setw(12) << "time (s)"
              << std::setw(12) << "info sets"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    // Logarithmic checkpoints.
    std::vector<int> checkpoints;
    for (int p = 1; p <= iterations; p *= 10) {
        checkpoints.push_back(p);
        if (p * 5 <= iterations) checkpoints.push_back(p * 5);
    }
    checkpoints.push_back(iterations);

    auto t0 = Clock::now();
    int done = 0;

    for (int cp : checkpoints) {
        if (cp <= done) continue;
        int batch = cp - done;
        double gv = trainer.train(batch, root_fn);
        done = cp;

        double elapsed = std::chrono::duration<double>(
            Clock::now() - t0).count();
        double exploit = trainer.exploitability(root_fn);

        std::cout << std::setw(12) << done
                  << std::fixed << std::setprecision(8)
                  << std::setw(16) << exploit
                  << std::setw(16) << gv
                  << std::setprecision(3)
                  << std::setw(12) << elapsed
                  << std::setw(12) << trainer.num_info_sets()
                  << "\n";
    }
    std::cout << "\n";
}

void usage() {
    std::cerr << "Usage: solve <kuhn|leduc> [iterations] [--json file.json]\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(); return 1; }

    std::string game_name = argv[1];
    int iterations = kDefaultIterations;
    std::string json_path;

    // Parse remaining args.
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) {
            json_path = argv[++i];
        } else {
            int n = std::atoi(arg.c_str());
            if (n > 0) iterations = n;
        }
    }

    // Select game.
    bluefish::CfrTrainer::RootFn root_fn;
    GameActions ga;
    std::string nash_value_str;

    if (game_name == "kuhn") {
        root_fn = bluefish::kuhn::root_states;
        ga = kuhn_actions();
        nash_value_str = "-0.0556 (-1/18)";
    } else if (game_name == "leduc") {
        root_fn = bluefish::leduc::root_states;
        ga = leduc_actions();
        nash_value_str = "~-0.0856";
    } else {
        usage();
        return 1;
    }

    std::cout << "Bluefish — " << game_name << " poker CFR solver\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    bluefish::CfrTrainer trainer;
    auto t0 = Clock::now();

    train_with_logging(trainer, root_fn, iterations);

    double total_sec = std::chrono::duration<double>(
        Clock::now() - t0).count();

    // Print learned strategies.
    std::cout << "Learned average strategy:\n\n";
    print_strategy(trainer, ga);

    // Summary.
    double exploit = trainer.exploitability(root_fn);
    std::cout << "Final exploitability: " << std::scientific << exploit << "\n";
    std::cout << "Information sets:     " << trainer.num_info_sets() << "\n";
    std::cout << "Nash game value:      " << nash_value_str << "\n";
    std::cout << "Total time:           " << std::fixed << std::setprecision(3)
              << total_sec << " s\n";
    std::cout << "Iters/sec:            " << std::fixed << std::setprecision(0)
              << static_cast<double>(iterations) / total_sec << "\n";

    // JSON export.
    if (!json_path.empty()) {
        std::ofstream out(json_path);
        out << trainer.serialize_json();
        std::cout << "\nStrategy written to " << json_path << "\n";
    }

    return 0;
}
