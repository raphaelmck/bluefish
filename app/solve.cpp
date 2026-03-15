// solve.cpp — Train CFR-family algorithms on Kuhn or Leduc poker
//
// Usage:
//   ./solve <game> [algorithm] [iterations] [--json file] [--csv file] [--seed N]
//
// Games: kuhn, leduc
// Algorithms: cfr (default), cfr+, mccfr

#include "bluefish/cfr.h"
#include "bluefish/cfr_plus.h"
#include "bluefish/mccfr.h"
#include "bluefish/kuhn.h"
#include "bluefish/leduc.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

namespace {

using Clock = std::chrono::steady_clock;

// ─ Strategy display ─

void print_strategy(const bluefish::Solver& solver,
                    const std::vector<std::string>& action_names) {
    std::map<std::string, std::vector<double>> strategies;
    for (auto& [key, node] : solver.info_map())
        strategies[key] = node.average_strategy();

    std::size_t max_key = 8;
    for (auto& [key, _] : strategies)
        max_key = std::max(max_key, key.size());

    std::cout << std::setw(static_cast<int>(max_key) + 2) << "info set";
    for (auto& name : action_names)
        std::cout << std::setw(10) << name;
    std::cout << "\n";
    std::cout << std::string(max_key + 2 + action_names.size() * 10, '-')
              << "\n";

    for (auto& [key, avg] : strategies) {
        std::cout << std::setw(static_cast<int>(max_key) + 2) << key;
        for (auto v : avg)
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(10) << v;
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ─ Log-spaced checkpoints ─

std::vector<int> log_checkpoints(int total) {
    std::vector<int> cp;
    for (int p = 1; p <= total; p *= 10) {
        cp.push_back(p);
        if (p * 5 <= total) cp.push_back(p * 5);
    }
    cp.push_back(total);
    return cp;
}

// ─ Training with convergence logging ─

void train_with_logging(bluefish::Solver& solver,
                        bluefish::Solver::RootFn root_fn,
                        int iterations,
                        std::ofstream* csv_out) {
    // Console header
    std::cout << std::setw(12) << "iteration"
              << std::setw(16) << "exploitability"
              << std::setw(16) << "game value"
              << std::setw(12) << "time (s)"
              << std::setw(14) << "nodes/iter"
              << std::setw(12) << "info sets"
              << "\n";
    std::cout << std::string(82, '-') << "\n";

    // CSV header
    if (csv_out) {
        *csv_out << "iteration,exploitability,game_value,wall_sec,"
                 << "nodes_touched,terminal_visits,info_set_visits,"
                 << "chance_visits,info_sets,iters_per_sec,nodes_per_sec\n";
    }

    auto checkpoints = log_checkpoints(iterations);
    auto t0 = Clock::now();
    int done = 0;
    int64_t prev_nodes = 0;

    for (int cp : checkpoints) {
        if (cp <= done) continue;
        int batch = cp - done;
        double gv = solver.train(batch, root_fn);
        done = cp;

        double elapsed = std::chrono::duration<double>(
            Clock::now() - t0).count();
        double exploit = solver.exploitability(root_fn);

        auto& st = solver.stats();
        int64_t batch_nodes = st.nodes_touched - prev_nodes;
        double nodes_per_iter = (batch > 0)
            ? static_cast<double>(batch_nodes) / static_cast<double>(batch)
            : 0.0;
        prev_nodes = st.nodes_touched;

        // Console output
        std::cout << std::setw(12) << done
                  << std::fixed << std::setprecision(8)
                  << std::setw(16) << exploit
                  << std::setw(16) << gv
                  << std::setprecision(3)
                  << std::setw(12) << elapsed
                  << std::setprecision(0)
                  << std::setw(14) << nodes_per_iter
                  << std::setw(12) << solver.num_info_sets()
                  << "\n";

        // CSV output
        if (csv_out) {
            double ips = (elapsed > 0)
                ? static_cast<double>(done) / elapsed : 0.0;
            double nps = (elapsed > 0)
                ? static_cast<double>(st.nodes_touched) / elapsed : 0.0;

            *csv_out << done << ","
                     << std::scientific << std::setprecision(10)
                     << exploit << ","
                     << gv << ","
                     << std::fixed << std::setprecision(6) << elapsed << ","
                     << st.nodes_touched << ","
                     << st.terminal_visits << ","
                     << st.info_set_visits << ","
                     << st.chance_visits << ","
                     << solver.num_info_sets() << ","
                     << std::setprecision(1) << ips << ","
                     << nps << "\n";
        }
    }
    std::cout << "\n";
}

void usage() {
    std::cerr
        << "Usage: solve <game> [algorithm] [iterations] [options]\n"
        << "\n"
        << "Games:      kuhn, leduc\n"
        << "Algorithms: cfr (default), cfr+, mccfr\n"
        << "Options:    --json <file>   export strategy as JSON\n"
        << "            --csv <file>    export convergence data as CSV\n"
        << "            --seed <N>      RNG seed for MCCFR (default: 42)\n"
        << "\n"
        << "Examples:\n"
        << "  solve kuhn\n"
        << "  solve leduc cfr+ 50000\n"
        << "  solve leduc mccfr 5000000 --csv data.csv --seed 123\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(); return 1; }

    std::string game_name = argv[1];
    std::string algo_name = "cfr";
    int iterations = 100'000;
    std::string json_path, csv_path;
    uint64_t seed = 42;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) {
            json_path = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (arg == "cfr" || arg == "cfr+" || arg == "mccfr") {
            algo_name = arg;
        } else {
            int n = std::atoi(arg.c_str());
            if (n > 0) iterations = n;
        }
    }

    // Select game
    bluefish::Solver::RootFn root_fn;
    std::vector<std::string> action_names;
    std::string nash_str;

    if (game_name == "kuhn") {
        root_fn = bluefish::kuhn::root_states;
        action_names = {"pass", "bet"};
        nash_str = "-0.0556 (-1/18)";
    } else if (game_name == "leduc") {
        root_fn = bluefish::leduc::root_states;
        action_names = {"fold", "call", "raise"};
        nash_str = "~-0.0856";
    } else {
        usage(); return 1;
    }

    // Select algorithm
    std::unique_ptr<bluefish::Solver> solver;
    if (algo_name == "cfr")
        solver = std::make_unique<bluefish::CfrSolver>();
    else if (algo_name == "cfr+")
        solver = std::make_unique<bluefish::CfrPlusSolver>();
    else if (algo_name == "mccfr")
        solver = std::make_unique<bluefish::MccfrSolver>(seed);
    else { usage(); return 1; }

    std::cout << "Bluefish — " << game_name << " poker / "
              << solver->name() << "\n";
    std::cout << "Iterations: " << iterations;
    if (algo_name == "mccfr")
        std::cout << " (seed=" << seed << ")";
    std::cout << "\n\n";

    // Open CSV file if requested
    std::ofstream csv_file;
    std::ofstream* csv_ptr = nullptr;
    if (!csv_path.empty()) {
        csv_file.open(csv_path);
        csv_ptr = &csv_file;
    }

    auto t0 = Clock::now();
    train_with_logging(*solver, root_fn, iterations, csv_ptr);
    double total_sec = std::chrono::duration<double>(
        Clock::now() - t0).count();

    // Strategy
    std::cout << "Learned average strategy:\n\n";
    print_strategy(*solver, action_names);

    // Summary
    double exploit = solver->exploitability(root_fn);
    auto& st = solver->stats();
    double ips = static_cast<double>(iterations) / total_sec;
    double nps = static_cast<double>(st.nodes_touched) / total_sec;

    std::cout << "Final exploitability: " << std::scientific << exploit << "\n";
    std::cout << "Information sets:     " << solver->num_info_sets() << "\n";
    std::cout << "Nash game value:      " << nash_str << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total time:           " << total_sec << " s\n";
    std::cout << std::setprecision(0);
    std::cout << "Iters/sec:            " << ips << "\n";
    std::cout << "Nodes/sec:            " << nps << "\n";
    std::cout << "Nodes/iter (avg):     "
              << static_cast<double>(st.nodes_touched)
                 / static_cast<double>(iterations) << "\n";
    std::cout << "Total nodes:          " << st.nodes_touched << "\n";
    std::cout << "Terminal visits:      " << st.terminal_visits << "\n";
    std::cout << "Info set visits:      " << st.info_set_visits << "\n";
    std::cout << "Chance visits:        " << st.chance_visits << "\n";

    // Validation
    std::string err = solver->validate();
    if (!err.empty()) {
        std::cerr << "VALIDATION FAILED: " << err << "\n";
        return 2;
    }

    // JSON export
    if (!json_path.empty()) {
        std::ofstream out(json_path);
        out << solver->serialize_json();
        std::cout << "\nStrategy written to " << json_path << "\n";
    }
    if (!csv_path.empty()) {
        std::cout << "Convergence data written to " << csv_path << "\n";
    }

    return 0;
}
