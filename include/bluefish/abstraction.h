// bluefish/abstraction.h - Card abstraction for imperfect-information games
//
// An abstraction merges information sets: multiple exact info sets
// share a single strategy. This reduces memory and training cost
// at the expense of strategic precision.
//
// The abstraction is a pure integer mapping: exact_is_id → abstract_is_id.
// apply_abstraction() creates a new FlatGame with remapped IDs.
// lifted_flat_exploitability() evaluates the abstract strategy on the
// exact game tree — the correct way to measure abstraction quality.
//
// Leduc built-in abstractions:
//   exact       — identity, 288 info sets (baseline)
//   jq_merge    — merge J and Q ranks everywhere, ~132 info sets
//   pair_only   — postflop: only distinguish paired vs not-paired, ~198

#pragma once

#include "flat_game.h"

namespace bluefish {

// - Abstraction mapping -

struct Abstraction {
    std::vector<int> mapping;        // mapping[exact_is] = abstract_is
    int num_abstract = 0;
    std::string name;
    std::vector<std::string> abstract_keys;  // display keys per abstract IS
};

Abstraction make_abstraction(const FlatGame& exact_game,
                             const std::string& name,
                             std::function<std::string(const std::string&)> remap);

// - Apply / lift -

FlatGame apply_abstraction(const FlatGame& exact, const Abstraction& abs);

double lifted_flat_exploitability(const FlatGame& exact_game,
                                  const RegretTable& abstract_table,
                                  const Abstraction& abs);

// - Leduc abstractions -
namespace leduc_abs {

// Identity: every exact info set is its own abstract info set.
Abstraction exact(const FlatGame& game);

Abstraction jq_merge(const FlatGame& game);

Abstraction pair_only(const FlatGame& game);

} // namespace leduc_abs

} // namespace bluefish
