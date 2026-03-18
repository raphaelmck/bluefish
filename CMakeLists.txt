cmake_minimum_required(VERSION 3.20)
project(bluefish VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)

# Core library
add_library(bluefish_core
	src/solver.cpp
    src/cfr.cpp
    src/cfr_plus.cpp
    src/mccfr.cpp
)
target_include_directories(bluefish_core PUBLIC include)

# Solver CLI
add_executable(solve app/solve.cpp)
target_link_libraries(solve PRIVATE bluefish_core)

add_executable(experiment app/experiment.cpp)
target_link_libraries(experiment PRIVATE bluefish_core)

add_executable(bench app/bench.cpp)
target_link_libraries(bench PRIVATE bluefish_core)

# Tests
enable_testing()

include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

add_executable(test_kuhn tests/test_kuhn_cfr.cpp)
target_link_libraries(test_kuhn PRIVATE bluefish_core doctest::doctest)
add_test(NAME kuhn_tests COMMAND test_kuhn)
 
add_executable(test_leduc tests/test_leduc.cpp)
target_link_libraries(test_leduc PRIVATE bluefish_core doctest::doctest)
add_test(NAME leduc_tests COMMAND test_leduc)
 
add_executable(test_algorithms tests/test_algorithms.cpp)
target_link_libraries(test_algorithms PRIVATE bluefish_core doctest::doctest)
add_test(NAME algorithm_tests COMMAND test_algorithms)
