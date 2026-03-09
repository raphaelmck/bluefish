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
    src/cfr.cpp
)
target_include_directories(bluefish_core PUBLIC include)

# Solver CLI
add_executable(solve
    app/solve.cpp
)
target_link_libraries(solve PRIVATE bluefish_core)

# Tests
enable_testing()

include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

add_executable(tests
    tests/test_kuhn_cfr.cpp
)
target_link_libraries(tests PRIVATE bluefish_core doctest::doctest)
add_test(NAME bluefish_tests COMMAND tests)
