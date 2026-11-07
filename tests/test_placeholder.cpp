#include <catch2/catch_test_macros.hpp>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    return CatchShim::run_all_tests();
}

