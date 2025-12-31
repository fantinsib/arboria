
##################### TESTS CMAKE ######################

add_executable(arboria_tests
    test_gini.cpp
    test_entropy.cpp
    test_cart_threshold.cpp
    test_dataset.cpp
    test_count.cpp
)

target_link_libraries(arboria_tests
    PRIVATE Catch2::Catch2WithMain
    PRIVATE arboria_lib
)

target_include_directories(arboria_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)

add_test(
    NAME arboria_tests
    COMMAND arboria_tests
)