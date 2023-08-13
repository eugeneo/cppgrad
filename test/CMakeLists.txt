include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/455fcb7773dedc70ab489109fb12d8abc7fd59b6.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(tensor_test Tensor.test.cc Dimensions.test.cc)

target_include_directories(
  tensor_test
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(
  tensor_test
  GTest::gtest_main
  GTest::gmock
  CppGrad
)

include(GoogleTest)
gtest_discover_tests(tensor_test)