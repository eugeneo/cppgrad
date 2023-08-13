add_library(CppGrad Dimensions.h Tensor.h Tensor.cc)
add_executable(CppGradDemo main.cc)
target_link_libraries(CppGradDemo CppGrad)
target_include_directories(
  CppGrad
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)