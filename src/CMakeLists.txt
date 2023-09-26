add_library(CppGrad Tensor.h Tensor.cc Units.h Units.cc)
add_executable(CppGradDemo main.cc)
target_link_libraries(CppGradDemo CppGrad)
target_include_directories(
  CppGrad
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)