#include <gtest/gtest.h>

#include "Tensor.h"
#include "Units.h"

using CppGrad::Dimensions;
using CppGrad::Index;
using CppGrad::Tensor;

TEST(TensorTest, AllDimensions) {
  Tensor a({2, 3, 4, 5});
  ASSERT_EQ(a.dimensions(), Dimensions({2, 3, 4, 5}));
  ASSERT_EQ(a.dimensions(Index(a.dimensions(), {1, 1})), Dimensions({4, 5}));
}

TEST(TensorTest, Tensor1d) {
  Tensor a({1});
  a = 10;
  ASSERT_EQ(a, Tensor(10));
  ASSERT_EQ(a.dimensions(), Dimensions::Scalar());
}

TEST(TensorTest, Tensor4d) {
  Tensor a = Tensor::Generate({2, 3, 4, 5}, [](auto index, auto dims) {
    return dims.flat_index(index);
  });
  auto el = a[1][1][1][1];
  EXPECT_EQ(el, Tensor(86));
  el = 100;
  EXPECT_EQ(a[1][1][1][1], Tensor(100));
}

TEST(TensorTest, Generator) {
  auto tensor = Tensor::Generate({2, 3}, [](auto index, auto dims) {
    return dims.flat_index(index) * 7;
  });
  size_t i = 0;
  for (float value : tensor.raw_data()) {
    EXPECT_EQ(value, i++ * 7);
  }
}