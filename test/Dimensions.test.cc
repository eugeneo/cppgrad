#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include "Dimensions.h"
#include "Tensor.h"

using CppGrad::Dimensions;
using CppGrad::Index;

TEST(DimensionsTest, flat_index) {
  Dimensions dims({2, 3, 4, 5});
  ASSERT_EQ(dims.flat_index(std::array<size_t, 4>{0, 0, 0, 0}), 0);
  ASSERT_EQ(dims.flat_index(std::array<size_t, 4>{0, 0, 0, 1}), 1);
  ASSERT_EQ(dims.flat_index(std::array<size_t, 4>{0, 0, 1, 0}), 5);
  ASSERT_EQ(dims.flat_index(std::array<size_t, 4>{0, 1, 0, 0}), 20);
  ASSERT_EQ(dims.flat_index(std::array<size_t, 4>{1, 0, 0, 0}), 60);
}

TEST(DimensionsTest, element_count) {
  ASSERT_EQ(Dimensions({2, 3, 4, 5}).element_count(), 120);
  ASSERT_EQ(Dimensions({2, 3}).element_count(), 6);
  ASSERT_EQ(Dimensions({2}).element_count(), 2);
  ASSERT_EQ(Dimensions({}).element_count(), 0);
}

TEST(DimensionsTest, from_flat_index) {
  Dimensions dims({2, 3, 4, 5});
  EXPECT_THAT(dims.index_from_flat(0), ::testing::ElementsAre(0, 0, 0, 0));
  EXPECT_THAT(dims.index_from_flat(1), ::testing::ElementsAre(0, 0, 0, 1));
  EXPECT_THAT(dims.index_from_flat(5), ::testing::ElementsAre(0, 0, 1, 0));
  EXPECT_THAT(dims.index_from_flat(20), ::testing::ElementsAre(0, 1, 0, 0));
  EXPECT_THAT(dims.index_from_flat(60), ::testing::ElementsAre(1, 0, 0, 0));
  EXPECT_THAT(dims.index_from_flat(dims.element_count() - 1),
              ::testing::ElementsAre(1, 2, 3, 4));
}