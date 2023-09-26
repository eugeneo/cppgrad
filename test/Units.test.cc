#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include "Tensor.h"
#include "Units.h"

using CppGrad::Dimensions;
using CppGrad::Index;

TEST(DimensionsTest, flat_index) {
  Dimensions dims({2, 3, 4, 5});
  ASSERT_EQ(dims.flat_index(Index(dims, {0, 0, 0, 0})), 0);
  ASSERT_EQ(dims.flat_index(Index(dims, {0, 0, 0, 1})), 1);
  ASSERT_EQ(dims.flat_index(Index(dims, {0, 0, 1, 0})), 5);
  ASSERT_EQ(dims.flat_index(Index(dims, {0, 1, 0, 0})), 20);
  ASSERT_EQ(dims.flat_index(Index(dims, {1, 0, 0, 0})), 60);
}

TEST(DimensionsTest, element_count) {
  ASSERT_EQ(Dimensions({2, 3, 4, 5}).element_count(), 120);
  ASSERT_EQ(Dimensions({2, 3}).element_count(), 6);
  ASSERT_EQ(Dimensions({2}).element_count(), 2);
  ASSERT_EQ(Dimensions({}).element_count(), 0);
}

TEST(DimensionsTest, from_flat_index) {
  Dimensions dims({2, 3, 4, 5});
  EXPECT_EQ(dims.index_from_flat(0), Index(dims, {0, 0, 0, 0}));
  EXPECT_EQ(dims.index_from_flat(1), Index(dims, {0, 0, 0, 1}));
  EXPECT_EQ(dims.index_from_flat(5), Index(dims, {0, 0, 1, 0}));
  EXPECT_EQ(dims.index_from_flat(20), Index(dims, {0, 1, 0, 0}));
  EXPECT_EQ(dims.index_from_flat(60), Index(dims, {1, 0, 0, 0}));
  EXPECT_EQ(dims.index_from_flat(dims.element_count() - 1),
            Index(dims, {1, 2, 3, 4}));
}

TEST(DimensionsTest, stride) {
  Dimensions dims({2, 3, 4, 5});
  EXPECT_EQ(dims.stride(0), 60);
  EXPECT_EQ(dims.stride(1), 20);
  EXPECT_EQ(dims.stride(2), 5);
  EXPECT_EQ(dims.stride(3), 1);
  dims = Dimensions::Scalar();
  EXPECT_EQ(dims.stride(0), 1);
}

TEST(IndexTest, flat_index) {
  Dimensions dims({2, 3, 4, 5});
  Index ind(dims, {});
  EXPECT_EQ(ind.flat_index(), 0);
  EXPECT_EQ(ind.order(), 0);
  ind = Index(dims, {1});
  EXPECT_EQ(ind.flat_index(), 60);
  EXPECT_EQ(ind.order(), 1);
  ind = Index(dims, {1, 2});
  EXPECT_EQ(ind.flat_index(), 100);
  EXPECT_EQ(ind.order(), 2);
  ind = Index(dims, {1, 2, 3});
  EXPECT_EQ(ind.flat_index(), 115);
  EXPECT_EQ(ind.order(), 3);
  ind = Index(dims, {1, 2, 3, 4});
  EXPECT_EQ(ind.flat_index(), 119);
  EXPECT_EQ(ind.order(), 4);
}

TEST(IndexTest, subelement) {
  Dimensions dims({2, 3, 4, 5});
  Index ind(dims, {});
  ind = ind.subelement(dims, 1);
  EXPECT_EQ(ind, Index(dims, {1}));
  ind = ind.subelement(dims, 2);
  EXPECT_EQ(ind, Index(dims, {1, 2}));
  ind = ind.subelement(dims, 3);
  EXPECT_EQ(ind, Index(dims, {1, 2, 3}));
  ind = ind.subelement(dims, 4);
  EXPECT_EQ(ind, Index(dims, {1, 2, 3, 4}));
}