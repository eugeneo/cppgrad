#include "Units.h"

#include <iostream>

namespace CppGrad {

Index::Index(const Dimensions &dimensions, std::span<const size_t> index)
    : index_(index.empty() ? 0 : *index.begin()), order_(index.size()) {
  assert(index.size() <= dimensions.size());
  std::vector<size_t> long_ind(index.begin(), index.end());
  long_ind.resize(dimensions.size(), 0);
  for (size_t i = 1; i < dimensions.size(); ++i) {
    index_ = (i < index.size() ? index[i] : 0) + index_ * dimensions.subdim(i);
  }
}

Index Index::subelement(const Dimensions &dimensions, size_t index) const {
  assert(dimensions.size() > order_);
  assert(index < dimensions.subdim(order_));
  return Index(index_ + index * dimensions.stride(order_), order_ + 1);
}
} // namespace CppGrad