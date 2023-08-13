#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

#include "Dimensions.h"

namespace CppGrad {

class TensorElement;

class Tensor {
public:
  Tensor(const Dimensions &dimensions)
      : data_(dimensions.element_count()), dimensions_(dimensions) {}
  Tensor(float v) : Tensor(Dimensions::Scalar()) { data_[0] = v; }

  Dimensions dimensions() const { return dimensions_; }
  Dimensions dimensions(const Index &index) const {
    assert(dimensions_.IndexInBounds(index));
    Dimensions result = dimensions_;
    for (size_t i = 0; i < index.size(); ++i) {
      result = result.nested();
    }
    if (result.element_count() == 0) {
      return Dimensions::Scalar();
    }
    return result;
  }

  TensorElement operator[](size_t index);
  TensorElement operator[](Index index);
  const TensorElement operator[](size_t index) const;

  Tensor &operator=(float value) {
    assert(dimensions_.element_count() == 1);
    data_[0] = value;
    return *this;
  }

  bool operator==(const Tensor &other) const {
    return data_ == other.data_ && dimensions_ == other.dimensions_;
  }

  std::span<const float> raw_data() const { return data_; }
  std::span<const float> raw_data(Index index) const;
  void Set(const Index &index, float value);

  template <typename T>
  static Tensor Generate(const Dimensions &dimensions, const T &generator) {
    Tensor tensor(dimensions);
    for (size_t i = 0; i < dimensions.element_count(); ++i) {
      tensor.data_[i] =
          generator(dimensions.index_from_flat(i), dimensions);
    }
    return tensor;
  }

  friend void PrintTo(const Tensor &tensor, std::ostream *os);

private:
  std::vector<float> data_;
  Dimensions dimensions_;
};

class TensorElement {
public:
  TensorElement(Tensor *tensor, std::span<const size_t> index)
      : tensor_(tensor), index_(index.begin(), index.end()) {}
  TensorElement(const TensorElement &other) = default;
  TensorElement(TensorElement &&other) = default;
  TensorElement operator[](size_t index);
  TensorElement &operator=(float value);
  Dimensions dimensions() const { return tensor_->dimensions(index_); }
  std::span<const float> raw_data() const;

  friend void PrintTo(const TensorElement &tensor, std::ostream *os);

private:
  Tensor *tensor_;
  std::vector<size_t> index_;
};

bool operator==(const TensorElement &a, const Tensor &b);

} // namespace CppGrad

#endif /* SRC_TENSOR_H_ */