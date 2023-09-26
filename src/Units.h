#ifndef SRC_DIMENSIONS_H_
#define SRC_DIMENSIONS_H_

#include <cassert>
#include <numeric>
#include <ostream>
#include <span>
#include <vector>

namespace CppGrad {
namespace {

constexpr size_t element_count_fn(std::span<const size_t> dimensions) {
  if (dimensions.empty()) {
    return 0;
  }
  return std::accumulate(dimensions.begin(), dimensions.end(), 1,
                         std::multiplies<size_t>());
}

} // namespace

class Dimensions;

class Index {
public:
  Index(const Dimensions &dimensions, std::initializer_list<size_t> index)
      : Index(dimensions, std::span<const size_t>(index.begin(), index.end())) {
  }
  Index(const Dimensions &dimensions, std::span<const size_t> index);
  Index(size_t index, size_t order) : index_(index), order_(order) {}

  size_t flat_index() const { return index_; }
  size_t order() const { return order_; }
  Index subelement(const Dimensions &dimensions, size_t index) const;
  bool operator==(const Index &other) const = default;

  friend void PrintTo(const Index &index, std::ostream *os) {
    *os << "(" << index.index_ << ", " << index.order_ << ")";
  }

private:
  size_t index_;
  size_t order_;
};

class Dimensions {
public:
  static Dimensions Scalar() { return Dimensions({1}); }

  Dimensions(std::initializer_list<size_t> dimensions)
      : Dimensions(
            std::span<const size_t>(dimensions.begin(), dimensions.end())) {}
  Dimensions(const Dimensions &other) = default;
  Dimensions(Dimensions &&other) = default;
  ~Dimensions() = default;

  size_t element_count() const { return max_; }

  size_t flat_index(const Index &index) const {
    assert(index.flat_index() <= max_);
    return index.flat_index();
  }

  Index index_from_flat(size_t flat_index) const {
    size_t remaining = flat_index;
    std::vector<size_t> result(dimensions_.size());
    size_t index = dimensions_.size();
    for (auto dim = dimensions_.rbegin(); dim < dimensions_.rend(); ++dim) {
      result[--index] = remaining % (*dim);
      remaining /= *dim;
    }
    return Index(*this, result);
  }

  size_t size() const { return dimensions_.size(); }

  size_t stride(size_t order) const {
    assert(order < dimensions_.size());
    size_t result = 1;
    for (size_t i = order + 1; i < dimensions_.size(); ++i) {
      result *= dimensions_[i];
    }
    return result;
  }

  bool operator==(const Dimensions &other) const {
    return dimensions_ == other.dimensions_;
  }

  Dimensions &operator=(const Dimensions &other) {
    dimensions_ = other.dimensions_;
    return *this;
  }

  Dimensions &operator=(Dimensions &&other) {
    dimensions_ = std::move(other.dimensions_);
    return *this;
  }

  bool IndexInBounds(const Index &index) const {
    return index.flat_index() < max_;
  }

  Dimensions nested() const {
    assert(dimensions_.size() > 0);
    return Dimensions(std::span(dimensions_.begin() + 1, dimensions_.end()));
  }

  size_t subdim(size_t index) const {
    assert(index < dimensions_.size());
    return dimensions_[index];
  }

  void Print(std::ostream *os) const {
    bool first = true;
    *os << "(";
    for (size_t dim : dimensions_) {
      if (!first) {
        *os << ", ";
      }
      *os << dim;
      first = false;
    }
    *os << ")";
  }

private:
  Dimensions(std::span<const size_t> dimensions)
      : dimensions_(dimensions.begin(), dimensions.end()),
        max_(element_count_fn(dimensions)) {}

  std::vector<size_t> dimensions_;
  size_t max_;
};

} // namespace CppGrad

#endif // SRC_DIMENSIONS_H_