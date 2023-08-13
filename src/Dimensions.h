#ifndef SRC_DIMENSIONS_H_
#define SRC_DIMENSIONS_H_

#include <numeric>
#include <span>

namespace CppGrad {

using Index = std::span<const size_t>;

class Dimensions {
public:
  Dimensions(std::initializer_list<size_t> dimensions)
      : dimensions_(dimensions.begin(), dimensions.end()) {}
  Dimensions(const Dimensions &other) = default;
  Dimensions(Dimensions &&other) = default;
  ~Dimensions() = default;

  size_t element_count() const {
    if (dimensions_.empty()) {
      return 0;
    }
    return std::accumulate(dimensions_.begin(), dimensions_.end(), 1,
                           std::multiplies<size_t>());
  }

  size_t flat_index(const Index &index) const {
    assert(index.size() <= dimensions_.size());
    size_t ind = 0;
    for (size_t i = 0; i < index.size(); ++i) {
      assert(index[i] < dimensions_[i]);
      ind = ind * dimensions_[i] + index[i];
    }
    return ind;
  }

  std::vector<size_t> index_from_flat(size_t flat_index) const {
    size_t remaining = flat_index;
    std::vector<size_t> result(dimensions_.size());
    size_t index = dimensions_.size();
    for (auto dim = dimensions_.rbegin(); dim < dimensions_.rend(); ++dim) {
      result[--index] = remaining % (*dim);
      remaining /= *dim;
    }
    return result;
  }

  size_t size() const { return dimensions_.size(); }

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
    if (index.size() > dimensions_.size()) {
      return false;
    }
    for (size_t i = 0; i < index.size(); ++i) {
      if (index[i] >= dimensions_[i]) {
        return false;
      }
    }
    return true;
  }

  Dimensions nested() const {
    assert(dimensions_.size() > 0);
    return Dimensions(std::span(dimensions_.begin() + 1, dimensions_.end()));
  }

  static Dimensions Scalar() { return Dimensions({1}); }

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
      : dimensions_(dimensions.begin(), dimensions.end()) {}
  std::vector<size_t> dimensions_;
};

} // namespace CppGrad

#endif // SRC_DIMENSIONS_H_