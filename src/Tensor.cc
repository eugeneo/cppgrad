#include <Tensor.h>

#include <ostream>

namespace CppGrad {
namespace {
template <typename T> bool SameSpans(std::span<T> a, std::span<T> b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

void Print(const Dimensions &dimensions, std::string_view tag,
           std::span<const float> data, std::ostream *os) {
  *os << tag;
  bool first = true;
  dimensions.Print(os);
  first = true;
  *os << ")"
      << "{";
  for (float value : data) {
    if (!first) {
      *os << ", ";
    }
    *os << value;
    first = false;
  }
  *os << "}";
}

class SourceLocation {
public:
  constexpr SourceLocation(std::string_view file = __FILE__,
                           size_t line = __LINE__)
      : file_(file), line_(line) {}
  std::string_view file() const { return file_; }
  size_t line() const { return line_; }

private:
  std::string_view file_;
  size_t line_;
};

std::ostream &operator<<(std::ostream &os, const SourceLocation &loc) {
  return os << loc.file() << ":" << loc.line();
}

void log_data(std::span<const float> data,
              SourceLocation location = SourceLocation()) {
  std::cerr << "[" << location << "] Length: " << data.size() << " (";
  bool first = true;
  for (float value : data) {
    if (!first) {
      std::cerr << ", ";
    }
    std::cerr << value;
    first = false;
  }
  std::cerr << ")" << std::endl;
}

} // namespace

TensorElement TensorElement::operator[](size_t index) {
  return tensor_->operator[](index_.subelement(dimensions(), index));
}

TensorElement &TensorElement::operator=(float value) {
  tensor_->Set(index_, value);
  return *this;
}

std::span<const float> TensorElement::raw_data() const {
  std::span<const float> data = tensor_->raw_data();
  return data.subspan(tensor_->dimensions().flat_index(index_),
                             dimensions().element_count());
}

void PrintTo(const TensorElement &tensor, std::ostream *os) {
  Print(tensor.dimensions(), "TensorElement", tensor.raw_data(), os);
}

TensorElement Tensor::operator[](size_t index) {
  return TensorElement(this, Index(index, dimensions().size()));
}

TensorElement Tensor::operator[](Index index) {
  return TensorElement(this, index);
}

void Tensor::Set(const Index &index, float value) {
  data_[dimensions_.flat_index(index)] = value;
}

bool operator==(const TensorElement &a, const Tensor &b) {
  return a.dimensions() == b.dimensions() &&
         SameSpans(a.raw_data(), b.raw_data());
}

void PrintTo(const Tensor &tensor, std::ostream *os) {
  Print(tensor.dimensions(), "Tensor", tensor.raw_data(), os);
}

} // namespace CppGrad