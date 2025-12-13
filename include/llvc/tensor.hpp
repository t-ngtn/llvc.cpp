#ifndef LLVC_TENSOR_HPP
#define LLVC_TENSOR_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <initializer_list>

namespace llvc {

// Simple 3D tensor class for FPGA-friendly implementation
// Shape convention: [batch, channels, length] for 3D
//                   [batch, length] for 2D
//                   [length] for 1D
class Tensor {
public:
    Tensor() : dims_{0, 0, 0}, size_(0) {}

    // 1D constructor
    explicit Tensor(size_t d0)
        : dims_{d0, 1, 1}, size_(d0), data_(d0, 0.0f) {}

    // 2D constructor
    Tensor(size_t d0, size_t d1)
        : dims_{d0, d1, 1}, size_(d0 * d1), data_(d0 * d1, 0.0f) {}

    // 3D constructor
    Tensor(size_t d0, size_t d1, size_t d2)
        : dims_{d0, d1, d2}, size_(d0 * d1 * d2), data_(d0 * d1 * d2, 0.0f) {}

    // 4D constructor (for decoder buffer)
    Tensor(size_t d0, size_t d1, size_t d2, size_t d3)
        : dims_{d0, d1, d2, d3}, size_(d0 * d1 * d2 * d3), data_(d0 * d1 * d2 * d3, 0.0f) {}

    // Constructor from data
    Tensor(const std::vector<size_t>& shape, const float* data);

    // Copy and move
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Accessors
    size_t dim(size_t i) const { return dims_[i]; }
    size_t ndim() const {
        if (dims_[3] > 1) return 4;
        if (dims_[2] > 1) return 3;
        if (dims_[1] > 1) return 2;
        return 1;
    }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Element access - 1D
    float& operator()(size_t i) { return data_[i]; }
    const float& operator()(size_t i) const { return data_[i]; }

    // Element access - 2D
    float& operator()(size_t i, size_t j) {
        return data_[i * dims_[1] + j];
    }
    const float& operator()(size_t i, size_t j) const {
        return data_[i * dims_[1] + j];
    }

    // Element access - 3D [batch, channel, time]
    float& operator()(size_t b, size_t c, size_t t) {
        return data_[(b * dims_[1] + c) * dims_[2] + t];
    }
    const float& operator()(size_t b, size_t c, size_t t) const {
        return data_[(b * dims_[1] + c) * dims_[2] + t];
    }

    // Element access - 4D [d0, d1, d2, d3]
    float& operator()(size_t d0, size_t d1, size_t d2, size_t d3) {
        return data_[((d0 * dims_[1] + d1) * dims_[2] + d2) * dims_[3] + d3];
    }
    const float& operator()(size_t d0, size_t d1, size_t d2, size_t d3) const {
        return data_[((d0 * dims_[1] + d1) * dims_[2] + d2) * dims_[3] + d3];
    }

    // Fill with value
    void fill(float val) {
        std::fill(data_.begin(), data_.end(), val);
    }

    // Zero fill
    void zero() { fill(0.0f); }

    // Reshape (must have same total size)
    void reshape(size_t d0, size_t d1 = 1, size_t d2 = 1, size_t d3 = 1) {
        assert(d0 * d1 * d2 * d3 == size_);
        dims_[0] = d0;
        dims_[1] = d1;
        dims_[2] = d2;
        dims_[3] = d3;
    }

    // Clone
    Tensor clone() const;

    // Slice along last dimension [start, end)
    Tensor slice_last(size_t start, size_t end) const;

    // Concatenate along last dimension
    static Tensor cat_last(const Tensor& a, const Tensor& b);

    // Pad last dimension
    Tensor pad_last(size_t left, size_t right, float val = 0.0f) const;

    // Element-wise operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;

    // Transpose last two dimensions (for attention)
    Tensor transpose_last2() const;

private:
    size_t dims_[4] = {0, 0, 0, 0};
    size_t size_ = 0;
    std::vector<float> data_;
};

// Matrix multiplication: [M, K] x [K, N] -> [M, N]
Tensor matmul(const Tensor& a, const Tensor& b);

// Batched matrix multiplication: [B, M, K] x [B, K, N] -> [B, M, N]
Tensor bmm(const Tensor& a, const Tensor& b);

} // namespace llvc

#endif // LLVC_TENSOR_HPP
