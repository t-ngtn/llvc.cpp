#include "llvc/tensor.hpp"

namespace llvc {

// Constructor from data
Tensor::Tensor(const std::vector<size_t>& shape, const float* data) {
    size_ = 1;
    for (size_t i = 0; i < shape.size() && i < 4; ++i) {
        dims_[i] = shape[i];
        size_ *= shape[i];
    }
    for (size_t i = shape.size(); i < 4; ++i) {
        dims_[i] = 1;
    }
    data_.assign(data, data + size_);
}

// Clone
Tensor Tensor::clone() const {
    Tensor t;
    t.dims_[0] = dims_[0];
    t.dims_[1] = dims_[1];
    t.dims_[2] = dims_[2];
    t.dims_[3] = dims_[3];
    t.size_ = size_;
    t.data_ = data_;
    return t;
}

// Slice along last dimension [start, end)
Tensor Tensor::slice_last(size_t start, size_t end) const {
    size_t new_len = end - start;
    size_t batch_channels = size_ / dims_[ndim() - 1];
    size_t old_len = dims_[ndim() - 1];

    Tensor result;
    result.size_ = batch_channels * new_len;
    result.data_.resize(result.size_);

    if (ndim() == 3) {
        result.dims_[0] = dims_[0];
        result.dims_[1] = dims_[1];
        result.dims_[2] = new_len;
        result.dims_[3] = 1;
        for (size_t b = 0; b < dims_[0]; ++b) {
            for (size_t c = 0; c < dims_[1]; ++c) {
                for (size_t t = 0; t < new_len; ++t) {
                    result(b, c, t) = (*this)(b, c, start + t);
                }
            }
        }
    } else if (ndim() == 2) {
        result.dims_[0] = dims_[0];
        result.dims_[1] = new_len;
        result.dims_[2] = 1;
        result.dims_[3] = 1;
        for (size_t i = 0; i < dims_[0]; ++i) {
            for (size_t j = 0; j < new_len; ++j) {
                result(i, j) = (*this)(i, start + j);
            }
        }
    } else {
        result.dims_[0] = new_len;
        result.dims_[1] = 1;
        result.dims_[2] = 1;
        result.dims_[3] = 1;
        for (size_t i = 0; i < new_len; ++i) {
            result(i) = (*this)(start + i);
        }
    }
    return result;
}

// Concatenate along last dimension
Tensor Tensor::cat_last(const Tensor& a, const Tensor& b) {
    assert(a.ndim() == b.ndim());
    size_t nd = a.ndim();

    Tensor result;
    if (nd == 3) {
        assert(a.dim(0) == b.dim(0) && a.dim(1) == b.dim(1));
        size_t new_len = a.dim(2) + b.dim(2);
        result = Tensor(a.dim(0), a.dim(1), new_len);
        for (size_t ba = 0; ba < a.dim(0); ++ba) {
            for (size_t c = 0; c < a.dim(1); ++c) {
                for (size_t t = 0; t < a.dim(2); ++t) {
                    result(ba, c, t) = a(ba, c, t);
                }
                for (size_t t = 0; t < b.dim(2); ++t) {
                    result(ba, c, a.dim(2) + t) = b(ba, c, t);
                }
            }
        }
    } else if (nd == 2) {
        assert(a.dim(0) == b.dim(0));
        size_t new_len = a.dim(1) + b.dim(1);
        result = Tensor(a.dim(0), new_len);
        for (size_t i = 0; i < a.dim(0); ++i) {
            for (size_t j = 0; j < a.dim(1); ++j) {
                result(i, j) = a(i, j);
            }
            for (size_t j = 0; j < b.dim(1); ++j) {
                result(i, a.dim(1) + j) = b(i, j);
            }
        }
    } else {
        size_t new_len = a.dim(0) + b.dim(0);
        result = Tensor(new_len);
        for (size_t i = 0; i < a.dim(0); ++i) {
            result(i) = a(i);
        }
        for (size_t i = 0; i < b.dim(0); ++i) {
            result(a.dim(0) + i) = b(i);
        }
    }
    return result;
}

// Pad last dimension
Tensor Tensor::pad_last(size_t left, size_t right, float val) const {
    size_t nd = ndim();
    Tensor result;

    if (nd == 3) {
        size_t new_len = dims_[2] + left + right;
        result = Tensor(dims_[0], dims_[1], new_len);
        result.fill(val);
        for (size_t b = 0; b < dims_[0]; ++b) {
            for (size_t c = 0; c < dims_[1]; ++c) {
                for (size_t t = 0; t < dims_[2]; ++t) {
                    result(b, c, left + t) = (*this)(b, c, t);
                }
            }
        }
    } else if (nd == 2) {
        size_t new_len = dims_[1] + left + right;
        result = Tensor(dims_[0], new_len);
        result.fill(val);
        for (size_t i = 0; i < dims_[0]; ++i) {
            for (size_t j = 0; j < dims_[1]; ++j) {
                result(i, left + j) = (*this)(i, j);
            }
        }
    } else {
        size_t new_len = dims_[0] + left + right;
        result = Tensor(new_len);
        result.fill(val);
        for (size_t i = 0; i < dims_[0]; ++i) {
            result(left + i) = (*this)(i);
        }
    }
    return result;
}

// Element-wise addition
Tensor Tensor::operator+(const Tensor& other) const {
    assert(size_ == other.size_);
    Tensor result = clone();
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] += other.data_[i];
    }
    return result;
}

// Element-wise multiplication
Tensor Tensor::operator*(const Tensor& other) const {
    assert(size_ == other.size_);
    Tensor result = clone();
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] *= other.data_[i];
    }
    return result;
}

// Scalar multiplication
Tensor Tensor::operator*(float scalar) const {
    Tensor result = clone();
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] *= scalar;
    }
    return result;
}

// Transpose last two dimensions (for attention)
Tensor Tensor::transpose_last2() const {
    assert(ndim() >= 2);
    if (ndim() == 2) {
        Tensor result(dims_[1], dims_[0]);
        for (size_t i = 0; i < dims_[0]; ++i) {
            for (size_t j = 0; j < dims_[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    } else if (ndim() == 3) {
        Tensor result(dims_[0], dims_[2], dims_[1]);
        for (size_t b = 0; b < dims_[0]; ++b) {
            for (size_t i = 0; i < dims_[1]; ++i) {
                for (size_t j = 0; j < dims_[2]; ++j) {
                    result(b, j, i) = (*this)(b, i, j);
                }
            }
        }
        return result;
    }
    return clone();
}

// Matrix multiplication: [M, K] x [K, N] -> [M, N]
Tensor matmul(const Tensor& a, const Tensor& b) {
    assert(a.dim(1) == b.dim(0));
    size_t M = a.dim(0);
    size_t K = a.dim(1);
    size_t N = b.dim(1);

    Tensor result(M, N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Batched matrix multiplication: [B, M, K] x [B, K, N] -> [B, M, N]
Tensor bmm(const Tensor& a, const Tensor& b) {
    assert(a.dim(0) == b.dim(0));
    assert(a.dim(2) == b.dim(1));
    size_t B = a.dim(0);
    size_t M = a.dim(1);
    size_t K = a.dim(2);
    size_t N = b.dim(2);

    Tensor result(B, M, N);
    for (size_t batch = 0; batch < B; ++batch) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a(batch, i, k) * b(batch, k, j);
                }
                result(batch, i, j) = sum;
            }
        }
    }
    return result;
}

} // namespace llvc
