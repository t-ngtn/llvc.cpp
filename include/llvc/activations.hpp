#ifndef LLVC_ACTIVATIONS_HPP
#define LLVC_ACTIVATIONS_HPP

#include "tensor.hpp"
#include <cmath>

namespace llvc {

// ReLU activation: max(0, x)
inline Tensor relu(const Tensor& x) {
    Tensor result = x.clone();
    float* data = result.data();
    for (size_t i = 0; i < result.size(); ++i) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
    return result;
}

// In-place ReLU
inline void relu_inplace(Tensor& x) {
    float* data = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
}

// LeakyReLU activation: max(alpha * x, x)
inline Tensor leaky_relu(const Tensor& x, float alpha = 0.01f) {
    Tensor result = x.clone();
    float* data = result.data();
    for (size_t i = 0; i < result.size(); ++i) {
        data[i] = data[i] > 0.0f ? data[i] : alpha * data[i];
    }
    return result;
}

// In-place LeakyReLU
inline void leaky_relu_inplace(Tensor& x, float alpha = 0.01f) {
    float* data = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        data[i] = data[i] > 0.0f ? data[i] : alpha * data[i];
    }
}

// Tanh activation
inline Tensor tanh_activation(const Tensor& x) {
    Tensor result = x.clone();
    float* data = result.data();
    for (size_t i = 0; i < result.size(); ++i) {
        data[i] = std::tanh(data[i]);
    }
    return result;
}

// In-place Tanh
inline void tanh_inplace(Tensor& x) {
    float* data = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        data[i] = std::tanh(data[i]);
    }
}

// Sigmoid activation: 1 / (1 + exp(-x))
inline Tensor sigmoid(const Tensor& x) {
    Tensor result = x.clone();
    float* data = result.data();
    for (size_t i = 0; i < result.size(); ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
    return result;
}

// In-place Sigmoid
inline void sigmoid_inplace(Tensor& x) {
    float* data = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

// Softmax along last dimension
Tensor softmax(const Tensor& x);

// LayerNorm for [B, C, T] tensor - normalize over C dimension
class LayerNorm {
public:
    LayerNorm() : channels_(0), eps_(1e-5f) {}

    LayerNorm(size_t channels, float eps = 1e-5f)
        : channels_(channels), eps_(eps), weight_(channels), bias_(channels) {
        weight_.fill(1.0f);
        bias_.fill(0.0f);
    }

    // Load weights from Weights object
    void load_weights(const Tensor& weight, const Tensor& bias);

    // Forward: normalize over channel dimension for each position
    // Input: [B, C, T]
    // Output: [B, C, T]
    Tensor forward(const Tensor& x) const;

private:
    size_t channels_;
    float eps_;
    Tensor weight_;
    Tensor bias_;
};

// BatchNorm1d for [B, C, T] tensor - normalize over B,T dimensions
class BatchNorm1d {
public:
    BatchNorm1d() : channels_(0), eps_(1e-5f), momentum_(0.1f) {}

    BatchNorm1d(size_t channels, float eps = 1e-5f, float momentum = 0.1f)
        : channels_(channels), eps_(eps), momentum_(momentum),
          weight_(channels), bias_(channels),
          running_mean_(channels), running_var_(channels) {
        weight_.fill(1.0f);
        bias_.fill(0.0f);
        running_mean_.fill(0.0f);
        running_var_.fill(1.0f);
    }

    // Load weights
    void load_weights(const Tensor& weight, const Tensor& bias,
                      const Tensor& running_mean, const Tensor& running_var);

    // Forward (inference mode - use running stats)
    // Input: [B, C, T]
    // Output: [B, C, T]
    Tensor forward(const Tensor& x) const;

private:
    size_t channels_;
    float eps_;
    float momentum_;
    Tensor weight_;
    Tensor bias_;
    Tensor running_mean_;
    Tensor running_var_;
};

} // namespace llvc

#endif // LLVC_ACTIVATIONS_HPP
