#ifndef LLVC_LAYERS_HPP
#define LLVC_LAYERS_HPP

#include "tensor.hpp"
#include "activations.hpp"
#include <cassert>
#include <cmath>

namespace llvc {

// Linear layer: y = x @ W^T + b
// Input: [*, in_features]
// Output: [*, out_features]
class Linear {
public:
    Linear() : in_features_(0), out_features_(0), has_bias_(false) {}

    Linear(size_t in_features, size_t out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias),
          weight_(out_features, in_features), bias_(bias ? out_features : 0) {
        weight_.fill(0.0f);
        if (has_bias_) {
            bias_.fill(0.0f);
        }
    }

    void load_weights(const Tensor& weight, const Tensor* bias = nullptr);

    // Forward for 2D input [B, in_features] -> [B, out_features]
    Tensor forward(const Tensor& x) const;

private:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    Tensor weight_;  // [out_features, in_features]
    Tensor bias_;    // [out_features]
};

// Conv1d layer
// Input: [B, C_in, T]
// Output: [B, C_out, T_out]
// weight: [C_out, C_in/groups, kernel_size]
class Conv1d {
public:
    Conv1d() : in_channels_(0), out_channels_(0), kernel_size_(0),
               stride_(1), padding_(0), dilation_(1), groups_(1), has_bias_(false) {}

    Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0, size_t dilation = 1,
           size_t groups = 1, bool bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          dilation_(dilation), groups_(groups), has_bias_(bias),
          weight_(out_channels, in_channels / groups, kernel_size),
          bias_(bias ? out_channels : 0) {
        weight_.fill(0.0f);
        if (has_bias_) {
            bias_.fill(0.0f);
        }
    }

    void load_weights(const Tensor& weight, const Tensor* bias = nullptr);
    void set_params(size_t stride, size_t padding, size_t dilation, size_t groups);

    // Forward
    Tensor forward(const Tensor& x) const;

    size_t kernel_size() const { return kernel_size_; }
    size_t dilation() const { return dilation_; }
    size_t stride() const { return stride_; }
    size_t padding() const { return padding_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t dilation_;
    size_t groups_;
    bool has_bias_;
    Tensor weight_;  // [out_channels, in_channels/groups, kernel_size]
    Tensor bias_;    // [out_channels]
};

// ConvTranspose1d layer
// Input: [B, C_in, T_in]
// Output: [B, C_out, T_out]
// T_out = (T_in - 1) * stride - 2 * padding + kernel_size + output_padding
class ConvTranspose1d {
public:
    ConvTranspose1d() : in_channels_(0), out_channels_(0), kernel_size_(0),
                        stride_(1), padding_(0), output_padding_(0), has_bias_(false) {}

    ConvTranspose1d(size_t in_channels, size_t out_channels, size_t kernel_size,
                    size_t stride = 1, size_t padding = 0, size_t output_padding = 0,
                    bool bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          output_padding_(output_padding), has_bias_(bias),
          weight_(in_channels, out_channels, kernel_size),
          bias_(bias ? out_channels : 0) {
        weight_.fill(0.0f);
        if (has_bias_) {
            bias_.fill(0.0f);
        }
    }

    void load_weights(const Tensor& weight, const Tensor* bias = nullptr);
    void set_params(size_t stride, size_t padding, size_t output_padding = 0);

    Tensor forward(const Tensor& x) const;

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t output_padding_;
    bool has_bias_;
    Tensor weight_;  // [in_channels, out_channels, kernel_size]
    Tensor bias_;    // [out_channels]
};

// Dropout - identity in inference mode
class Dropout {
public:
    Dropout(float p = 0.5f) : p_(p) {}

    Tensor forward(const Tensor& x) const {
        // In inference mode, dropout is identity
        return x.clone();
    }

private:
    float p_;
};

// Dropout1d - same as Dropout for inference
using Dropout1d = Dropout;

} // namespace llvc

#endif // LLVC_LAYERS_HPP
