#include "llvc/layers.hpp"

namespace llvc {

// ============================================================================
// Linear
// ============================================================================

void Linear::load_weights(const Tensor& weight, const Tensor* bias) {
    weight_ = weight.clone();
    out_features_ = weight.dim(0);
    in_features_ = weight.dim(1);
    if (bias) {
        bias_ = bias->clone();
        has_bias_ = true;
    }
}

Tensor Linear::forward(const Tensor& x) const {
    size_t batch = x.dim(0);
    Tensor result(batch, out_features_);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t o = 0; o < out_features_; ++o) {
            float sum = has_bias_ ? bias_(o) : 0.0f;
            for (size_t i = 0; i < in_features_; ++i) {
                sum += x(b, i) * weight_(o, i);
            }
            result(b, o) = sum;
        }
    }
    return result;
}

// ============================================================================
// Conv1d
// ============================================================================

void Conv1d::load_weights(const Tensor& weight, const Tensor* bias) {
    weight_ = weight.clone();
    out_channels_ = weight.dim(0);
    if (weight.ndim() >= 3) {
        in_channels_ = weight.dim(1) * groups_;
        kernel_size_ = weight.dim(2);
    } else if (weight.ndim() == 2) {
        in_channels_ = weight.dim(1) * groups_;
        kernel_size_ = 1;
    }
    if (bias) {
        bias_ = bias->clone();
        has_bias_ = true;
    }
}

void Conv1d::set_params(size_t stride, size_t padding, size_t dilation, size_t groups) {
    stride_ = stride;
    padding_ = padding;
    dilation_ = dilation;
    groups_ = groups;
}

Tensor Conv1d::forward(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t C_in = x.dim(1);
    size_t T_in = x.dim(2);

    // Calculate output length
    size_t effective_kernel = (kernel_size_ - 1) * dilation_ + 1;
    size_t T_out = (T_in + 2 * padding_ - effective_kernel) / stride_ + 1;

    Tensor result(B, out_channels_, T_out);
    result.fill(0.0f);

    size_t channels_per_group = C_in / groups_;
    size_t out_channels_per_group = out_channels_ / groups_;

    for (size_t b = 0; b < B; ++b) {
        for (size_t g = 0; g < groups_; ++g) {
            size_t c_in_start = g * channels_per_group;
            size_t c_out_start = g * out_channels_per_group;

            for (size_t c_out = 0; c_out < out_channels_per_group; ++c_out) {
                size_t out_idx = c_out_start + c_out;

                for (size_t t_out = 0; t_out < T_out; ++t_out) {
                    float sum = has_bias_ ? bias_(out_idx) : 0.0f;

                    for (size_t c_in = 0; c_in < channels_per_group; ++c_in) {
                        for (size_t k = 0; k < kernel_size_; ++k) {
                            int t_in = static_cast<int>(t_out * stride_ + k * dilation_) - static_cast<int>(padding_);
                            if (t_in >= 0 && t_in < static_cast<int>(T_in)) {
                                float w = weight_(out_idx, c_in, k);
                                float v = x(b, c_in_start + c_in, t_in);
                                sum += w * v;
                            }
                        }
                    }
                    result(b, out_idx, t_out) = sum;
                }
            }
        }
    }

    return result;
}

// ============================================================================
// ConvTranspose1d
// ============================================================================

void ConvTranspose1d::load_weights(const Tensor& weight, const Tensor* bias) {
    weight_ = weight.clone();
    in_channels_ = weight.dim(0);
    out_channels_ = weight.dim(1);
    kernel_size_ = weight.dim(2);
    if (bias) {
        bias_ = bias->clone();
        has_bias_ = true;
    }
}

void ConvTranspose1d::set_params(size_t stride, size_t padding, size_t output_padding) {
    stride_ = stride;
    padding_ = padding;
    output_padding_ = output_padding;
}

Tensor ConvTranspose1d::forward(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t C_in = x.dim(1);
    size_t T_in = x.dim(2);

    // Calculate output length
    size_t T_out = (T_in - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;

    Tensor result(B, out_channels_, T_out);
    result.fill(0.0f);

    // Add bias first if exists
    if (has_bias_) {
        for (size_t b = 0; b < B; ++b) {
            for (size_t c = 0; c < out_channels_; ++c) {
                for (size_t t = 0; t < T_out; ++t) {
                    result(b, c, t) = bias_(c);
                }
            }
        }
    }

    // Convolution transpose
    for (size_t b = 0; b < B; ++b) {
        for (size_t c_in = 0; c_in < C_in; ++c_in) {
            for (size_t t_in = 0; t_in < T_in; ++t_in) {
                float input_val = x(b, c_in, t_in);
                for (size_t c_out = 0; c_out < out_channels_; ++c_out) {
                    for (size_t k = 0; k < kernel_size_; ++k) {
                        int t_out = static_cast<int>(t_in * stride_ + k) - static_cast<int>(padding_);
                        if (t_out >= 0 && t_out < static_cast<int>(T_out)) {
                            result(b, c_out, t_out) += input_val * weight_(c_in, c_out, k);
                        }
                    }
                }
            }
        }
    }

    return result;
}

} // namespace llvc
