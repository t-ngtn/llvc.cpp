#include "llvc/activations.hpp"

namespace llvc {

// Softmax along last dimension
Tensor softmax(const Tensor& x) {
    Tensor result = x.clone();
    size_t last_dim = x.dim(x.ndim() - 1);
    size_t batch_size = x.size() / last_dim;

    for (size_t b = 0; b < batch_size; ++b) {
        // Find max for numerical stability
        float max_val = result.data()[b * last_dim];
        for (size_t i = 1; i < last_dim; ++i) {
            max_val = std::max(max_val, result.data()[b * last_dim + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            result.data()[b * last_dim + i] = std::exp(result.data()[b * last_dim + i] - max_val);
            sum += result.data()[b * last_dim + i];
        }

        // Normalize
        for (size_t i = 0; i < last_dim; ++i) {
            result.data()[b * last_dim + i] /= sum;
        }
    }
    return result;
}

// ============================================================================
// LayerNorm
// ============================================================================

void LayerNorm::load_weights(const Tensor& weight, const Tensor& bias) {
    weight_ = weight.clone();
    bias_ = bias.clone();
    channels_ = weight.size();
}

Tensor LayerNorm::forward(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t C = x.dim(1);
    size_t T = x.dim(2);

    Tensor result(B, C, T);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            // Compute mean over channels
            float mean = 0.0f;
            for (size_t c = 0; c < C; ++c) {
                mean += x(b, c, t);
            }
            mean /= C;

            // Compute variance over channels
            float var = 0.0f;
            for (size_t c = 0; c < C; ++c) {
                float diff = x(b, c, t) - mean;
                var += diff * diff;
            }
            var /= C;

            // Normalize
            float inv_std = 1.0f / std::sqrt(var + eps_);
            for (size_t c = 0; c < C; ++c) {
                result(b, c, t) = (x(b, c, t) - mean) * inv_std * weight_(c) + bias_(c);
            }
        }
    }

    return result;
}

// ============================================================================
// BatchNorm1d
// ============================================================================

void BatchNorm1d::load_weights(const Tensor& weight, const Tensor& bias,
                               const Tensor& running_mean, const Tensor& running_var) {
    weight_ = weight.clone();
    bias_ = bias.clone();
    running_mean_ = running_mean.clone();
    running_var_ = running_var.clone();
    channels_ = weight.size();
}

Tensor BatchNorm1d::forward(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t C = x.dim(1);
    size_t T = x.dim(2);

    Tensor result(B, C, T);

    for (size_t c = 0; c < C; ++c) {
        float mean = running_mean_(c);
        float var = running_var_(c);
        float inv_std = 1.0f / std::sqrt(var + eps_);
        float w = weight_(c);
        float b = bias_(c);

        for (size_t ba = 0; ba < B; ++ba) {
            for (size_t t = 0; t < T; ++t) {
                result(ba, c, t) = (x(ba, c, t) - mean) * inv_std * w + b;
            }
        }
    }

    return result;
}

} // namespace llvc
