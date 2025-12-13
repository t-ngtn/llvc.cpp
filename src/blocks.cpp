#include "llvc/blocks.hpp"

namespace llvc {

// ============================================================================
// DepthwiseSeparableConv
// ============================================================================

DepthwiseSeparableConv::DepthwiseSeparableConv(size_t in_channels, size_t out_channels,
                                               size_t kernel_size, size_t stride, size_t padding,
                                               size_t dilation)
    : depthwise_(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, true),
      ln1_(in_channels),
      pointwise_(in_channels, out_channels, 1, 1, 0, 1, 1, true),
      ln2_(out_channels) {}

void DepthwiseSeparableConv::load_weights(const Weights& weights, const std::string& prefix) {
    // depthwise conv: layers.0
    depthwise_.load_weights(weights.get(prefix + ".layers.0.weight"),
                            &weights.get(prefix + ".layers.0.bias"));

    // layer norm 1: layers.1
    ln1_.load_weights(weights.get(prefix + ".layers.1.weight"),
                      weights.get(prefix + ".layers.1.bias"));

    // pointwise conv: layers.3
    pointwise_.load_weights(weights.get(prefix + ".layers.3.weight"),
                            &weights.get(prefix + ".layers.3.bias"));

    // layer norm 2: layers.4
    ln2_.load_weights(weights.get(prefix + ".layers.4.weight"),
                      weights.get(prefix + ".layers.4.bias"));
}

void DepthwiseSeparableConv::set_dilation(size_t dilation) {
    depthwise_.set_params(1, 0, dilation, depthwise_.kernel_size());
}

Tensor DepthwiseSeparableConv::forward(const Tensor& x) const {
    // Depthwise conv
    Tensor out = depthwise_.forward(x);

    // LayerNorm + ReLU
    out = ln1_.forward(out);
    relu_inplace(out);

    // Pointwise conv
    out = pointwise_.forward(out);

    // LayerNorm + ReLU
    out = ln2_.forward(out);
    relu_inplace(out);

    return out;
}

// ============================================================================
// ResidualBlock
// ============================================================================

ResidualBlock::ResidualBlock(size_t in_channels, size_t out_channels,
                             size_t kernel_size, size_t dilation)
    : kernel_size_(kernel_size), dilation_(dilation),
      output_crop_(dilation * (kernel_size - 1)),
      filter_(in_channels, out_channels, kernel_size, 1, 0, dilation, 1, true),
      gate_(in_channels, out_channels, kernel_size, 1, 0, dilation, 1, true) {}

void ResidualBlock::load_weights(const Weights& weights, const std::string& prefix) {
    filter_.load_weights(weights.get(prefix + ".filter.weight"),
                         &weights.get(prefix + ".filter.bias"));
    gate_.load_weights(weights.get(prefix + ".gate.weight"),
                       &weights.get(prefix + ".gate.bias"));
}

Tensor ResidualBlock::forward(const Tensor& x) const {
    // Apply filter and gate
    Tensor filtered = filter_.forward(x);
    Tensor gated = gate_.forward(x);

    // tanh(filter) * sigmoid(gate)
    tanh_inplace(filtered);
    sigmoid_inplace(gated);
    Tensor residual = filtered * gated;

    // Residual connection: x[..., output_crop:] + residual
    // Note: x needs to be cropped to match residual size
    size_t T_out = residual.dim(2);
    size_t B = x.dim(0);
    size_t C_out = residual.dim(1);
    size_t C_in = x.dim(1);

    Tensor result(B, C_out, T_out);

    // Pad channel dimension if needed
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C_out; ++c) {
            for (size_t t = 0; t < T_out; ++t) {
                float x_val = 0.0f;
                if (c < C_in) {
                    x_val = x(b, c, output_crop_ + t);
                }
                result(b, c, t) = x_val + residual(b, c, t);
            }
        }
    }

    return result;
}

// ============================================================================
// CachedConvNet
// ============================================================================

CachedConvNet::CachedConvNet(size_t num_channels,
                             const std::vector<size_t>& kernel_sizes,
                             const std::vector<size_t>& dilations,
                             const std::vector<size_t>& out_channels)
    : num_layers_(kernel_sizes.size()),
      kernel_sizes_(kernel_sizes),
      dilations_(dilations),
      out_channels_(out_channels) {

    // Compute buffer lengths and indices
    buf_lengths_.resize(num_layers_);
    buf_indices_.resize(num_layers_);
    size_t total_buf = 0;

    for (size_t i = 0; i < num_layers_; ++i) {
        buf_lengths_[i] = (kernel_sizes[i] - 1) * dilations[i];
        buf_indices_[i] = total_buf;
        total_buf += buf_lengths_[i];
    }
    total_buf_length_ = total_buf;

    // Create layers
    blocks_.resize(num_layers_);
    size_t in_ch = num_channels;
    for (size_t i = 0; i < num_layers_; ++i) {
        blocks_[i] = ResidualBlock(in_ch, out_channels[i], kernel_sizes[i], dilations[i]);
        in_ch = out_channels[i];
    }

    ctx_height_ = *std::max_element(out_channels.begin(), out_channels.end());
}

void CachedConvNet::load_weights(const Weights& weights, const std::string& prefix) {
    for (size_t i = 0; i < num_layers_; ++i) {
        blocks_[i].load_weights(weights, prefix + ".down_convs." + std::to_string(i));
    }
}

Tensor CachedConvNet::init_ctx_buf(size_t batch_size) const {
    return Tensor(batch_size, ctx_height_, total_buf_length_);
}

std::pair<Tensor, Tensor> CachedConvNet::forward(const Tensor& x, Tensor& ctx) const {
    Tensor out = x.clone();

    for (size_t i = 0; i < num_layers_; ++i) {
        size_t buf_start = buf_indices_[i];
        size_t buf_len = buf_lengths_[i];

        // Get context for this layer
        size_t B = out.dim(0);
        size_t C = out.dim(1);
        size_t T = out.dim(2);

        // Concatenate context with input
        Tensor conv_in(B, C, buf_len + T);
        for (size_t b = 0; b < B; ++b) {
            for (size_t c = 0; c < C; ++c) {
                // Copy context
                for (size_t t = 0; t < buf_len; ++t) {
                    conv_in(b, c, t) = ctx(b, c, buf_start + t);
                }
                // Copy input
                for (size_t t = 0; t < T; ++t) {
                    conv_in(b, c, buf_len + t) = out(b, c, t);
                }
            }
        }

        // Update context buffer with the tail of conv_in
        for (size_t b = 0; b < B; ++b) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t t = 0; t < buf_len; ++t) {
                    ctx(b, c, buf_start + t) = conv_in(b, c, conv_in.dim(2) - buf_len + t);
                }
            }
        }

        // Apply residual block
        out = blocks_[i].forward(conv_in);
    }

    return {out, ctx};
}

// ============================================================================
// DilatedCausalConvEncoder
// ============================================================================

DilatedCausalConvEncoder::DilatedCausalConvEncoder(size_t channels, size_t num_layers, size_t kernel_size)
    : channels_(channels), num_layers_(num_layers), kernel_size_(kernel_size) {

    // Compute buffer lengths: (kernel_size - 1) * 2^i
    buf_lengths_.resize(num_layers);
    buf_indices_.resize(num_layers);
    size_t total = 0;

    for (size_t i = 0; i < num_layers; ++i) {
        buf_lengths_[i] = (kernel_size - 1) * (1 << i);  // 2^i
        buf_indices_[i] = total;
        total += buf_lengths_[i];
    }
    total_buf_length_ = total;

    // Create DCC layers
    dcc_layers_.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        dcc_layers_[i] = DepthwiseSeparableConv(channels, channels, kernel_size, 1, 0, 1 << i);
    }
}

void DilatedCausalConvEncoder::load_weights(const Weights& weights, const std::string& prefix) {
    for (size_t i = 0; i < num_layers_; ++i) {
        dcc_layers_[i].load_weights(weights, prefix + ".dcc_layers.dcc_" + std::to_string(i));
    }
}

Tensor DilatedCausalConvEncoder::init_ctx_buf(size_t batch_size) const {
    return Tensor(batch_size, channels_, total_buf_length_);
}

std::pair<Tensor, Tensor> DilatedCausalConvEncoder::forward(const Tensor& x, Tensor& ctx_buf) const {
    Tensor out = x.clone();
    size_t T = x.dim(2);

    for (size_t i = 0; i < num_layers_; ++i) {
        size_t buf_start = buf_indices_[i];
        size_t buf_len = buf_lengths_[i];
        size_t B = out.dim(0);
        size_t C = out.dim(1);

        // Concatenate context with input
        Tensor dcc_in(B, C, buf_len + T);
        for (size_t b = 0; b < B; ++b) {
            for (size_t c = 0; c < C; ++c) {
                // Copy context
                for (size_t t = 0; t < buf_len; ++t) {
                    dcc_in(b, c, t) = ctx_buf(b, c, buf_start + t);
                }
                // Copy current output
                for (size_t t = 0; t < T; ++t) {
                    dcc_in(b, c, buf_len + t) = out(b, c, t);
                }
            }
        }

        // Update context buffer
        for (size_t b = 0; b < B; ++b) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t t = 0; t < buf_len; ++t) {
                    ctx_buf(b, c, buf_start + t) = dcc_in(b, c, dcc_in.dim(2) - buf_len + t);
                }
            }
        }

        // Apply DCC layer with residual connection
        Tensor dcc_out = dcc_layers_[i].forward(dcc_in);

        // Residual connection
        for (size_t b = 0; b < B; ++b) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t t = 0; t < T; ++t) {
                    out(b, c, t) = out(b, c, t) + dcc_out(b, c, t);
                }
            }
        }
    }

    return {out, ctx_buf};
}

} // namespace llvc
