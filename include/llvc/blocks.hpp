#ifndef LLVC_BLOCKS_HPP
#define LLVC_BLOCKS_HPP

#include "tensor.hpp"
#include "layers.hpp"
#include "activations.hpp"
#include "weights.hpp"
#include <vector>
#include <string>

namespace llvc {

// DepthwiseSeparableConv: depthwise conv + pointwise conv
// Used in DilatedCausalConvEncoder
class DepthwiseSeparableConv {
public:
    DepthwiseSeparableConv() = default;
    DepthwiseSeparableConv(size_t in_channels, size_t out_channels,
                           size_t kernel_size, size_t stride, size_t padding,
                           size_t dilation);

    void load_weights(const Weights& weights, const std::string& prefix);
    void set_dilation(size_t dilation);
    Tensor forward(const Tensor& x) const;

private:
    Conv1d depthwise_;
    LayerNorm ln1_;
    Conv1d pointwise_;
    LayerNorm ln2_;
};

// ResidualBlock for CachedConvNet
// filter and gate -> tanh(filter) * sigmoid(gate)
class ResidualBlock {
public:
    ResidualBlock() = default;
    ResidualBlock(size_t in_channels, size_t out_channels,
                  size_t kernel_size, size_t dilation);

    void load_weights(const Weights& weights, const std::string& prefix);
    size_t output_crop() const { return output_crop_; }
    Tensor forward(const Tensor& x) const;

private:
    size_t kernel_size_;
    size_t dilation_;
    size_t output_crop_;
    Conv1d filter_;
    Conv1d gate_;
};

// CachedConvNet - preprocessing network with context buffering
class CachedConvNet {
public:
    CachedConvNet() = default;
    CachedConvNet(size_t num_channels,
                  const std::vector<size_t>& kernel_sizes,
                  const std::vector<size_t>& dilations,
                  const std::vector<size_t>& out_channels);

    void load_weights(const Weights& weights, const std::string& prefix);

    // Initialize context buffer
    Tensor init_ctx_buf(size_t batch_size) const;

    // Forward with context buffer
    std::pair<Tensor, Tensor> forward(const Tensor& x, Tensor& ctx) const;

    size_t total_buf_length() const { return total_buf_length_; }

private:
    size_t num_layers_;
    std::vector<size_t> kernel_sizes_;
    std::vector<size_t> dilations_;
    std::vector<size_t> out_channels_;
    std::vector<size_t> buf_lengths_;
    std::vector<size_t> buf_indices_;
    size_t total_buf_length_;
    size_t ctx_height_;
    std::vector<ResidualBlock> blocks_;
};

// DilatedCausalConvEncoder - encoder with dilated causal convolutions
class DilatedCausalConvEncoder {
public:
    DilatedCausalConvEncoder() = default;
    DilatedCausalConvEncoder(size_t channels, size_t num_layers, size_t kernel_size = 3);

    void load_weights(const Weights& weights, const std::string& prefix);

    // Initialize context buffer
    Tensor init_ctx_buf(size_t batch_size) const;

    // Forward with context buffer
    std::pair<Tensor, Tensor> forward(const Tensor& x, Tensor& ctx_buf) const;

    size_t total_buf_length() const { return total_buf_length_; }
    size_t channels() const { return channels_; }

private:
    size_t channels_;
    size_t num_layers_;
    size_t kernel_size_;
    std::vector<size_t> buf_lengths_;
    std::vector<size_t> buf_indices_;
    size_t total_buf_length_;
    std::vector<DepthwiseSeparableConv> dcc_layers_;
};

} // namespace llvc

#endif // LLVC_BLOCKS_HPP
