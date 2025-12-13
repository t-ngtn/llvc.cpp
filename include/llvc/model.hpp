#ifndef LLVC_MODEL_HPP
#define LLVC_MODEL_HPP

#include "tensor.hpp"
#include "layers.hpp"
#include "activations.hpp"
#include "blocks.hpp"
#include "attention.hpp"
#include "weights.hpp"
#include <tuple>

namespace llvc {

// MaskNet - generates time-domain mask
class MaskNet {
public:
    MaskNet() = default;
    MaskNet(size_t enc_dim, size_t num_enc_layers,
            size_t dec_dim, size_t dec_buf_len, size_t dec_chunk_size,
            size_t num_dec_layers, size_t nhead = 8, size_t ff_dim = 0,
            bool skip_connection = true, bool proj = true);

    void load_weights(const Weights& weights, const std::string& prefix);

    // Initialize buffers
    std::pair<Tensor, Tensor> init_buffers(size_t batch_size) const;

    // Forward pass
    // x: [B, C, T] - input features
    // l: [B, C] - label embedding
    // Returns: mask [B, C, T], updated enc_buf, updated dec_buf
    std::tuple<Tensor, Tensor, Tensor> forward(const Tensor& x, const Tensor& l,
                                                Tensor& enc_buf, Tensor& dec_buf) const;

private:
    size_t enc_dim_;
    size_t dec_dim_;
    bool skip_connection_;
    bool proj_;
    DilatedCausalConvEncoder encoder_;
    CausalTransformerDecoder decoder_;
    Conv1d proj_e2d_e_;
    Conv1d proj_e2d_l_;
    Conv1d proj_d2e_;
};

// LabelEmbedding - simple linear embedding
class LabelEmbedding {
public:
    LabelEmbedding() = default;
    LabelEmbedding(size_t label_len, size_t enc_dim);

    void load_weights(const Weights& weights, const std::string& prefix);

    // Forward: [B, label_len] -> [B, enc_dim]
    Tensor forward(const Tensor& x) const;

private:
    Linear linear1_;
    LayerNorm ln1_;
    Linear linear2_;
    LayerNorm ln2_;
};

// Net - main LLVC model
class Net {
public:
    // Model configuration
    struct Config {
        size_t label_len = 1;
        size_t L = 16;  // hop length
        size_t enc_dim = 512;
        size_t num_enc_layers = 8;
        size_t dec_dim = 256;
        size_t dec_buf_len = 13;
        size_t num_dec_layers = 1;
        size_t dec_chunk_size = 13;
        size_t out_buf_len = 4;
        bool use_pos_enc = true;
        bool skip_connection = true;
        bool proj = true;
        bool lookahead = true;

        // ConvNet config
        bool convnet_prenet = true;
        std::vector<size_t> convnet_kernel_sizes = {3,3,3,3,3,3,3,3,3,3,3,3};
        std::vector<size_t> convnet_dilations = {1,1,1,1,1,1,1,1,1,1,1,1};
        std::vector<size_t> convnet_out_channels = {1,1,1,1,1,1,1,1,1,1,1,1};
    };

    Net() = default;
    explicit Net(const Config& config);

    void load_weights(const Weights& weights);

    // Initialize all buffers
    struct Buffers {
        Tensor enc_buf;
        Tensor dec_buf;
        Tensor out_buf;
        Tensor convnet_ctx;
    };

    Buffers init_buffers(size_t batch_size) const;

    // Forward pass (non-streaming)
    Tensor forward(const Tensor& x) const;

    // Forward pass (streaming) - processes one chunk
    std::pair<Tensor, Buffers> forward_stream(const Tensor& x, Buffers& bufs) const;

    size_t L() const { return config_.L; }
    size_t dec_chunk_size() const { return config_.dec_chunk_size; }
    bool lookahead() const { return config_.lookahead; }
    size_t out_buf_len() const { return config_.out_buf_len; }
    size_t enc_dim() const { return config_.enc_dim; }

private:
    Config config_;
    bool has_convnet_pre_ = false;
    CachedConvNet convnet_pre_;
    Conv1d in_conv_;
    LabelEmbedding label_embedding_;
    MaskNet mask_gen_;
    ConvTranspose1d out_conv_;
};

} // namespace llvc

#endif // LLVC_MODEL_HPP
