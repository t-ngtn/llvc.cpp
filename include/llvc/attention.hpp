#ifndef LLVC_ATTENTION_HPP
#define LLVC_ATTENTION_HPP

#include "tensor.hpp"
#include "layers.hpp"
#include "activations.hpp"
#include "weights.hpp"
#include <cmath>
#include <vector>

namespace llvc {

// Positional Encoding
class PositionalEncoding {
public:
    PositionalEncoding() : model_dim_(0), max_len_(0) {}
    PositionalEncoding(size_t model_dim, size_t max_len = 200);

    void load_weights(const Tensor& pe);

    // Add positional encoding to input
    // Input: [B, T, D]
    // Output: [B, T, D]
    Tensor forward(const Tensor& x) const;

private:
    size_t model_dim_;
    size_t max_len_;
    Tensor pe_;  // [1, max_len, model_dim]
};

// Multi-Head Attention
class MultiHeadAttention {
public:
    MultiHeadAttention() : d_model_(0), nhead_(0), head_dim_(0) {}
    MultiHeadAttention(size_t d_model, size_t nhead);

    void load_weights(const Weights& weights, const std::string& prefix);

    // Forward pass
    // query: [B, T_q, D]
    // key: [B, T_k, D]
    // value: [B, T_k, D]
    // Returns: [B, T_q, D]
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value) const;

private:
    size_t d_model_;
    size_t nhead_;
    size_t head_dim_;
    Tensor in_proj_weight_;  // [3*D, D]
    Tensor in_proj_bias_;    // [3*D]
    Tensor out_proj_weight_; // [D, D]
    Tensor out_proj_bias_;   // [D]

    Tensor project(const Tensor& x, size_t offset) const;
    Tensor reshape_to_heads(const Tensor& x) const;
    Tensor reshape_from_heads(const Tensor& x, size_t B, size_t T) const;
    Tensor compute_attention_scores(const Tensor& Q, const Tensor& K, float scale) const;
    Tensor softmax_2d(const Tensor& x) const;
    Tensor attention_output(const Tensor& weights, const Tensor& V) const;
    Tensor output_projection(const Tensor& x) const;
};

// Causal Transformer Decoder Layer
class CausalTransformerDecoderLayer {
public:
    CausalTransformerDecoderLayer() : d_model_(0), nhead_(0), ff_dim_(0) {}
    CausalTransformerDecoderLayer(size_t d_model, size_t nhead, size_t ff_dim, float dropout = 0.1f);

    void load_weights(const Weights& weights, const std::string& prefix);

    // Forward for causal attention
    // tgt: [B, T_tgt, D] - target sequence
    // memory: [B, T_mem, D] - encoder output
    // chunk_size: number of new tokens to process
    // Returns: output for last chunk_size tokens
    Tensor forward(const Tensor& tgt, const Tensor& memory, size_t chunk_size) const;

private:
    size_t d_model_;
    size_t nhead_;
    size_t ff_dim_;
    MultiHeadAttention self_attn_;
    MultiHeadAttention cross_attn_;
    Tensor linear1_weight_;  // [ff_dim, d_model]
    Tensor linear1_bias_;
    Tensor linear2_weight_;  // [d_model, ff_dim]
    Tensor linear2_bias_;
    LayerNorm norm1_;
    LayerNorm norm2_;
    LayerNorm norm3_;

    Tensor apply_layer_norm(const Tensor& x, const LayerNorm& norm) const;
    Tensor ffn_forward(const Tensor& x) const;
};

// Causal Transformer Decoder
class CausalTransformerDecoder {
public:
    CausalTransformerDecoder() : model_dim_(0), ctx_len_(0), chunk_size_(0), num_layers_(0), nhead_(0) {}
    CausalTransformerDecoder(size_t model_dim, size_t ctx_len, size_t chunk_size,
                             size_t num_layers, size_t nhead, size_t ff_dim, float dropout = 0.1f);

    void load_weights(const Weights& weights, const std::string& prefix);

    // Initialize decoder context buffer
    // Shape: [B, num_layers+1, ctx_len, model_dim]
    Tensor init_ctx_buf(size_t batch_size) const;

    // Forward pass
    // tgt: [B, C, T] - target features
    // mem: [B, C, T] - encoder output
    // ctx_buf: [B, num_layers+1, ctx_len, model_dim]
    // Returns: output [B, C, T] and updated context
    std::pair<Tensor, Tensor> forward(const Tensor& tgt, const Tensor& mem, Tensor& ctx_buf) const;

private:
    size_t model_dim_;
    size_t ctx_len_;
    size_t chunk_size_;
    size_t num_layers_;
    size_t nhead_;
    PositionalEncoding pos_enc_;
    std::vector<CausalTransformerDecoderLayer> layers_;

    Tensor causal_unfold(const Tensor& x) const;
};

} // namespace llvc

#endif // LLVC_ATTENTION_HPP
