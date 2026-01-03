#include "llvc/attention.hpp"

namespace llvc {

// ============================================================================
// PositionalEncoding
// ============================================================================

PositionalEncoding::PositionalEncoding(size_t model_dim, size_t max_len)
    : model_dim_(model_dim), max_len_(max_len), pe_(1, max_len, model_dim) {

    // Compute positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d))
    //                               PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    for (size_t pos = 0; pos < max_len; ++pos) {
        for (size_t i = 0; i < model_dim; i += 2) {
            float div_term = std::exp(-static_cast<float>(i) * std::log(10000.0f) / model_dim);
            pe_(0, pos, i) = std::sin(pos * div_term);
            if (i + 1 < model_dim) {
                pe_(0, pos, i + 1) = std::cos(pos * div_term);
            }
        }
    }
}

void PositionalEncoding::load_weights(const Tensor& pe) {
    pe_ = pe.clone();
    max_len_ = pe.dim(1);
    model_dim_ = pe.dim(2);
}

Tensor PositionalEncoding::forward(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t T = x.dim(1);
    size_t D = x.dim(2);

    Tensor result = x.clone();
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T && t < max_len_; ++t) {
            for (size_t d = 0; d < D; ++d) {
                result(b, t, d) += pe_(0, t, d);
            }
        }
    }
    return result;
}

// ============================================================================
// MultiHeadAttention
// ============================================================================

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t nhead)
    : d_model_(d_model), nhead_(nhead), head_dim_(d_model / nhead),
      in_proj_weight_(3 * d_model, d_model),
      in_proj_bias_(3 * d_model),
      out_proj_weight_(d_model, d_model),
      out_proj_bias_(d_model) {}

void MultiHeadAttention::load_weights(const Weights& weights, const std::string& prefix) {
    in_proj_weight_ = weights.get(prefix + ".in_proj_weight").clone();
    in_proj_bias_ = weights.get(prefix + ".in_proj_bias").clone();
    out_proj_weight_ = weights.get(prefix + ".out_proj.weight").clone();
    out_proj_bias_ = weights.get(prefix + ".out_proj.bias").clone();

    d_model_ = out_proj_weight_.dim(0);
    // Infer nhead from standard attention dimensions
    nhead_ = 8;  // Fixed for LLVC
    head_dim_ = d_model_ / nhead_;
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value) const {
    size_t B = query.dim(0);
    size_t T_q = query.dim(1);
    size_t T_k = key.dim(1);

    // Project Q, K, V
    Tensor Q = project(query, 0);           // [B, T_q, D]
    Tensor K = project(key, d_model_);      // [B, T_k, D]
    Tensor V = project(value, 2 * d_model_); // [B, T_k, D]

    // Reshape to [B * nhead, T, head_dim]
    Tensor Q_heads = reshape_to_heads(Q);
    Tensor K_heads = reshape_to_heads(K);
    Tensor V_heads = reshape_to_heads(V);

    // Compute attention scores: Q @ K^T / sqrt(head_dim)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    Tensor scores = compute_attention_scores(Q_heads, K_heads, scale);

    // Softmax
    Tensor attn_weights = softmax_2d(scores);

    // Attention output: attn_weights @ V
    Tensor attn_out = attention_output(attn_weights, V_heads);

    // Reshape back to [B, T_q, D]
    Tensor out = reshape_from_heads(attn_out, B, T_q);

    // Output projection
    out = output_projection(out);

    return out;
}

Tensor MultiHeadAttention::project(const Tensor& x, size_t offset) const {
    size_t B = x.dim(0);
    size_t T = x.dim(1);
    size_t D = x.dim(2);

    Tensor result(B, T, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                float sum = in_proj_bias_(offset + d);
                for (size_t k = 0; k < D; ++k) {
                    sum += x(b, t, k) * in_proj_weight_(offset + d, k);
                }
                result(b, t, d) = sum;
            }
        }
    }
    return result;
}

Tensor MultiHeadAttention::reshape_to_heads(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t T = x.dim(1);

    Tensor result(B * nhead_, T, head_dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < nhead_; ++h) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    result(b * nhead_ + h, t, d) = x(b, t, h * head_dim_ + d);
                }
            }
        }
    }
    return result;
}

Tensor MultiHeadAttention::reshape_from_heads(const Tensor& x, size_t B, size_t T) const {
    Tensor result(B, T, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t h = 0; h < nhead_; ++h) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    result(b, t, h * head_dim_ + d) = x(b * nhead_ + h, t, d);
                }
            }
        }
    }
    return result;
}

Tensor MultiHeadAttention::compute_attention_scores(const Tensor& Q, const Tensor& K, float scale) const {
    size_t BH = Q.dim(0);
    size_t T_q = Q.dim(1);
    size_t T_k = K.dim(1);

    Tensor scores(BH, T_q, T_k);
    for (size_t bh = 0; bh < BH; ++bh) {
        for (size_t i = 0; i < T_q; ++i) {
            for (size_t j = 0; j < T_k; ++j) {
                float sum = 0.0f;
                for (size_t d = 0; d < head_dim_; ++d) {
                    sum += Q(bh, i, d) * K(bh, j, d);
                }
                scores(bh, i, j) = sum * scale;
            }
        }
    }
    return scores;
}

Tensor MultiHeadAttention::softmax_2d(const Tensor& x) const {
    size_t D0 = x.dim(0);
    size_t D1 = x.dim(1);
    size_t D2 = x.dim(2);

    Tensor result = x.clone();
    for (size_t i = 0; i < D0; ++i) {
        for (size_t j = 0; j < D1; ++j) {
            // Find max
            float max_val = result(i, j, 0);
            for (size_t k = 1; k < D2; ++k) {
                max_val = std::max(max_val, result(i, j, k));
            }
            // Exp and sum
            float sum = 0.0f;
            for (size_t k = 0; k < D2; ++k) {
                result(i, j, k) = std::exp(result(i, j, k) - max_val);
                sum += result(i, j, k);
            }
            // Normalize
            for (size_t k = 0; k < D2; ++k) {
                result(i, j, k) /= sum;
            }
        }
    }
    return result;
}

Tensor MultiHeadAttention::attention_output(const Tensor& weights, const Tensor& V) const {
    size_t BH = weights.dim(0);
    size_t T_q = weights.dim(1);
    size_t T_k = weights.dim(2);

    Tensor result(BH, T_q, head_dim_);
    for (size_t bh = 0; bh < BH; ++bh) {
        for (size_t i = 0; i < T_q; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                float sum = 0.0f;
                for (size_t j = 0; j < T_k; ++j) {
                    sum += weights(bh, i, j) * V(bh, j, d);
                }
                result(bh, i, d) = sum;
            }
        }
    }
    return result;
}

Tensor MultiHeadAttention::output_projection(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t T = x.dim(1);

    Tensor result(B, T, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                float sum = out_proj_bias_(d);
                for (size_t k = 0; k < d_model_; ++k) {
                    sum += x(b, t, k) * out_proj_weight_(d, k);
                }
                result(b, t, d) = sum;
            }
        }
    }
    return result;
}

// ============================================================================
// CausalTransformerDecoderLayer
// ============================================================================

CausalTransformerDecoderLayer::CausalTransformerDecoderLayer(size_t d_model, size_t nhead, size_t ff_dim, float dropout)
    : d_model_(d_model), nhead_(nhead), ff_dim_(ff_dim),
      self_attn_(d_model, nhead),
      cross_attn_(d_model, nhead),
      linear1_weight_(ff_dim, d_model),
      linear1_bias_(ff_dim),
      linear2_weight_(d_model, ff_dim),
      linear2_bias_(d_model),
      norm1_(d_model),
      norm2_(d_model),
      norm3_(d_model) {}

void CausalTransformerDecoderLayer::load_weights(const Weights& weights, const std::string& prefix) {
    self_attn_.load_weights(weights, prefix + ".self_attn");
    cross_attn_.load_weights(weights, prefix + ".multihead_attn");

    linear1_weight_ = weights.get(prefix + ".linear1.weight").clone();
    linear1_bias_ = weights.get(prefix + ".linear1.bias").clone();
    linear2_weight_ = weights.get(prefix + ".linear2.weight").clone();
    linear2_bias_ = weights.get(prefix + ".linear2.bias").clone();

    // Load norms (2D tensors treated as 1D)
    Tensor n1w = weights.get(prefix + ".norm1.weight");
    Tensor n1b = weights.get(prefix + ".norm1.bias");
    Tensor n2w = weights.get(prefix + ".norm2.weight");
    Tensor n2b = weights.get(prefix + ".norm2.bias");
    Tensor n3w = weights.get(prefix + ".norm3.weight");
    Tensor n3b = weights.get(prefix + ".norm3.bias");

    norm1_.load_weights(n1w, n1b);
    norm2_.load_weights(n2w, n2b);
    norm3_.load_weights(n3w, n3b);

    d_model_ = linear2_weight_.dim(0);
    ff_dim_ = linear1_weight_.dim(0);
}

std::pair<Tensor, DecLayerDebug> CausalTransformerDecoderLayer::forward(const Tensor& tgt, const Tensor& memory, size_t chunk_size) const {
    size_t B = tgt.dim(0);
    size_t T_tgt = tgt.dim(1);

    DecLayerDebug dbg;

    // Get last chunk_size tokens
    Tensor tgt_last(B, chunk_size, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < chunk_size; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                tgt_last(b, t, d) = tgt(b, T_tgt - chunk_size + t, d);
            }
        }
    }

    // Self-attention
    Tensor sa_out = self_attn_.forward(tgt_last, tgt, tgt);
    dbg.sa_out = sa_out;

    // Residual + LayerNorm
    Tensor out1(B, chunk_size, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < chunk_size; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                out1(b, t, d) = tgt_last(b, t, d) + sa_out(b, t, d);
            }
        }
    }
    out1 = apply_layer_norm(out1, norm1_);

    // Cross-attention
    Tensor ca_out = cross_attn_.forward(out1, memory, memory);
    dbg.ca_out = ca_out;

    // Residual + LayerNorm
    Tensor out2(B, chunk_size, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < chunk_size; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                out2(b, t, d) = out1(b, t, d) + ca_out(b, t, d);
            }
        }
    }
    out2 = apply_layer_norm(out2, norm2_);

    // FFN: linear1 -> ReLU -> linear2
    Tensor ff_out = ffn_forward(out2);

    // Residual + LayerNorm
    Tensor out3(B, chunk_size, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < chunk_size; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                out3(b, t, d) = out2(b, t, d) + ff_out(b, t, d);
            }
        }
    }
    out3 = apply_layer_norm(out3, norm3_);

    return {out3, dbg};
}

Tensor CausalTransformerDecoderLayer::apply_layer_norm(const Tensor& x, const LayerNorm& norm) const {
    size_t B = x.dim(0);
    size_t T = x.dim(1);
    size_t D = x.dim(2);

    // Reshape to [B*T, D, 1] for LayerNorm
    Tensor reshaped(B * T, D, 1);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                reshaped(b * T + t, d, 0) = x(b, t, d);
            }
        }
    }

    Tensor normed = norm.forward(reshaped);

    // Reshape back
    Tensor result(B, T, D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                result(b, t, d) = normed(b * T + t, d, 0);
            }
        }
    }
    return result;
}

Tensor CausalTransformerDecoderLayer::ffn_forward(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t T = x.dim(1);

    // Linear1
    Tensor h(B, T, ff_dim_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t f = 0; f < ff_dim_; ++f) {
                float sum = linear1_bias_(f);
                for (size_t d = 0; d < d_model_; ++d) {
                    sum += x(b, t, d) * linear1_weight_(f, d);
                }
                // ReLU
                h(b, t, f) = sum > 0 ? sum : 0;
            }
        }
    }

    // Linear2
    Tensor out(B, T, d_model_);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                float sum = linear2_bias_(d);
                for (size_t f = 0; f < ff_dim_; ++f) {
                    sum += h(b, t, f) * linear2_weight_(d, f);
                }
                out(b, t, d) = sum;
            }
        }
    }

    return out;
}

// ============================================================================
// CausalTransformerDecoder
// ============================================================================

CausalTransformerDecoder::CausalTransformerDecoder(size_t model_dim, size_t ctx_len, size_t chunk_size,
                                                   size_t num_layers, size_t nhead, size_t ff_dim, float dropout)
    : model_dim_(model_dim), ctx_len_(ctx_len), chunk_size_(chunk_size),
      num_layers_(num_layers), nhead_(nhead),
      pos_enc_(model_dim, 200) {

    layers_.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layers_[i] = CausalTransformerDecoderLayer(model_dim, nhead, ff_dim, dropout);
    }
}

void CausalTransformerDecoder::load_weights(const Weights& weights, const std::string& prefix) {
    pos_enc_.load_weights(weights.get(prefix + ".pos_enc.pe"));

    for (size_t i = 0; i < num_layers_; ++i) {
        layers_[i].load_weights(weights, prefix + ".tf_dec_layers." + std::to_string(i));
    }
}

Tensor CausalTransformerDecoder::init_ctx_buf(size_t batch_size) const {
    return Tensor(batch_size, num_layers_ + 1, ctx_len_, model_dim_);
}

std::tuple<Tensor, Tensor, DecDebug> CausalTransformerDecoder::forward(const Tensor& tgt, const Tensor& mem, Tensor& ctx_buf) const {
    size_t B = tgt.dim(0);
    size_t C = tgt.dim(1);
    size_t T = tgt.dim(2);

    DecDebug dbg;

    // Mod pad to chunk_size
    size_t mod = (T % chunk_size_ != 0) ? chunk_size_ - (T % chunk_size_) : 0;
    size_t T_padded = T + mod;

    // Permute [B, C, T] -> [B, T, C]
    Tensor tgt_perm(B, T_padded, C);
    Tensor mem_perm(B, T_padded, C);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t c = 0; c < C; ++c) {
                tgt_perm(b, t, c) = tgt(b, c, t);
                mem_perm(b, t, c) = mem(b, c, t);
            }
        }
        // Pad with zeros
        for (size_t t = T; t < T_padded; ++t) {
            for (size_t c = 0; c < C; ++c) {
                tgt_perm(b, t, c) = 0.0f;
                mem_perm(b, t, c) = 0.0f;
            }
        }
    }

    // Prepend mem with context
    Tensor mem_with_ctx(B, ctx_len_ + T_padded, C);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < ctx_len_; ++t) {
            for (size_t c = 0; c < C; ++c) {
                mem_with_ctx(b, t, c) = ctx_buf(b, 0, t, c);
            }
        }
        for (size_t t = 0; t < T_padded; ++t) {
            for (size_t c = 0; c < C; ++c) {
                mem_with_ctx(b, ctx_len_ + t, c) = mem_perm(b, t, c);
            }
        }
    }

    // Update mem context
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < ctx_len_; ++t) {
            for (size_t c = 0; c < C; ++c) {
                ctx_buf(b, 0, t, c) = mem_with_ctx(b, T_padded + t, c);
            }
        }
    }

    // Apply positional encoding to mem
    Tensor mem_ctx = causal_unfold(mem_with_ctx);
    mem_ctx = pos_enc_.forward(mem_ctx);

    // Process through layers
    Tensor tgt_out = tgt_perm;
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        // Prepend tgt with context
        Tensor tgt_with_ctx(B, ctx_len_ + T_padded, C);
        for (size_t b = 0; b < B; ++b) {
            for (size_t t = 0; t < ctx_len_; ++t) {
                for (size_t c = 0; c < C; ++c) {
                    tgt_with_ctx(b, t, c) = ctx_buf(b, layer + 1, t, c);
                }
            }
            for (size_t t = 0; t < T_padded; ++t) {
                for (size_t c = 0; c < C; ++c) {
                    tgt_with_ctx(b, ctx_len_ + t, c) = tgt_out(b, t, c);
                }
            }
        }

        // Update tgt context
        for (size_t b = 0; b < B; ++b) {
            for (size_t t = 0; t < ctx_len_; ++t) {
                for (size_t c = 0; c < C; ++c) {
                    ctx_buf(b, layer + 1, t, c) = tgt_with_ctx(b, T_padded + t, c);
                }
            }
        }

        // Unfold and apply positional encoding
        Tensor tgt_ctx = causal_unfold(tgt_with_ctx);
        if (layer == 0) {
            tgt_ctx = pos_enc_.forward(tgt_ctx);
        }

        // Process chunks
        size_t num_chunks = T_padded / chunk_size_;
        Tensor new_tgt_out(B, T_padded, C);

        for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
            // Get chunk from unfolded tensors
            size_t chunk_start = chunk * (ctx_len_ + chunk_size_);
            Tensor tgt_chunk(B, ctx_len_ + chunk_size_, C);
            Tensor mem_chunk(B, ctx_len_ + chunk_size_, C);

            for (size_t b = 0; b < B; ++b) {
                for (size_t t = 0; t < ctx_len_ + chunk_size_; ++t) {
                    for (size_t c = 0; c < C; ++c) {
                        tgt_chunk(b, t, c) = tgt_ctx(chunk * B + b, t, c);
                        mem_chunk(b, t, c) = mem_ctx(chunk * B + b, t, c);
                    }
                }
            }

            // Apply decoder layer
            auto [out_chunk, layer_dbg] = layers_[layer].forward(tgt_chunk, mem_chunk, chunk_size_);

            // Capture debug output from first layer, first chunk
            if (layer == 0 && chunk == 0) {
                dbg.sa_out = layer_dbg.sa_out;
                dbg.ca_out = layer_dbg.ca_out;
            }

            // Store output
            for (size_t b = 0; b < B; ++b) {
                for (size_t t = 0; t < chunk_size_; ++t) {
                    for (size_t c = 0; c < C; ++c) {
                        new_tgt_out(b, chunk * chunk_size_ + t, c) = out_chunk(b, t, c);
                    }
                }
            }
        }

        tgt_out = new_tgt_out;
    }

    // Permute back [B, T, C] -> [B, C, T]
    Tensor result(B, C, T);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t c = 0; c < C; ++c) {
                result(b, c, t) = tgt_out(b, t, c);
            }
        }
    }

    return {result, ctx_buf, dbg};
}

Tensor CausalTransformerDecoder::causal_unfold(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t T_total = x.dim(1);
    size_t C = x.dim(2);
    size_t L = T_total - ctx_len_;
    size_t num_chunks = L / chunk_size_;

    Tensor result(num_chunks * B, ctx_len_ + chunk_size_, C);

    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t start = chunk * chunk_size_;
        for (size_t b = 0; b < B; ++b) {
            for (size_t t = 0; t < ctx_len_ + chunk_size_; ++t) {
                for (size_t c = 0; c < C; ++c) {
                    result(chunk * B + b, t, c) = x(b, start + t, c);
                }
            }
        }
    }

    return result;
}

} // namespace llvc
