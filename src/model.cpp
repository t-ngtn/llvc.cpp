#include "llvc/model.hpp"

namespace llvc {

// ============================================================================
// MaskNet
// ============================================================================

MaskNet::MaskNet(size_t enc_dim, size_t num_enc_layers,
                 size_t dec_dim, size_t dec_buf_len, size_t dec_chunk_size,
                 size_t num_dec_layers, size_t nhead, size_t ff_dim,
                 bool skip_connection, bool proj)
    : enc_dim_(enc_dim), dec_dim_(dec_dim),
      skip_connection_(skip_connection), proj_(proj),
      encoder_(enc_dim, num_enc_layers, 3),
      decoder_(dec_dim, dec_buf_len, dec_chunk_size, num_dec_layers, nhead,
               ff_dim > 0 ? ff_dim : 2 * dec_dim) {

    if (proj) {
        proj_e2d_e_ = Conv1d(enc_dim, dec_dim, 1, 1, 0, 1, dec_dim, true);
        proj_e2d_l_ = Conv1d(enc_dim, dec_dim, 1, 1, 0, 1, dec_dim, true);
        proj_d2e_ = Conv1d(dec_dim, enc_dim, 1, 1, 0, 1, dec_dim, true);
    }
}

void MaskNet::load_weights(const Weights& weights, const std::string& prefix) {
    encoder_.load_weights(weights, prefix + ".encoder");

    if (proj_) {
        proj_e2d_e_.load_weights(weights.get(prefix + ".proj_e2d_e.0.weight"),
                                  &weights.get(prefix + ".proj_e2d_e.0.bias"));
        proj_e2d_l_.load_weights(weights.get(prefix + ".proj_e2d_l.0.weight"),
                                  &weights.get(prefix + ".proj_e2d_l.0.bias"));
        proj_d2e_.load_weights(weights.get(prefix + ".proj_d2e.0.weight"),
                                &weights.get(prefix + ".proj_d2e.0.bias"));
    }

    decoder_.load_weights(weights, prefix + ".decoder");
}

std::pair<Tensor, Tensor> MaskNet::init_buffers(size_t batch_size) const {
    Tensor enc_buf = encoder_.init_ctx_buf(batch_size);
    Tensor dec_buf = decoder_.init_ctx_buf(batch_size);
    return {enc_buf, dec_buf};
}

std::tuple<Tensor, Tensor, Tensor> MaskNet::forward(const Tensor& x, const Tensor& l,
                                                    Tensor& enc_buf, Tensor& dec_buf) const {
    size_t B = x.dim(0);
    size_t C = x.dim(1);
    size_t T = x.dim(2);

    // Encode input
    auto [e, enc_buf_out] = encoder_.forward(x, enc_buf);
    enc_buf = enc_buf_out;

    // Label integration: l.unsqueeze(2) * e
    Tensor le(B, C, T);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t t = 0; t < T; ++t) {
                le(b, c, t) = l(b, c) * e(b, c, t);
            }
        }
    }

    Tensor m;
    if (proj_) {
        // Project to decoder dimensions
        Tensor e_proj = proj_e2d_e_.forward(e);
        relu_inplace(e_proj);

        Tensor l_proj = proj_e2d_l_.forward(le);
        relu_inplace(l_proj);

        // Cross-attention
        auto [m_dec, dec_buf_out] = decoder_.forward(l_proj, e_proj, dec_buf);
        dec_buf = dec_buf_out;

        // Project back to encoder dimensions
        m = proj_d2e_.forward(m_dec);
        relu_inplace(m);
    } else {
        auto [m_dec, dec_buf_out] = decoder_.forward(le, e, dec_buf);
        dec_buf = dec_buf_out;
        m = m_dec;
    }

    // Skip connection
    if (skip_connection_) {
        for (size_t b = 0; b < B; ++b) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t t = 0; t < T; ++t) {
                    m(b, c, t) = le(b, c, t) + m(b, c, t);
                }
            }
        }
    }

    return {m, enc_buf, dec_buf};
}

// ============================================================================
// LabelEmbedding
// ============================================================================

LabelEmbedding::LabelEmbedding(size_t label_len, size_t enc_dim)
    : linear1_(label_len, 512),
      ln1_(512),
      linear2_(512, enc_dim),
      ln2_(enc_dim) {}

void LabelEmbedding::load_weights(const Weights& weights, const std::string& prefix) {
    // Linear1
    Tensor w1 = weights.get(prefix + ".0.weight");
    Tensor b1 = weights.get(prefix + ".0.bias");
    linear1_.load_weights(w1, &b1);

    // LayerNorm1
    ln1_.load_weights(weights.get(prefix + ".1.weight"),
                      weights.get(prefix + ".1.bias"));

    // Linear2
    Tensor w2 = weights.get(prefix + ".3.weight");
    Tensor b2 = weights.get(prefix + ".3.bias");
    linear2_.load_weights(w2, &b2);

    // LayerNorm2
    ln2_.load_weights(weights.get(prefix + ".4.weight"),
                      weights.get(prefix + ".4.bias"));
}

Tensor LabelEmbedding::forward(const Tensor& x) const {
    // Linear1
    Tensor h = linear1_.forward(x);

    // LayerNorm1 + ReLU (treat as [B, 512, 1] for LayerNorm)
    size_t B = h.dim(0);
    Tensor h3d(B, 512, 1);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < 512; ++c) {
            h3d(b, c, 0) = h(b, c);
        }
    }
    h3d = ln1_.forward(h3d);
    relu_inplace(h3d);

    // Convert back to 2D
    Tensor h2d(B, 512);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < 512; ++c) {
            h2d(b, c) = h3d(b, c, 0);
        }
    }

    // Linear2
    Tensor out = linear2_.forward(h2d);

    // LayerNorm2 + ReLU
    size_t enc_dim = out.dim(1);
    Tensor out3d(B, enc_dim, 1);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < enc_dim; ++c) {
            out3d(b, c, 0) = out(b, c);
        }
    }
    out3d = ln2_.forward(out3d);
    relu_inplace(out3d);

    // Convert back to 2D
    Tensor result(B, enc_dim);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < enc_dim; ++c) {
            result(b, c) = out3d(b, c, 0);
        }
    }

    return result;
}

// ============================================================================
// Net
// ============================================================================

Net::Net(const Config& config)
    : config_(config),
      label_embedding_(config.label_len, config.enc_dim),
      mask_gen_(config.enc_dim, config.num_enc_layers,
                config.dec_dim, config.dec_buf_len, config.dec_chunk_size,
                config.num_dec_layers, 8, 2 * config.dec_dim,
                config.skip_connection, config.proj) {

    // Input conv
    size_t kernel_size = config.lookahead ? 3 * config.L : config.L;
    in_conv_ = Conv1d(1, config.enc_dim, kernel_size, config.L, 0, 1, 1, false);

    // Output conv (transposed)
    size_t out_kernel = (config.out_buf_len + 1) * config.L;
    size_t out_padding = config.out_buf_len * config.L;
    out_conv_ = ConvTranspose1d(config.enc_dim, 1, out_kernel, config.L, out_padding, 0, false);

    // ConvNet preprocessing
    if (config.convnet_prenet) {
        has_convnet_pre_ = true;
        convnet_pre_ = CachedConvNet(1, config.convnet_kernel_sizes,
                                      config.convnet_dilations,
                                      config.convnet_out_channels);
    } else {
        has_convnet_pre_ = false;
    }
}

void Net::load_weights(const Weights& weights) {
    // ConvNet pre
    if (has_convnet_pre_) {
        convnet_pre_.load_weights(weights, "convnet_pre");
    }

    // Input conv
    in_conv_.load_weights(weights.get("in_conv.0.weight"), nullptr);

    // Label embedding
    label_embedding_.load_weights(weights, "label_embedding");

    // MaskNet
    mask_gen_.load_weights(weights, "mask_gen");

    // Output conv
    out_conv_.load_weights(weights.get("out_conv.0.weight"), nullptr);
}

Net::Buffers Net::init_buffers(size_t batch_size) const {
    Buffers bufs;
    auto [enc_buf, dec_buf] = mask_gen_.init_buffers(batch_size);
    bufs.enc_buf = enc_buf;
    bufs.dec_buf = dec_buf;
    bufs.out_buf = Tensor(batch_size, config_.enc_dim, config_.out_buf_len);

    if (has_convnet_pre_) {
        bufs.convnet_ctx = convnet_pre_.init_ctx_buf(batch_size);
    }

    return bufs;
}

Tensor Net::forward(const Tensor& x) const {
    size_t B = x.dim(0);
    size_t T = x.dim(2);

    // Initialize buffers
    Buffers bufs = init_buffers(B);

    // Pad input
    Tensor padded = x;
    if (config_.lookahead) {
        padded = x.pad_last(config_.L, config_.L);
    }

    // Mod pad to L
    size_t mod = 0;
    size_t T_padded = padded.dim(2);
    if (T_padded % config_.L != 0) {
        mod = config_.L - (T_padded % config_.L);
        padded = padded.pad_last(0, mod);
    }

    // ConvNet preprocessing
    if (has_convnet_pre_) {
        auto [conv_out, ctx] = convnet_pre_.forward(padded, bufs.convnet_ctx);
        // Skip connection: add
        for (size_t i = 0; i < padded.size(); ++i) {
            padded.data()[i] += conv_out.data()[i];
        }
    }

    // Input conv + ReLU
    Tensor h = in_conv_.forward(padded);
    relu_inplace(h);

    // Label embedding (zeros)
    Tensor label(B, 1);
    label.fill(0.0f);
    Tensor l = label_embedding_.forward(label);

    // Mask generation
    auto [m, enc_buf, dec_buf] = mask_gen_.forward(h, l, bufs.enc_buf, bufs.dec_buf);

    // Apply mask
    Tensor masked = h * m;

    // Concatenate with output buffer
    Tensor with_buf = Tensor::cat_last(bufs.out_buf, masked);

    // Output conv + Tanh
    Tensor out = out_conv_.forward(with_buf);
    tanh_inplace(out);

    // Remove padding
    if (mod > 0) {
        out = out.slice_last(0, out.dim(2) - mod);
    }

    return out;
}

std::pair<Tensor, Net::Buffers> Net::forward_stream(const Tensor& x, Buffers& bufs) const {
    size_t B = x.dim(0);

    // ConvNet preprocessing
    Tensor processed = x;
    if (has_convnet_pre_) {
        auto [conv_out, ctx] = convnet_pre_.forward(x, bufs.convnet_ctx);
        bufs.convnet_ctx = ctx;
        // Skip connection: add
        for (size_t i = 0; i < processed.size(); ++i) {
            processed.data()[i] += conv_out.data()[i];
        }
    }

    // Input conv + ReLU
    Tensor h = in_conv_.forward(processed);
    relu_inplace(h);

    // Label embedding (zeros)
    Tensor label(B, 1);
    label.fill(0.0f);
    Tensor l = label_embedding_.forward(label);

    // Mask generation
    auto [m, enc_buf, dec_buf] = mask_gen_.forward(h, l, bufs.enc_buf, bufs.dec_buf);
    bufs.enc_buf = enc_buf;
    bufs.dec_buf = dec_buf;

    // Apply mask
    Tensor masked = h * m;

    // Concatenate with output buffer
    Tensor with_buf = Tensor::cat_last(bufs.out_buf, masked);

    // Update output buffer
    size_t T_out = masked.dim(2);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < config_.enc_dim; ++c) {
            for (size_t t = 0; t < config_.out_buf_len; ++t) {
                bufs.out_buf(b, c, t) = masked(b, c, T_out - config_.out_buf_len + t);
            }
        }
    }

    // Output conv + Tanh
    Tensor out = out_conv_.forward(with_buf);
    tanh_inplace(out);

    return {out, bufs};
}

} // namespace llvc
