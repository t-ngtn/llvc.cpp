#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "llvc/tensor.hpp"
#include "llvc/audio.hpp"
#include "llvc/weights.hpp"
#include "llvc/model.hpp"

namespace fs = std::filesystem;

// Default paths
const std::string DEFAULT_INPUT_DIR = "test_wavs";
const std::string DEFAULT_OUTPUT_DIR = "converted_out";
const std::string DEFAULT_WEIGHTS = "models/llvc_weights.bin";

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -m, --model <type>     Model type: llvc (default), llvc_nc\n"
              << "  -w, --weights <path>   Path to weights file (auto-selected by model type if not specified)\n"
              << "  -i, --input <path>     Path to input WAV file or directory (default: " << DEFAULT_INPUT_DIR << "/)\n"
              << "  -o, --output <path>    Path to output WAV file or directory (default: " << DEFAULT_OUTPUT_DIR << "/)\n"
              << "  -s, --streaming        Use streaming inference\n"
              << "  -n, --chunk-factor <n> Chunk factor for streaming (default: 1)\n"
              << "  -d, --dump <dir>       Dump first chunk I/O to text files for CSIM (streaming only)\n"
              << "  -h, --help             Show this help\n";
}

// Save tensor data to text file (1 value per line)
void dump_tensor_to_file(const std::string& path, const llvc::Tensor& tensor) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Warning: Could not open " << path << " for writing" << std::endl;
        return;
    }
    ofs << std::setprecision(9);
    for (size_t i = 0; i < tensor.size(); ++i) {
        ofs << tensor.data()[i] << "\n";
    }
    std::cout << "Dumped " << tensor.size() << " values to " << path << std::endl;
}

std::vector<std::string> get_wav_files(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav") {
                files.push_back(entry.path().string());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Process a single audio file
void process_file(llvc::Net& model, const std::string& input_path, const std::string& output_path,
                  bool streaming, int chunk_factor, const std::string& dump_dir = "") {
    // Load audio
    std::cout << "Loading audio from " << input_path << "..." << std::endl;
    uint32_t sample_rate;
    llvc::Tensor audio = llvc::read_wav(input_path, &sample_rate);
    std::cout << "Loaded " << audio.size() << " samples at " << sample_rate << " Hz" << std::endl;

    // Resample to 16kHz if needed
    const uint32_t target_sr = 16000;
    if (sample_rate != target_sr) {
        std::cout << "Resampling from " << sample_rate << " to " << target_sr << " Hz..." << std::endl;
        audio = llvc::resample(audio, sample_rate, target_sr);
        sample_rate = target_sr;
    }

    llvc::Tensor output;

    if (streaming) {
        // Streaming inference
        std::cout << "Running streaming inference with chunk_factor=" << chunk_factor << "..." << std::endl;

        size_t L = model.L();
        size_t chunk_len = model.dec_chunk_size() * L * chunk_factor;
        size_t original_len = audio.size();

        // Pad audio
        size_t padded_len = audio.size();
        if (padded_len % chunk_len != 0) {
            padded_len = ((padded_len / chunk_len) + 1) * chunk_len;
            audio = audio.pad_last(0, padded_len - audio.size());
        }

        // Shift audio by L
        llvc::Tensor shifted(padded_len);
        for (size_t i = 0; i < padded_len - L; ++i) {
            shifted(i) = audio(L + i);
        }
        for (size_t i = padded_len - L; i < padded_len; ++i) {
            shifted(i) = 0.0f;
        }

        // Split into chunks with lookahead context
        size_t num_chunks = padded_len / chunk_len;
        std::vector<llvc::Tensor> chunks;

        for (size_t i = 0; i < num_chunks; ++i) {
            size_t start = i * chunk_len;
            llvc::Tensor chunk(chunk_len + 2 * L);

            // Front context
            for (size_t j = 0; j < 2 * L; ++j) {
                if (i == 0) {
                    chunk(j) = 0.0f;
                } else {
                    size_t prev_idx = (i - 1) * chunk_len + chunk_len - 2 * L + j;
                    chunk(j) = shifted(prev_idx);
                }
            }

            // Main chunk
            for (size_t j = 0; j < chunk_len; ++j) {
                chunk(2 * L + j) = shifted(start + j);
            }

            chunks.push_back(chunk);
        }

        // Initialize buffers
        auto bufs = model.init_buffers(1);

        // Process chunks
        std::vector<llvc::Tensor> outputs;
        std::vector<double> times;

        for (size_t i = 0; i < chunks.size(); ++i) {
            // Reshape to [1, 1, T]
            llvc::Tensor input(1, 1, chunks[i].size());
            for (size_t t = 0; t < chunks[i].size(); ++t) {
                input(0, 0, t) = chunks[i](t);
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto [out, new_bufs, dbg] = model.forward_stream(input, bufs);
            bufs = new_bufs;
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(time_ms);

            // Dump first chunk I/O for CSIM
            if (i == 0 && !dump_dir.empty()) {
                if (!fs::exists(dump_dir)) {
                    fs::create_directories(dump_dir);
                }
                // Input/Output
                dump_tensor_to_file((fs::path(dump_dir) / "00_input.txt").string(), input);
                dump_tensor_to_file((fs::path(dump_dir) / "01_in_conv_out.txt").string(), dbg.in_conv_out);
                dump_tensor_to_file((fs::path(dump_dir) / "02_label_emb.txt").string(), dbg.label_emb);

                // MaskNet internals
                dump_tensor_to_file((fs::path(dump_dir) / "03_encoder_out.txt").string(), dbg.masknet.encoder_out);
                dump_tensor_to_file((fs::path(dump_dir) / "04_le.txt").string(), dbg.masknet.le);
                dump_tensor_to_file((fs::path(dump_dir) / "05_proj_e2d_e_out.txt").string(), dbg.masknet.proj_e2d_e_out);
                dump_tensor_to_file((fs::path(dump_dir) / "06_proj_e2d_l_out.txt").string(), dbg.masknet.proj_e2d_l_out);
                dump_tensor_to_file((fs::path(dump_dir) / "06a_sa_out.txt").string(), dbg.masknet.sa_ln_out);  // SA+Residual+LN1 (for HLS)
                dump_tensor_to_file((fs::path(dump_dir) / "06b_ca_out.txt").string(), dbg.masknet.ca_ln_out);  // CA+Residual+LN2 (for HLS)
                dump_tensor_to_file((fs::path(dump_dir) / "07_decoder_out.txt").string(), dbg.masknet.decoder_out);
                dump_tensor_to_file((fs::path(dump_dir) / "08_proj_d2e_out.txt").string(), dbg.masknet.proj_d2e_out);
                dump_tensor_to_file((fs::path(dump_dir) / "09_mask.txt").string(), dbg.masknet.mask);

                // Post-mask
                dump_tensor_to_file((fs::path(dump_dir) / "10_masked.txt").string(), dbg.masked);
                dump_tensor_to_file((fs::path(dump_dir) / "11_with_buf.txt").string(), dbg.with_buf);
                dump_tensor_to_file((fs::path(dump_dir) / "12_output.txt").string(), dbg.out);
            }

            // Convert to 1D
            llvc::Tensor out_1d(out.dim(2));
            for (size_t t = 0; t < out.dim(2); ++t) {
                out_1d(t) = out(0, 0, t);
            }
            outputs.push_back(out_1d);
        }

        // Concatenate outputs
        size_t total_len = 0;
        for (const auto& o : outputs) {
            total_len += o.size();
        }

        output = llvc::Tensor(total_len);
        size_t offset = 0;
        for (const auto& o : outputs) {
            for (size_t i = 0; i < o.size(); ++i) {
                output(offset + i) = o(i);
            }
            offset += o.size();
        }

        // Trim to original length
        if (output.size() > original_len) {
            output = output.slice_last(0, original_len);
        }

        // Calculate metrics
        double avg_time = 0;
        for (double t : times) avg_time += t;
        avg_time /= times.size();

        size_t chunk_len_calc = model.dec_chunk_size() * model.L() * chunk_factor;
        double rtf = (static_cast<double>(chunk_len_calc) / target_sr * 1000.0) / avg_time;
        double e2e_latency = (static_cast<double>(2 * model.L() + chunk_len_calc) / target_sr * 1000.0) + avg_time;

        std::cout << "RTF: " << rtf << std::endl;
        std::cout << "End-to-end latency: " << e2e_latency << " ms" << std::endl;

    } else {
        // Non-streaming inference
        std::cout << "Running inference..." << std::endl;

        // Reshape to [1, 1, T]
        llvc::Tensor input(1, 1, audio.size());
        for (size_t i = 0; i < audio.size(); ++i) {
            input(0, 0, i) = audio(i);
        }

        auto start = std::chrono::high_resolution_clock::now();
        llvc::Tensor out = model.forward(input);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Inference time: " << time_ms << " ms" << std::endl;

        // Convert to 1D
        output = llvc::Tensor(out.dim(2));
        for (size_t i = 0; i < out.dim(2); ++i) {
            output(i) = out(0, 0, i);
        }
    }

    // Save output
    std::cout << "Saving output to " << output_path << "..." << std::endl;
    llvc::write_wav(output_path, output, 16000);
    std::cout << "Done!" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_type = "llvc";
    std::string weights_path;
    bool weights_specified = false;
    std::string input_path = DEFAULT_INPUT_DIR;
    std::string output_path = DEFAULT_OUTPUT_DIR;
    bool streaming = false;
    int chunk_factor = 1;
    std::string dump_dir;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) {
            if (++i < argc) model_type = argv[i];
        } else if (std::strcmp(argv[i], "-w") == 0 || std::strcmp(argv[i], "--weights") == 0) {
            if (++i < argc) {
                weights_path = argv[i];
                weights_specified = true;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 || std::strcmp(argv[i], "--input") == 0) {
            if (++i < argc) input_path = argv[i];
        } else if (std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) {
            if (++i < argc) output_path = argv[i];
        } else if (std::strcmp(argv[i], "-s") == 0 || std::strcmp(argv[i], "--streaming") == 0) {
            streaming = true;
        } else if (std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--chunk-factor") == 0) {
            if (++i < argc) chunk_factor = std::atoi(argv[i]);
        } else if (std::strcmp(argv[i], "-d") == 0 || std::strcmp(argv[i], "--dump") == 0) {
            if (++i < argc) dump_dir = argv[i];
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Validate model type
    if (model_type != "llvc" && model_type != "llvc_nc") {
        std::cerr << "Error: Invalid model type '" << model_type << "'. Must be 'llvc' or 'llvc_nc'." << std::endl;
        return 1;
    }

    // Warn if dump is used without streaming
    if (!dump_dir.empty() && !streaming) {
        std::cerr << "Warning: --dump requires --streaming mode. Enabling streaming mode." << std::endl;
        streaming = true;
    }

    // Set default weights path based on model type if not specified
    if (!weights_specified) {
        weights_path = (model_type == "llvc_nc") ? "models/llvc_nc_weights.bin" : "models/llvc_weights.bin";
    }

    try {
        // Load weights
        std::cout << "Loading weights from " << weights_path << "..." << std::endl;
        llvc::Weights weights;
        weights.load(weights_path);
        std::cout << "Loaded " << weights.size() << " tensors" << std::endl;

        // Create model with config based on model type
        llvc::Net::Config config;
        config.convnet_prenet = (model_type != "llvc_nc");
        std::cout << "Model type: " << model_type << " (convnet_prenet=" << (config.convnet_prenet ? "true" : "false") << ")" << std::endl;
        llvc::Net model(config);
        model.load_weights(weights);

        // Check if input is a directory or file
        bool input_is_dir = fs::is_directory(input_path);
        bool output_is_dir = fs::is_directory(output_path) ||
                             (input_is_dir && !fs::exists(output_path));

        if (input_is_dir) {
            // Process all WAV files in directory
            std::vector<std::string> wav_files = get_wav_files(input_path);

            if (wav_files.empty()) {
                std::cerr << "No WAV files found in " << input_path << std::endl;
                return 1;
            }

            // Create output directory if needed
            if (!fs::exists(output_path)) {
                fs::create_directories(output_path);
                std::cout << "Created output directory: " << output_path << std::endl;
            }

            std::cout << "Processing " << wav_files.size() << " files from " << input_path << "..." << std::endl;
            std::cout << "Output directory: " << output_path << std::endl;
            std::cout << std::string(50, '=') << std::endl;

            for (size_t i = 0; i < wav_files.size(); ++i) {
                const std::string& in_file = wav_files[i];
                std::string filename = fs::path(in_file).filename().string();
                std::string out_file = (fs::path(output_path) / filename).string();

                std::cout << "\n[" << (i + 1) << "/" << wav_files.size() << "] " << filename << std::endl;
                // Only dump for first file
                process_file(model, in_file, out_file, streaming, chunk_factor, (i == 0) ? dump_dir : "");
            }

            std::cout << "\n" << std::string(50, '=') << std::endl;
            std::cout << "All " << wav_files.size() << " files processed successfully!" << std::endl;

        } else {
            // Single file mode
            if (!fs::exists(input_path)) {
                std::cerr << "Input file not found: " << input_path << std::endl;
                return 1;
            }

            // If output is a directory, generate output filename
            std::string final_output = output_path;
            if (output_is_dir) {
                if (!fs::exists(output_path)) {
                    fs::create_directories(output_path);
                }
                std::string filename = fs::path(input_path).filename().string();
                final_output = (fs::path(output_path) / filename).string();
            }

            process_file(model, input_path, final_output, streaming, chunk_factor, dump_dir);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
