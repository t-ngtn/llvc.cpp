#include "llvc/audio.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <algorithm>

namespace llvc {

// Read WAV file and return audio samples as 1D tensor (normalized to [-1, 1])
Tensor read_wav(const std::string& path, uint32_t* out_sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }

    // Read RIFF header
    char riff[4];
    file.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("Invalid WAV file: missing RIFF header");
    }

    // Skip file size
    file.seekg(4, std::ios::cur);

    // Read WAVE
    char wave[4];
    file.read(wave, 4);
    if (std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("Invalid WAV file: missing WAVE header");
    }

    // Find and read fmt chunk
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;

    while (file) {
        char chunk_id[4];
        uint32_t chunk_size;
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::string(chunk_id, 4) == "fmt ") {
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            file.seekg(4, std::ios::cur); // byte_rate
            file.seekg(2, std::ios::cur); // block_align
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            // Skip rest of fmt chunk if any
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (std::string(chunk_id, 4) == "data") {
            // Found data chunk
            if (out_sample_rate) {
                *out_sample_rate = sample_rate;
            }

            // Only support 16-bit PCM
            if (audio_format != 1 || bits_per_sample != 16) {
                throw std::runtime_error("Only 16-bit PCM WAV files are supported");
            }

            size_t num_samples = chunk_size / (num_channels * sizeof(int16_t));
            Tensor audio(num_samples);

            // Read samples
            std::vector<int16_t> buffer(num_samples * num_channels);
            file.read(reinterpret_cast<char*>(buffer.data()), chunk_size);

            // Convert to float and mix to mono if needed
            for (size_t i = 0; i < num_samples; ++i) {
                float sample = 0.0f;
                for (size_t ch = 0; ch < num_channels; ++ch) {
                    sample += static_cast<float>(buffer[i * num_channels + ch]) / 32768.0f;
                }
                audio(i) = sample / num_channels;
            }

            return audio;
        } else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    throw std::runtime_error("No data chunk found in WAV file");
}

// Write audio samples to WAV file (expects normalized [-1, 1] samples)
void write_wav(const std::string& path, const Tensor& audio, uint32_t sample_rate) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create WAV file: " + path);
    }

    size_t num_samples = audio.size();
    uint32_t data_size = num_samples * sizeof(int16_t);

    // Write header
    WavHeader header;
    std::memcpy(header.riff, "RIFF", 4);
    header.file_size = 36 + data_size;
    std::memcpy(header.wave, "WAVE", 4);
    std::memcpy(header.fmt, "fmt ", 4);
    header.fmt_size = 16;
    header.audio_format = 1; // PCM
    header.num_channels = 1; // Mono
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * sizeof(int16_t);
    header.block_align = sizeof(int16_t);
    header.bits_per_sample = 16;
    std::memcpy(header.data, "data", 4);
    header.data_size = data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write audio data
    std::vector<int16_t> buffer(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        float sample = audio(i);
        // Clamp to [-1, 1]
        sample = std::max(-1.0f, std::min(1.0f, sample));
        buffer[i] = static_cast<int16_t>(sample * 32767.0f);
    }
    file.write(reinterpret_cast<const char*>(buffer.data()), data_size);
}

// Resample audio (simple linear interpolation)
Tensor resample(const Tensor& audio, uint32_t from_rate, uint32_t to_rate) {
    if (from_rate == to_rate) {
        return audio.clone();
    }

    size_t old_len = audio.size();
    size_t new_len = static_cast<size_t>(static_cast<double>(old_len) * to_rate / from_rate);

    Tensor result(new_len);
    double ratio = static_cast<double>(from_rate) / to_rate;

    for (size_t i = 0; i < new_len; ++i) {
        double src_idx = i * ratio;
        size_t idx0 = static_cast<size_t>(src_idx);
        size_t idx1 = std::min(idx0 + 1, old_len - 1);
        double frac = src_idx - idx0;

        result(i) = static_cast<float>((1.0 - frac) * audio(idx0) + frac * audio(idx1));
    }

    return result;
}

} // namespace llvc
