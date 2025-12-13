#ifndef LLVC_AUDIO_HPP
#define LLVC_AUDIO_HPP

#include "tensor.hpp"
#include <string>
#include <cstdint>

namespace llvc {

// WAV file header structure (44 bytes)
#pragma pack(push, 1)
struct WavHeader {
    char riff[4];           // "RIFF"
    uint32_t file_size;     // File size - 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;      // 16 for PCM
    uint16_t audio_format;  // 1 for PCM
    uint16_t num_channels;  // 1 for mono
    uint32_t sample_rate;   // e.g., 16000
    uint32_t byte_rate;     // sample_rate * num_channels * bits_per_sample / 8
    uint16_t block_align;   // num_channels * bits_per_sample / 8
    uint16_t bits_per_sample; // 16
    char data[4];           // "data"
    uint32_t data_size;     // Number of bytes of audio data
};
#pragma pack(pop)

// Read WAV file and return audio samples as 1D tensor (normalized to [-1, 1])
Tensor read_wav(const std::string& path, uint32_t* out_sample_rate = nullptr);

// Write audio samples to WAV file (expects normalized [-1, 1] samples)
void write_wav(const std::string& path, const Tensor& audio, uint32_t sample_rate);

// Resample audio (simple linear interpolation)
Tensor resample(const Tensor& audio, uint32_t from_rate, uint32_t to_rate);

} // namespace llvc

#endif // LLVC_AUDIO_HPP
