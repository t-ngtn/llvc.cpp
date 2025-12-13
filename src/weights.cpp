#include "llvc/weights.hpp"
#include <fstream>
#include <stdexcept>

namespace llvc {

void Weights::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + path);
    }

    // Read magic
    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "LLVC") {
        throw std::runtime_error("Invalid weights file: missing LLVC magic");
    }

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != 1) {
        throw std::runtime_error("Unsupported weights file version: " + std::to_string(version));
    }

    // Read number of tensors
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), 4);

    // Read tensors
    for (uint32_t i = 0; i < num_tensors; ++i) {
        // Read name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        // Read shape
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), 4);
        std::vector<size_t> shape(ndim);
        size_t total_size = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), 4);
            shape[d] = dim;
            total_size *= dim;
        }

        // Read data
        std::vector<float> data(total_size);
        file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));

        // Create tensor
        Tensor tensor(shape, data.data());
        tensors_[name] = std::move(tensor);
    }
}

const Tensor& Weights::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

std::vector<std::string> Weights::names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        result.push_back(pair.first);
    }
    return result;
}

} // namespace llvc
