#ifndef LLVC_WEIGHTS_HPP
#define LLVC_WEIGHTS_HPP

#include "tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace llvc {

// Weight storage - maps tensor names to tensors
class Weights {
public:
    Weights() = default;

    // Load weights from binary file
    void load(const std::string& path);

    // Get tensor by name
    const Tensor& get(const std::string& name) const;

    // Check if tensor exists
    bool has(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    // Get number of tensors
    size_t size() const {
        return tensors_.size();
    }

    // Get all tensor names
    std::vector<std::string> names() const;

private:
    std::unordered_map<std::string, Tensor> tensors_;
};

} // namespace llvc

#endif // LLVC_WEIGHTS_HPP
