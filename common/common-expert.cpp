#include "common.h"
#include <sstream>
#include <algorithm>
#include <cctype>

// Parse expert range string like "0-31" or "0,5,10,15" into a set of expert IDs
std::set<int> parse_expert_range(const std::string & range) {
    std::set<int> expert_ids;

    if (range.empty()) {
        return expert_ids;
    }

    std::stringstream ss(range);
    std::string segment;

    // Split by commas
    while (std::getline(ss, segment, ',')) {
        // Trim whitespace
        segment.erase(0, segment.find_first_not_of(" \t"));
        segment.erase(segment.find_last_not_of(" \t") + 1);

        // Check if it's a range (contains '-')
        size_t dash_pos = segment.find('-');
        if (dash_pos != std::string::npos) {
            // It's a range like "0-31"
            int start = std::stoi(segment.substr(0, dash_pos));
            int end = std::stoi(segment.substr(dash_pos + 1));

            if (start <= end) {
                for (int i = start; i <= end; ++i) {
                    expert_ids.insert(i);
                }
            }
        } else {
            // It's a single number
            expert_ids.insert(std::stoi(segment));
        }
    }

    return expert_ids;
}

// Check if a tensor name indicates it's an expert tensor
bool is_expert_tensor(const std::string & name) {
    // Expert tensors have names like "blk.0.ffn_gate_exps", "blk.0.ffn_down_exps", "blk.0.ffn_up_exps"
    return name.find(".ffn_gate_exps") != std::string::npos ||
           name.find(".ffn_down_exps") != std::string::npos ||
           name.find(".ffn_up_exps") != std::string::npos ||
           name.find(".ffn_norm_exps") != std::string::npos ||
           name.find(".ffn_gate_exps_b") != std::string::npos ||
           name.find(".ffn_down_exps_b") != std::string::npos ||
           name.find(".ffn_up_exps_b") != std::string::npos;
}

// Extract expert ID from tensor name
// Note: In llama.cpp, expert tensors are stored in merged form, not per-expert
// So this function actually returns -1 for expert tensors in merged format
// We'll need to handle this differently in the model loader
int parse_expert_id_from_tensor_name(const std::string & name) {
    // Expert tensors in llama.cpp are stored in merged/concatenated form
    // like "blk.0.ffn_gate_exps" which contains ALL experts for that layer
    // So we can't extract a single expert ID from the name
    // Return -1 to indicate this is a merged expert tensor
    if (is_expert_tensor(name)) {
        return -1;  // Indicates merged expert tensor
    }
    return -2;  // Not an expert tensor at all
}
