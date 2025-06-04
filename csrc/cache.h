#pragma once

#include <torch/all.h>

#include <map>
#include <vector>

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping);

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping);

void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       const double kv_scale);

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype);
void reshape_and_cache_segment(
    torch::Tensor& key,           // [num_tokens, num_heads, head_size]
    torch::Tensor& value,         // [num_tokens, num_heads, head_size]
    int64_t global_memory,        // global memory pointer
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    int64_t layer_id,             // layer id
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t block_size,           // number of slots in a block
    const std::string& kv_cache_dtype, const double kv_scale);

// Just for unittest
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype);
