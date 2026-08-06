#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

bool ggml_cuda_gated_delta_net_undo(
        ggml_tensor       * state,
        const ggml_tensor * k,
        const ggml_tensor * delta,
        const ggml_tensor * decay,
        int64_t             n_undo,
        float             * elapsed_ms);

bool ggml_cuda_recurrent_conv_undo(
        ggml_tensor       * state,
        const ggml_tensor * evicted,
        int64_t             n_undo,
        float             * elapsed_ms);

bool ggml_cuda_gated_delta_net_replay(
        ggml_tensor       * state,
        const ggml_tensor * k,
        const ggml_tensor * delta,
        const ggml_tensor * gate,
        int64_t             offset,
        int64_t             count,
        float             * elapsed_ms);

bool ggml_cuda_recurrent_conv_replay(
        ggml_tensor       * state,
        const ggml_tensor * inserted,
        int64_t             offset,
        int64_t             count,
        float             * elapsed_ms);
