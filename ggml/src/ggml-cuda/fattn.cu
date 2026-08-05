#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-turbo.cuh"
#include "fattn-tile.cuh"
#include "turbo-quant.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <unordered_map>

template <ggml_type type>
static __device__ __forceinline__ float sparse_gather_dequant_element(
        const char * __restrict__ row,
        int64_t i0) {
    if constexpr (type == GGML_TYPE_Q4_0) {
        const block_q4_0 * block = (const block_q4_0 *) row + i0 / QK4_0;
        const int j = i0 % QK4_0;
        const int q = j < QK4_0 / 2 ? block->qs[j] & 0x0f : block->qs[j - QK4_0 / 2] >> 4;
        return __half2float(block->d) * (q - 8);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0 * block = (const block_q8_0 *) row + i0 / QK8_0;
        return __half2float(block->d) * block->qs[i0 % QK8_0];
    } else if constexpr (type == GGML_TYPE_TURBO4_0) {
        const block_turbo4_0 * block = (const block_turbo4_0 *) row + i0 / QK_TURBO4;
        const float norm = __half2float(block->norm);
        return turbo4_dequant_element(block, i0 % QK_TURBO4, norm);
    } else {
        static_assert(type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_TURBO4_0,
                "unsupported sparse gather type");
        return 0.0f;
    }
}

template <ggml_type type_K, ggml_type type_V>
static __global__ void gather_dequant_sparse_f16(
        const char * __restrict__ K,
        const char * __restrict__ V,
        const int *  __restrict__ indices,
        half *       __restrict__ K_f16,
        half *       __restrict__ V_f16,
        half *       __restrict__ mask_f16,
        int64_t ne0,
        int64_t n_kv,
        int64_t n_kv_padded,
        int64_t n_head_kv,
        int64_t nb11,
        int64_t nb12,
        int64_t nb21,
        int64_t nb22,
        int32_t n_indices,
        int32_t suffix_start) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_elements = n_head_kv * n_kv_padded * ne0;

    if (i < n_kv_padded) {
        mask_f16[i] = __float2half(i < n_kv ? 0.0f : -INFINITY);
    }
    if (i >= n_elements) {
        return;
    }

    const int64_t i0 = i % ne0;
    const int64_t row = i / ne0;
    const int64_t i_kv = row % n_kv_padded;
    const int64_t i_head = row / n_kv_padded;
    if (i_kv >= n_kv) {
        K_f16[i] = __float2half(0.0f);
        V_f16[i] = __float2half(0.0f);
        return;
    }

    const int64_t i_actual = n_indices == suffix_start ? i_kv :
            (i_kv < n_indices ? indices[i_kv] : suffix_start + i_kv - n_indices);
    const char * K_row = K + i_head * nb12 + i_actual * nb11;
    const char * V_row = V + i_head * nb22 + i_actual * nb21;

    K_f16[i] = __float2half(sparse_gather_dequant_element<type_K>(K_row, i0));
    V_f16[i] = __float2half(sparse_gather_dequant_element<type_V>(V_row, i0));
}

static __global__ void sparse_fattn_compare_outputs(
        const float * __restrict__ reference,
        const float * __restrict__ candidate,
        float * __restrict__ error_sums,
        int * __restrict__ invalid,
        int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float ref = reference[i];
    const float value = candidate[i];
    if (!isfinite(ref) || !isfinite(value)) {
        atomicExch(invalid, 1);
        return;
    }
    const float diff = value - ref;
    atomicAdd(error_sums + 0, diff * diff);
    atomicAdd(error_sums + 1, ref * ref);
}

static void ggml_cuda_flash_attn_ext_sparse_gather(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

static void ggml_cuda_flash_attn_ext_vec(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

static void ggml_cuda_flash_attn_ext_q8_turbo4_f16(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// ---------------------------------------------------------------------------
// turbo4 fused MMA decode dispatch (mirrors the f16 switch helpers, type-parametric).
// Only reached from the gate for turbo4 K==V, D in {128,256}, Q->ne[1] <= 4, turing MMA.
//
// The reachable (ncols1, ncols2) set for Q->ne[1] in {1..4} with GQA-packing is exactly
// {(1,8),(2,8),(4,8),(2,4),(4,4),(4,2),(8,1)} — the 7 compiled instances per D. Each ncols2
// has an explicit dispatcher so ONLY those pairs are instantiated (an unguarded ncols1=8/ncols2
// fallthrough would also instantiate uncompiled cases like (8,4) -> link error).

template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0]; // ncols2 == 8: (1,8),(2,8),(4,8)
    if (Q->ne[1] <= 1) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 1, 8, type_K, type_V>(ctx, dst); return; }
    if (Q->ne[1] <= 2) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 2, 8, type_K, type_V>(ctx, dst); return; }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 8, type_K, type_V>(ctx, dst); // Q->ne[1] in {3,4}
}
template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0]; // ncols2 == 4: (2,4),(4,4)
    if (Q->ne[1] <= 2) { ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 2, 4, type_K, type_V>(ctx, dst); return; }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 4, type_K, type_V>(ctx, dst); // Q->ne[1] in {3,4}
}

template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Mirror the f16 use_gqa_opt computation. Quantized tensors are skipped in the nb%16 loop.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio > 4) {                                  // ncols2 = 8
        ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_8<DKQ, DV, type_K, type_V>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio > 2) {                                  // ncols2 = 4
        ggml_cuda_flash_attn_ext_mma_turbo_dispatch_ncols1_4<DKQ, DV, type_K, type_V>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio > 1) {                                  // ncols2 = 2 -> (4,2)
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 4, 2, type_K, type_V>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8, 1, type_K, type_V>(ctx, dst); // ncols2 = 1 -> (8,1)
}

// Env latch for the fused turbo4 MMA decode path. DEFAULT OFF.
//
// The MMA path is correctness-validated (coherent output, KLD == VEC baseline 0.008396)
// and faster than VEC at every depth (beats rival "buun"), BUT it is NOT bit/token-identical
// to the VEC reference: MMA and VEC accumulate the P·V (VKQ) reduction in f16 with different
// reduction trees (tensor-core fragment order vs per-thread VEC order), so a near-tie greedy
// token can flip (~1 in ~25 tokens on a hard tie). This is the same irreducible f16-order
// difference that exists between the base f16-MMA and f16-VEC kernels — not a regression — but
// it fails strict token-identity. We therefore keep VEC the default and expose the faster MMA
// path as opt-in via GGML_TURBO_MMA_FUSED=1.
static bool ggml_cuda_turbo_mma_fused() {
    static const bool v = []{
        const char * s = getenv("GGML_TURBO_MMA_FUSED");
        return !(s && s[0] == '0');  // default ON (faster GQA-packed MMA, quality-neutral); GGML_TURBO_MMA_FUSED=0 = VEC kill-switch
    }();
    return v;
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        case 640: {
            // Padded turbo KV cache for GLM-4.7 Flash (K head_dim=576 zero-padded to 640).
            // D=640 shared memory (Q storage = ncols*(DKQ/2+4)*4) exceeds hardware limit at ncols1>=4.
            // Cap at ncols1=2 (ncols=32): Q=32*324*4=41KB + KV≈37KB = ~78KB total.
            GGML_ASSERT(V->ne[0] == 512);
            if (Q->ne[1] <= 1) {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 1, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_case<640, 512, 2, 16>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template <ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_gather_f16_case(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * indices = dst->src[5];

    GGML_ASSERT(Q->ne[1] == 1 && Q->ne[3] == 1);
    GGML_ASSERT((Q->ne[0] == 256 || Q->ne[0] == 512) && V->ne[0] == Q->ne[0]);
    GGML_ASSERT(K->type == type_K && V->type == type_V);
    GGML_ASSERT(indices == nullptr || indices->type == GGML_TYPE_I32);

    const int32_t n_indices = indices ? ggml_get_op_params_i32(dst, 4) : 0;
    const int32_t n_kv = indices ? ggml_get_op_params_i32(dst, 5) : K->ne[1];
    const int32_t suffix_start = indices ? ggml_get_op_params_i32(dst, 6) : 0;
    const int64_t n_kv_padded = GGML_PAD((int64_t) n_kv, (int64_t) FATTN_KQ_STRIDE);
    const int64_t n_elements = K->ne[0] * n_kv_padded * K->ne[2];

    ggml_cuda_pool_alloc<half> K_f16(ctx.pool(), n_elements);
    ggml_cuda_pool_alloc<half> V_f16(ctx.pool(), n_elements);
    ggml_cuda_pool_alloc<half> mask_f16(ctx.pool(), n_kv_padded);

    const int threads = 256;
    const int blocks = (int) ((std::max(n_elements, n_kv_padded) + threads - 1) / threads);
    gather_dequant_sparse_f16<type_K, type_V><<<blocks, threads, 0, ctx.stream()>>>(
            (const char *) K->data,
            (const char *) V->data,
            indices == nullptr ? nullptr : (const int *) indices->data,
            K_f16.ptr,
            V_f16.ptr,
            mask_f16.ptr,
            K->ne[0],
            n_kv,
            n_kv_padded,
            K->ne[2],
            K->nb[1],
            K->nb[2],
            V->nb[1],
            V->nb[2],
            n_indices,
            suffix_start);
    CUDA_CHECK(cudaGetLastError());

    ggml_tensor gathered_K = *K;
    gathered_K.type = GGML_TYPE_F16;
    gathered_K.ne[1] = n_kv_padded;
    gathered_K.nb[0] = sizeof(half);
    gathered_K.nb[1] = K->ne[0] * sizeof(half);
    gathered_K.nb[2] = n_kv_padded * gathered_K.nb[1];
    gathered_K.nb[3] = K->ne[2] * gathered_K.nb[2];
    gathered_K.data = K_f16.ptr;
    gathered_K.view_src = nullptr;
    gathered_K.view_offs = 0;

    ggml_tensor gathered_V = *V;
    gathered_V.type = GGML_TYPE_F16;
    gathered_V.ne[1] = n_kv_padded;
    gathered_V.nb[0] = sizeof(half);
    gathered_V.nb[1] = V->ne[0] * sizeof(half);
    gathered_V.nb[2] = n_kv_padded * gathered_V.nb[1];
    gathered_V.nb[3] = V->ne[2] * gathered_V.nb[2];
    gathered_V.data = V_f16.ptr;
    gathered_V.view_src = nullptr;
    gathered_V.view_offs = 0;

    ggml_tensor gathered_mask = {};
    gathered_mask.type = GGML_TYPE_F16;
    gathered_mask.ne[0] = n_kv_padded;
    gathered_mask.ne[1] = 1;
    gathered_mask.ne[2] = 1;
    gathered_mask.ne[3] = 1;
    gathered_mask.nb[0] = sizeof(half);
    gathered_mask.nb[1] = n_kv_padded * sizeof(half);
    gathered_mask.nb[2] = gathered_mask.nb[1];
    gathered_mask.nb[3] = gathered_mask.nb[2];
    gathered_mask.data = mask_f16.ptr;

    ggml_tensor gathered_dst = *dst;
    gathered_dst.src[1] = &gathered_K;
    gathered_dst.src[2] = &gathered_V;
    gathered_dst.src[3] = indices == nullptr ? dst->src[3] : &gathered_mask;
    gathered_dst.src[5] = nullptr;

    ggml_cuda_flash_attn_ext_mma_f16(ctx, &gathered_dst);
}

static void ggml_cuda_flash_attn_ext_q8_turbo4_f16(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_gather_f16_case<GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_sparse_gather(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * indices = dst->src[5];

    GGML_ASSERT(Q->ne[1] == 1 && Q->ne[3] == 1);
    GGML_ASSERT((Q->ne[0] == 256 || Q->ne[0] == 512) && V->ne[0] == Q->ne[0]);
    GGML_ASSERT(indices != nullptr && indices->type == GGML_TYPE_I32);

    // Gather and dequantize only the selected rows, then run the validated f16
    // MMA kernel. Avoiding an intermediate compressed gather removes one full
    // read/write pass and reduces peak temporary storage.
    if (K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0) {
        ggml_cuda_flash_attn_ext_gather_f16_case<GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>(ctx, dst);
        return;
    }
    if (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0) {
        ggml_cuda_flash_attn_ext_gather_f16_case<GGML_TYPE_Q8_0, GGML_TYPE_Q4_0>(ctx, dst);
        return;
    }
    if (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0) {
        ggml_cuda_flash_attn_ext_gather_f16_case<GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
        return;
    }
    if (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TURBO4_0) {
        ggml_cuda_flash_attn_ext_q8_turbo4_f16(ctx, dst);
        return;
    }
    GGML_ABORT("unsupported sparse gather attention: K=%s, V=%s",
            ggml_type_name(K->type), ggml_type_name(V->type));
}

static void ggml_cuda_flash_attn_ext_sparse_mma_q8_turbo4(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(dst->src[5] != nullptr);
    GGML_ASSERT(Q->ne[1] == 1 && Q->ne[3] == 1);
    GGML_ASSERT(K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TURBO4_0);
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (Q->ne[0] == 256) {
        GGML_ASSERT(V->ne[0] == 256 && gqa_ratio == 4);
        ggml_cuda_flash_attn_ext_mma_turbo_case<
                256, 256, 2, 4, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0>(ctx, dst);
        return;
    }

    GGML_ASSERT(Q->ne[0] == 512 && V->ne[0] == 512 && gqa_ratio == 8);
    ggml_cuda_flash_attn_ext_mma_turbo_case<
            512, 512, 1, 8, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0>(ctx, dst);
}

enum class sparse_q8_turbo4_impl : uint8_t {
    direct,
    fused_mma,
    gather_f16,
};

static const char * sparse_q8_turbo4_impl_name(sparse_q8_turbo4_impl impl) {
    switch (impl) {
        case sparse_q8_turbo4_impl::direct:     return "direct";
        case sparse_q8_turbo4_impl::fused_mma:  return "fused_mma";
        case sparse_q8_turbo4_impl::gather_f16: return "gather_f16";
    }
    GGML_ABORT("invalid q8/Turbo4 sparse implementation");
}

static void ggml_cuda_flash_attn_ext_sparse_q8_turbo4_launch(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        sparse_q8_turbo4_impl impl) {
    switch (impl) {
        case sparse_q8_turbo4_impl::direct:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            return;
        case sparse_q8_turbo4_impl::fused_mma:
            ggml_cuda_flash_attn_ext_sparse_mma_q8_turbo4(ctx, dst);
            return;
        case sparse_q8_turbo4_impl::gather_f16:
            ggml_cuda_flash_attn_ext_sparse_gather(ctx, dst);
            return;
    }
    GGML_ABORT("invalid q8/Turbo4 sparse implementation");
}

struct sparse_q8_turbo4_tune_key {
    int device;
    int cc;
    int dk;
    int dv;
    int n_head_kv;
    int gqa_ratio;
    int64_t context_bucket;
    int64_t selected_bucket;
    int layout;

    bool operator==(const sparse_q8_turbo4_tune_key & other) const {
        return device == other.device && cc == other.cc && dk == other.dk && dv == other.dv &&
                n_head_kv == other.n_head_kv && gqa_ratio == other.gqa_ratio &&
                context_bucket == other.context_bucket && selected_bucket == other.selected_bucket &&
                layout == other.layout;
    }
};

struct sparse_q8_turbo4_tune_key_hash {
    size_t operator()(const sparse_q8_turbo4_tune_key & key) const {
        size_t h = 0xcbf29ce484222325ULL;
        const auto mix = [&h](uint64_t value) {
            h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix((uint64_t) key.device);
        mix((uint64_t) key.cc);
        mix((uint64_t) key.dk);
        mix((uint64_t) key.dv);
        mix((uint64_t) key.n_head_kv);
        mix((uint64_t) key.gqa_ratio);
        mix((uint64_t) key.context_bucket);
        mix((uint64_t) key.selected_bucket);
        mix((uint64_t) key.layout);
        return h;
    }
};

struct sparse_q8_turbo4_tune_result {
    sparse_q8_turbo4_impl selected = sparse_q8_turbo4_impl::fused_mma;
    uint64_t cache_hits = 0;
};

static int64_t sparse_fattn_power_of_two_bucket(int64_t value) {
    int64_t bucket = 1;
    while (bucket < value && bucket <= INT64_MAX / 2) {
        bucket *= 2;
    }
    return bucket;
}

static sparse_q8_turbo4_tune_key sparse_q8_turbo4_make_tune_key(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const int n_indices = ggml_get_op_params_i32(dst, 4);
    const int n_kv = ggml_get_op_params_i32(dst, 5);
    const int suffix_start = ggml_get_op_params_i32(dst, 6);
    const int layout = n_indices == suffix_start && n_kv == K->ne[1] ? 2 :
            (n_kv > n_indices ? 1 : 0);
    const int device = ggml_cuda_get_device();

    return {
        device,
        ggml_cuda_info().devices[device].cc,
        (int) Q->ne[0],
        (int) dst->src[2]->ne[0],
        (int) K->ne[2],
        (int) (Q->ne[2] / K->ne[2]),
        sparse_fattn_power_of_two_bucket(K->ne[1]),
        sparse_fattn_power_of_two_bucket(n_kv),
        layout,
    };
}

static bool sparse_q8_turbo4_forced_impl(sparse_q8_turbo4_impl & impl) {
    const char * value = getenv("GGML_VEGAS_SPARSE_IMPL");
    if (value == nullptr || value[0] == '\0' || strcmp(value, "auto") == 0) {
        return false;
    }
    if (strcmp(value, "direct") == 0) {
        impl = sparse_q8_turbo4_impl::direct;
        return true;
    }
    if (strcmp(value, "fused") == 0 || strcmp(value, "fused_mma") == 0) {
        impl = sparse_q8_turbo4_impl::fused_mma;
        return true;
    }
    if (strcmp(value, "gather") == 0 || strcmp(value, "gather_f16") == 0) {
        impl = sparse_q8_turbo4_impl::gather_f16;
        return true;
    }

    static std::once_flag warning;
    std::call_once(warning, [value]() {
        GGML_LOG_WARN("GGML_VEGAS_SPARSE_IMPL=%s is invalid; using cached autotuning\n", value);
    });
    return false;
}

static float sparse_q8_turbo4_validate_candidate(
        ggml_backend_cuda_context & ctx,
        const float * reference,
        const float * candidate,
        int64_t n_elements,
        bool & valid) {
    ggml_cuda_pool_alloc<float> error_sums(ctx.pool(), 2);
    ggml_cuda_pool_alloc<int> invalid(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(error_sums.ptr, 0, 2 * sizeof(float), ctx.stream()));
    CUDA_CHECK(cudaMemsetAsync(invalid.ptr, 0, sizeof(int), ctx.stream()));

    constexpr int threads = 256;
    const int blocks = (int) ((n_elements + threads - 1) / threads);
    sparse_fattn_compare_outputs<<<blocks, threads, 0, ctx.stream()>>>(
            reference, candidate, error_sums.ptr, invalid.ptr, n_elements);
    CUDA_CHECK(cudaGetLastError());

    float host_sums[2] = {0.0f, 0.0f};
    int host_invalid = 0;
    CUDA_CHECK(cudaMemcpyAsync(host_sums, error_sums.ptr, sizeof(host_sums), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(&host_invalid, invalid.ptr, sizeof(host_invalid), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    const float nmse = host_sums[0] / std::max(host_sums[1], 1e-20f);
    valid = host_invalid == 0 && std::isfinite(nmse) && nmse <= 5e-4f;
    return nmse;
}

static float sparse_q8_turbo4_measure_candidate(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        sparse_q8_turbo4_impl impl) {
    constexpr int warmups = 2;
    constexpr int measurements = 9;
    for (int i = 0; i < warmups; ++i) {
        ggml_cuda_flash_attn_ext_sparse_q8_turbo4_launch(ctx, dst, impl);
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::array<float, measurements> elapsed_ms;
    for (int i = 0; i < measurements; ++i) {
        CUDA_CHECK(cudaEventRecord(start, ctx.stream()));
        ggml_cuda_flash_attn_ext_sparse_q8_turbo4_launch(ctx, dst, impl);
        CUDA_CHECK(cudaEventRecord(stop, ctx.stream()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(elapsed_ms.begin(), elapsed_ms.end());
    return elapsed_ms[measurements / 2];
}

static sparse_q8_turbo4_impl sparse_q8_turbo4_autotune(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    sparse_q8_turbo4_impl forced;
    if (sparse_q8_turbo4_forced_impl(forced)) {
        return forced;
    }

    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        static std::once_flag warning;
        std::call_once(warning, []() {
            GGML_LOG_WARN("Vegas q8/Turbo4 autotune miss during CUDA graph capture; using fused MMA\n");
        });
        return sparse_q8_turbo4_impl::fused_mma;
    }

    static std::mutex cache_mutex;
    static std::unordered_map<
            sparse_q8_turbo4_tune_key,
            sparse_q8_turbo4_tune_result,
            sparse_q8_turbo4_tune_key_hash> cache;

    const sparse_q8_turbo4_tune_key key = sparse_q8_turbo4_make_tune_key(dst);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto cached = cache.find(key);
    if (cached != cache.end()) {
        ++cached->second.cache_hits;
        return cached->second.selected;
    }

    const auto calibration_start = std::chrono::steady_clock::now();
    const int64_t n_elements = ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> reference(ctx.pool(), n_elements);
    ggml_cuda_pool_alloc<float> candidate(ctx.pool(), n_elements);
    ggml_tensor reference_dst = *dst;
    ggml_tensor candidate_dst = *dst;
    reference_dst.data = reference.ptr;
    candidate_dst.data = candidate.ptr;

    ggml_cuda_flash_attn_ext_sparse_gather(ctx, &reference_dst);
    bool reference_valid = false;
    const float reference_nmse = sparse_q8_turbo4_validate_candidate(
            ctx, reference.ptr, reference.ptr, n_elements, reference_valid);
    if (!reference_valid) {
        GGML_ABORT("q8/Turbo4 sparse gather reference failed autotune validation");
    }

    constexpr std::array<sparse_q8_turbo4_impl, 3> implementations = {
        sparse_q8_turbo4_impl::direct,
        sparse_q8_turbo4_impl::fused_mma,
        sparse_q8_turbo4_impl::gather_f16,
    };
    std::array<float, implementations.size()> nmse = {INFINITY, INFINITY, reference_nmse};
    std::array<float, implementations.size()> median_ms = {INFINITY, INFINITY, INFINITY};
    std::array<bool, implementations.size()> valid = {false, false, true};

    for (size_t i = 0; i < implementations.size(); ++i) {
        const sparse_q8_turbo4_impl impl = implementations[i];
        if (impl != sparse_q8_turbo4_impl::gather_f16) {
            ggml_cuda_flash_attn_ext_sparse_q8_turbo4_launch(ctx, &candidate_dst, impl);
            nmse[i] = sparse_q8_turbo4_validate_candidate(
                    ctx, reference.ptr, candidate.ptr, n_elements, valid[i]);
        }
        if (valid[i]) {
            median_ms[i] = sparse_q8_turbo4_measure_candidate(ctx, &candidate_dst, impl);
        } else {
            GGML_LOG_WARN("Vegas sparse autotune rejected %s: nmse=%g\n",
                    sparse_q8_turbo4_impl_name(impl), (double) nmse[i]);
        }
    }

    size_t fastest = implementations.size();
    for (size_t i = 0; i < implementations.size(); ++i) {
        if (valid[i] && (fastest == implementations.size() || median_ms[i] < median_ms[fastest])) {
            fastest = i;
        }
    }
    if (fastest == implementations.size()) {
        GGML_ABORT("all q8/Turbo4 sparse attention candidates failed autotune validation");
    }

    const int64_t n_kv = ggml_get_op_params_i32(dst, 5);
    const int64_t n_kv_padded = GGML_PAD(n_kv, (int64_t) FATTN_KQ_STRIDE);
    const size_t gather_scratch = (size_t) (dst->src[1]->ne[0] + dst->src[2]->ne[0]) *
            n_kv_padded * dst->src[1]->ne[2] * sizeof(half) + n_kv_padded * sizeof(half);
    const std::array<size_t, implementations.size()> scratch_bytes = {0, 0, gather_scratch};
    const std::array<int, implementations.size()> tie_priority = {1, 0, 2};

    size_t selected = fastest;
    for (size_t i = 0; i < implementations.size(); ++i) {
        if (!valid[i] || median_ms[i] > 1.03f * median_ms[fastest]) {
            continue;
        }
        if (scratch_bytes[i] < scratch_bytes[selected] ||
                (scratch_bytes[i] == scratch_bytes[selected] && tie_priority[i] < tie_priority[selected])) {
            selected = i;
        }
    }

    const auto calibration_stop = std::chrono::steady_clock::now();
    const double calibration_ms = std::chrono::duration<double, std::milli>(
            calibration_stop - calibration_start).count();
    const sparse_q8_turbo4_impl selected_impl = implementations[selected];
    cache.emplace(key, sparse_q8_turbo4_tune_result{selected_impl, 0});

    GGML_LOG_INFO(
            "vegas_sparse_autotune device=%d cc=%d d=%d gqa=%d context_bucket=%lld selected_bucket=%lld "
            "layout=%d direct_ms=%.6f direct_nmse=%g fused_ms=%.6f fused_nmse=%g "
            "gather_ms=%.6f selected=%s calibration_ms=%.3f scratch_bytes=%zu\n",
            key.device, key.cc, key.dk, key.gqa_ratio,
            (long long) key.context_bucket, (long long) key.selected_bucket, key.layout,
            (double) median_ms[0], (double) nmse[0],
            (double) median_ms[1], (double) nmse[1],
            (double) median_ms[2], sparse_q8_turbo4_impl_name(selected_impl),
            calibration_ms, scratch_bytes[selected]);
    return selected_impl;
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    FATTN_VEC_CASE(512, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    // TurboQuant3 KV cache types (always enabled)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)

    // Mixed turbo3/q8_0 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)

    // Mixed f16/turbo3 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,      GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_F16)

    // TurboQuant2 KV cache types (always enabled)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)

    // Mixed turbo2/q8_0 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)

    // Mixed f16/turbo2 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,      GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_F16)

    // Mixed turbo3/turbo2 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)

    // TurboQuant4 KV cache types (always enabled)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)

    // Mixed turbo4/q8_0 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)

    // Mixed f16/turbo4 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,      GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_F16)

    // Mixed turbo4/turbo3 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)

    // Mixed turbo4/turbo2 KV cache types
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0)

    GGML_ABORT("unsupported vector flash attention: D=%lld, K=%s, V=%s",
            (long long) Q->ne[0], ggml_type_name(K->type), ggml_type_name(V->type));
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            // turbo KV types; head-dim geometry is validated separately in
            // ggml_cuda_get_best_fattn_kernel (multiples of 64 only)
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(
        const int device, const ggml_tensor * dst, ggml_backend_cuda_context * ctx = nullptr) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
        case 640:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        const bool q8_q4 = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
        if (!q8_q4) {
            // Allow mixed KV types for combinations that have FA template instances compiled in:
            // - turbo2/3/4 + q8_0 (turbo cache work)
            // - f16/bf16 + q8_0 (common K=f16, V=q8_0 setup)
            auto is_kv_compat = [](ggml_type t) {
                return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0
                    || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
            };
            if (!is_kv_compat(K->type) || !is_kv_compat(V->type)) {
                return BEST_FATTN_KERNEL_NONE;
            }
        }
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // turbo VEC/MMA kernels are instantiated for head dims that are multiples of 64
    {
        auto is_turbo = [](ggml_type t) {
            return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
        };
        if ((is_turbo(K->type) && K->ne[0] % 64 != 0) ||
            (is_turbo(V->type) && V->ne[0] % 64 != 0)) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    const bool d512_vec_types =
        (K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0) ||
        (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0) ||
        (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0) ||
        (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TURBO3_0) ||
        (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TURBO4_0);
    const size_t d512_f16_scratch = (ggml_nelements(K) + ggml_nelements(V)) * sizeof(half);

    // Avoid very large full-cache f16 scratch for quantized D=512 attention batches
    // when the target and draft contexts leave insufficient device memory.
    if (ctx != nullptr && Q->ne[0] == 512 && Q->ne[1] <= 4 && d512_vec_types) {
        size_t free_vram;
        size_t total_vram;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
        const size_t pool_available = ctx->pool().available();
        const size_t scratch_missing = pool_available < d512_f16_scratch ? d512_f16_scratch - pool_available : 0;
        if (scratch_missing != 0 && free_vram < scratch_missing + 256ull * 1024 * 1024) {
            return BEST_FATTN_KERNEL_VEC;
        }
    }

#ifdef GGML_USE_HIP
    // HIP/ROCm: the TILE/MMA/WMMA FA paths allocate large f16 temp buffers for
    // quantized KV types (K_f16, V_f16 in launch_fattn). For SMALL batches (decode)
    // the VEC kernel is preferred: it does inline dequant with zero temp buffer
    // overhead, it natively supports the TurboQuant types, and it produces a
    // HIP-graph-safe op stream (no per-call cudaMalloc/cudaFree during capture).
    // For LARGE batches (prefill) the VEC kernel is far slower (sequential query
    // processing), so we deliberately fall through to the TILE/MMA path which is
    // ~3.4x faster; prefill runs eagerly (not captured) so its f16 temp buffer is
    // allocated/freed raw in launch_fattn without violating graph-capture rules.
    // Limitation: head_dim > 256 cannot use VEC (falls through to TILE).
    if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel && Q->ne[1] <= 8) {
        return BEST_FATTN_KERNEL_VEC;
    }
#endif // GGML_USE_HIP

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 640 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // TQ: RDNA4 fast path for TurboQuant cache types — prefer VEC for quantized K/V at small q-cols
    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff_rdna4 = 1;
        const int ncols2_max_rdna4 = (Q->ne[0] == 576 || Q->ne[0] == 640) ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff_rdna4) == 0 && gqa_ratio_eff_rdna4 < ncols2_max_rdna4) {
            gqa_ratio_eff_rdna4 *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff_rdna4 <= 8) {
            return BEST_FATTN_KERNEL_TILE;
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(device);

    // Temporary f16 K/V buffers are owned by the CUDA pool inside launch_fattn.
    return ggml_nbytes(dst);
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    if (dst->src[5] != nullptr) {
        GGML_ASSERT(dst->src[0]->ne[1] == 1 && dst->src[0]->ne[3] == 1);
        const ggml_sparse_fattn_mode sparse_mode = ggml_flash_attn_ext_get_sparse_mode(dst);
        const bool q8_turbo4 = dst->src[1]->type == GGML_TYPE_Q8_0 &&
                dst->src[2]->type == GGML_TYPE_TURBO4_0;
        if (sparse_mode == GGML_SPARSE_FATTN_MODE_GATHER) {
            ggml_cuda_flash_attn_ext_sparse_gather(ctx, dst);
            return;
        }
        if (sparse_mode == GGML_SPARSE_FATTN_MODE_AUTO && q8_turbo4) {
            const sparse_q8_turbo4_impl impl = sparse_q8_turbo4_autotune(ctx, dst);
            ggml_cuda_flash_attn_ext_sparse_q8_turbo4_launch(ctx, dst, impl);
            return;
        }
        ggml_cuda_flash_attn_ext_vec(ctx, dst);
        return;
    }

    if (dst->src[0]->ne[0] == 256 && dst->src[0]->ne[1] == 1 && dst->src[0]->ne[3] == 1 &&
            dst->src[1]->type == GGML_TYPE_Q8_0 && dst->src[2]->type == GGML_TYPE_TURBO4_0) {
        ggml_cuda_flash_attn_ext_q8_turbo4_f16(ctx, dst);
        return;
    }

    // Fused turbo MMA decode gate (DEFAULT ON — see ggml_cuda_turbo_mma_fused; GGML_TURBO_MMA_FUSED=0 disables).
    // Routes turbo4-K==turbo4-V, D in {128,256}, decode (Q->ne[1] <= 4) onto the GQA-packed
    // MMA path (KV read once per head-group instead of per query head). Q is ALREADY
    // graph-rotated (src/llama-graph.cpp) and the FA output is inverse-rotated there — this
    // path does NO inline FWHT and NO src swap. Default OFF (env unset / !=1) falls straight
    // GGML_TURBO_MMA_FUSED=0 falls straight through to the original VEC dispatch (kill-switch).
    {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const bool turbo_matched = (K->type == V->type &&
            (K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO2_0));
        if (ggml_cuda_turbo_mma_fused() && turbo_matched
                && Q->ne[1] <= 4 && V->ne[0] == Q->ne[0] && turing_mma_available(cc)) {
            if (Q->ne[0] == 128) {
                switch (K->type) {
                    case GGML_TYPE_TURBO4_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0>(ctx, dst); return;
                    case GGML_TYPE_TURBO3_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0>(ctx, dst); return;
                    case GGML_TYPE_TURBO2_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0>(ctx, dst); return;
                    default: break;
                }
            }
            if (Q->ne[0] == 256) {
                switch (K->type) {
                    case GGML_TYPE_TURBO4_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0>(ctx, dst); return;
                    case GGML_TYPE_TURBO3_0: ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0>(ctx, dst); return;
                    // turbo2 + head_dim 256: intentionally NO fused case (routes to VEC via
                    // default below). At 2-bit KV the fused path's GQA-pack saving is tiny while the
                    // dequant/no-pipeline overhead is unchanged, so it is neutral on high-BW GPUs and
                    // regresses ~1-2.5% on bandwidth-limited ones (tester @everson: Gemma-12B / RTX
                    // 5060 Ti). VEC == baseline there. turbo2 + hd128 keeps fused (a +6.6..+69% depth
                    // win on dense models); turbo3/turbo4 stay fused at both head dims.
                    default: break;
                }
            }
        }
    }

    const bool collect_score = dst->src[6] != nullptr;
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst, &ctx)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            GGML_ASSERT(!collect_score);
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    if (dst->src[5] != nullptr) {
        const ggml_sparse_fattn_mode sparse_mode = ggml_flash_attn_ext_get_sparse_mode(dst);
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const bool q8_turbo4 = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TURBO4_0;
        const bool gather_types = q8_turbo4 ||
                (K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0) ||
                (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0) ||
                (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0);
        if (sparse_mode == GGML_SPARSE_FATTN_MODE_GATHER) {
            return (Q->ne[0] == 256 || Q->ne[0] == 512) && V->ne[0] == Q->ne[0] &&
                    Q->ne[1] == 1 && Q->ne[3] == 1 && gather_types &&
                    turing_mma_available(ggml_cuda_info().devices[device].cc);
        }
        if (sparse_mode == GGML_SPARSE_FATTN_MODE_AUTO && q8_turbo4) {
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            return ((Q->ne[0] == 256 && gqa_ratio == 4) ||
                    (Q->ne[0] == 512 && gqa_ratio == 8)) && V->ne[0] == Q->ne[0] &&
                    Q->ne[1] == 1 && Q->ne[3] == 1 &&
                    turing_mma_available(ggml_cuda_info().devices[device].cc);
        }
        if ((sparse_mode == GGML_SPARSE_FATTN_MODE_DIRECT ||
                sparse_mode == GGML_SPARSE_FATTN_MODE_AUTO) && q8_turbo4) {
            return (Q->ne[0] == 256 || Q->ne[0] == 512) && V->ne[0] == Q->ne[0] &&
                    Q->ne[1] == 1 && Q->ne[3] == 1;
        }
    }
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
