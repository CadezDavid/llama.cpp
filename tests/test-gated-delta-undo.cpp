#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static ggml_backend_dev_t find_cuda_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                strstr(ggml_backend_dev_name(dev), "CUDA") != nullptr) {
            return dev;
        }
    }
    return nullptr;
}

static void fill_random(std::vector<float> & values, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float & value : values) {
        value = dist(rng);
    }
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = find_cuda_device();
    if (!dev) {
        fprintf(stderr, "SKIP: no CUDA device\n");
        return 0;
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend);

    constexpr int64_t S = 128;
    constexpr int64_t H = 48;
    constexpr int64_t H_K = 16;
    constexpr int64_t T = 16;
    constexpr int64_t N = 1;
    constexpr int64_t K = 16;

    ggml_init_params params = {
        /*.mem_size   =*/ 32 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H_K, T, N);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H_K, T, N);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, N);
    ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, N);
    ggml_tensor * b = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, N);
    ggml_tensor * state0 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, S, H, N);
    ggml_tensor * out = ggml_gated_delta_net_ext(ctx.get(), q, k, v, g, b, state0, K, true);

    const size_t attn_bytes = S * H * T * N * sizeof(float);
    const size_t state_bytes = S * S * H * N * sizeof(float);
    const size_t states_bytes = K * state_bytes;
    ggml_tensor * final_state = ggml_view_4d(ctx.get(), out, S, S, H, N,
            S * sizeof(float), S * S * sizeof(float), S * S * H * sizeof(float), attn_bytes);
    ggml_tensor * delta = ggml_view_4d(ctx.get(), out, S, H, T, N,
            S * sizeof(float), S * H * sizeof(float), S * H * T * sizeof(float),
            attn_bytes + states_bytes);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    GGML_ASSERT(buffer);

    std::vector<float> q_h(ggml_nelements(q));
    std::vector<float> k_h(ggml_nelements(k));
    std::vector<float> v_h(ggml_nelements(v));
    std::vector<float> g_h(ggml_nelements(g));
    std::vector<float> b_h(ggml_nelements(b));
    std::vector<float> state0_h(ggml_nelements(state0));
    fill_random(q_h, -0.2f, 0.2f, 1);
    fill_random(k_h, -0.1f, 0.1f, 2);
    fill_random(v_h, -0.3f, 0.3f, 3);
    fill_random(g_h, -0.10f, -0.001f, 4);
    fill_random(b_h, 0.01f, 0.99f, 5);
    fill_random(state0_h, -0.1f, 0.1f, 6);

    ggml_backend_tensor_set(q, q_h.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_h.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_h.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(g, g_h.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, b_h.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(state0, state0_h.data(), 0, ggml_nbytes(state0));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), gf) == GGML_STATUS_SUCCESS);

    std::vector<float> final_h(ggml_nelements(final_state));
    ggml_backend_tensor_get(final_state, final_h.data(), 0, state_bytes);

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto undo = (ggml_backend_gated_delta_net_undo_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_gated_delta_net_undo");
    GGML_ASSERT(undo);

    bool passed = true;
    for (int64_t depth : { 1, 4, 8, 16 }) {
        ggml_backend_tensor_set(final_state, final_h.data(), 0, state_bytes);
        float elapsed_ms = 0.0f;
        GGML_ASSERT(undo(final_state, k, delta, g, depth, &elapsed_ms));

        std::vector<float> actual(final_h.size());
        std::vector<float> expected(final_h.size());
        ggml_backend_tensor_get(final_state, actual.data(), 0, state_bytes);
        if (depth == T) {
            expected = state0_h;
        } else {
            const size_t reference_offset = attn_bytes + depth * state_bytes;
            ggml_backend_tensor_get(out, expected.data(), reference_offset, state_bytes);
        }

        double sq = 0.0;
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        size_t nonfinite = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float err = std::abs(actual[i] - expected[i]);
            max_abs = std::max(max_abs, err);
            max_rel = std::max(max_rel, err / std::max(1e-8f, std::abs(expected[i])));
            sq += double(err) * err;
            nonfinite += !std::isfinite(actual[i]);
        }
        const double rms = std::sqrt(sq / actual.size());
        printf("depth=%2lld kernel_ms=%.4f max_abs=%.9g rms=%.9g max_rel=%.9g nonfinite=%zu\n",
                (long long) depth, elapsed_ms, max_abs, rms, max_rel, nonfinite);
        passed &= nonfinite == 0;
    }

    return passed ? 0 : 1;
}
