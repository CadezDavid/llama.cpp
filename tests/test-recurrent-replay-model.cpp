#include "common.h"
#include "llama-context.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_memory_recurrent * get_recurrent(llama_context * ctx) {
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) return hybrid->get_mem_recr();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) return hybrid->get_mem_recr();
    return dynamic_cast<llama_memory_recurrent *>(mem);
}

static bool truncate_attention_only(llama_context * ctx, llama_pos pos) {
    llama_memory_i * mem = ctx->get_memory();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_attn()->seq_rm(0, pos, -1);
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_attn()->seq_rm(0, pos, -1);
    }
    return true;
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos pos0) {
    if (tokens.empty()) return true;
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], pos0 + i, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static llama_token argmax_token(llama_context * ctx, int n_vocab, int32_t idx = -1) {
    const float * logits = llama_get_logits_ith(ctx, idx);
    GGML_ASSERT(logits);
    return std::max_element(logits, logits + n_vocab) - logits;
}

static bool save_state(llama_context * ctx, std::vector<uint8_t> & data) {
    data.resize(llama_state_get_size(ctx));
    return llama_state_get_data(ctx, data.data(), data.size()) == data.size();
}

static bool load_state(llama_context * ctx, const std::vector<uint8_t> & data) {
    return llama_state_set_data(ctx, data.data(), data.size()) == data.size();
}

struct state_comparison {
    bool r_exact = true;
    bool s_exact = true;
    size_t compared_bytes = 0;
};

static state_comparison exact_recurrent_state(
        llama_memory_recurrent * a, llama_memory_recurrent * b) {
    state_comparison result;
    for (size_t il = 0; il < a->s_l.size(); ++il) {
        if (!a->s_l[il]) continue;
        for (const auto & pair : { std::pair(a->r_l[il], b->r_l[il]),
                                  std::pair(a->s_l[il], b->s_l[il]) }) {
            const size_t bytes = ggml_row_size(pair.first->type, pair.first->ne[0]);
            std::vector<uint8_t> av(bytes);
            std::vector<uint8_t> bv(bytes);
            const size_t a_offset = (size_t) a->rs_idx[0] * a->size * pair.first->nb[1];
            const size_t b_offset = (size_t) b->rs_idx[0] * b->size * pair.second->nb[1];
            ggml_backend_tensor_get(pair.first, av.data(), a_offset, bytes);
            ggml_backend_tensor_get(pair.second, bv.data(), b_offset, bytes);
            if (pair.first == a->r_l[il]) result.r_exact &= av == bv;
            else result.s_exact &= av == bv;
            result.compared_bytes += bytes;
        }
    }
    return result;
}

static uint64_t hash_tokens(const std::vector<llama_token> & tokens) {
    uint64_t hash = 1469598103934665603ULL;
    for (llama_token token : tokens) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash ^= (uint32_t(token) >> shift) & 0xff;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_gpu_layers = 999;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-ngl") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s -m model.gguf [-ngl N]\n", argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-ngl N]\n", argv[0]);
        return 1;
    }

    common_init();
    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams));
    if (!model || !llama_model_is_hybrid(model.get())) return 1;

    auto make_context = [&](uint32_t snapshots, uint32_t stride) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 4096;
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
        cparams.n_seq_max = 1;
        cparams.n_rs_seq = snapshots;
        cparams.n_rs_undo = 20;
        cparams.n_rs_stride = stride;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return llama_context_ptr(llama_init_from_model(model.get(), cparams));
    };
    llama_context_ptr reference = make_context(20, 1);
    llama_context_ptr replay = make_context(5, 4);
    if (!reference || !replay || !get_recurrent(reference.get()) || !get_recurrent(replay.get())) return 1;

    const llama_chat_message messages[] = {
        { "system", "You are a precise technical assistant." },
        { "user", "Explain how to brace a small greenhouse in a windy coastal garden." },
    };
    const char * tmpl = llama_model_chat_template(model.get(), nullptr);
    int32_t formatted_size = llama_chat_apply_template(tmpl, messages, 2, true, nullptr, 0);
    std::vector<char> formatted(formatted_size + 1);
    formatted_size = llama_chat_apply_template(tmpl, messages, 2, true, formatted.data(), formatted.size());
    const auto prompt = common_tokenize(reference.get(), std::string(formatted.data(), formatted_size), true, true);
    if (!decode(reference.get(), prompt, 0) || !decode(replay.get(), prompt, 0)) return 1;

    std::vector<uint8_t> saved_reference;
    std::vector<uint8_t> saved_replay;
    if (!save_state(reference.get(), saved_reference) || !save_state(replay.get(), saved_replay)) return 1;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::vector<llama_token> block;
    for (int i = 0; i < 20; ++i) {
        const llama_token token = argmax_token(reference.get(), n_vocab);
        block.push_back(token);
        if (!decode(reference.get(), { token }, prompt.size() + i)) return 1;
    }

    bool passed = true;
    for (uint32_t valid = 0; valid <= 20; ++valid) {
        if (!load_state(reference.get(), saved_reference) || !load_state(replay.get(), saved_replay)) return 1;
        if (!llama_memory_checkpoint_recurrent(reference.get(), 0) ||
                !llama_memory_checkpoint_recurrent_pass(replay.get(), 0) ||
                !decode(reference.get(), block, prompt.size()) ||
                !decode(replay.get(), block, prompt.size())) return 1;

        if (valid == 0) {
            if (!llama_memory_restore_recurrent(reference.get(), 0) ||
                    !truncate_attention_only(reference.get(), prompt.size())) return 1;
        } else if (valid < block.size()) {
            if (!reference->get_memory()->seq_rm(0, prompt.size() + valid, -1)) return 1;
        }

        llama_recurrent_replay_stats stats = {};
        if (!llama_memory_restore_recurrent_prefix(
                    replay.get(), 0, prompt.size(), valid, block.size(), &stats)) return 1;
        llama_synchronize(reference.get());
        llama_synchronize(replay.get());

        const state_comparison state = exact_recurrent_state(
                get_recurrent(reference.get()), get_recurrent(replay.get()));
        const bool state_exact = state.r_exact && state.s_exact;
        const bool pos_exact = llama_memory_seq_pos_max(llama_get_memory(reference.get()), 0) ==
                llama_memory_seq_pos_max(llama_get_memory(replay.get()), 0);

        const llama_token bridge = valid < block.size() ? block[valid] : argmax_token(reference.get(), n_vocab);
        if (!decode(reference.get(), { bridge }, prompt.size() + valid) ||
                !decode(replay.get(), { bridge }, prompt.size() + valid)) return 1;
        std::vector<llama_token> ref_out = { bridge };
        std::vector<llama_token> replay_out = { bridge };
        bool logits_exact = true;
        for (int step = 0; step < 4; ++step) {
            const float * ref_logits = llama_get_logits_ith(reference.get(), -1);
            const float * replay_logits = llama_get_logits_ith(replay.get(), -1);
            logits_exact &= memcmp(ref_logits, replay_logits, n_vocab * sizeof(float)) == 0;
            const llama_token rt = argmax_token(reference.get(), n_vocab);
            const llama_token pt = argmax_token(replay.get(), n_vocab);
            ref_out.push_back(rt);
            replay_out.push_back(pt);
            if (!decode(reference.get(), { rt }, prompt.size() + valid + 1 + step) ||
                    !decode(replay.get(), { pt }, prompt.size() + valid + 1 + step)) return 1;
        }
        const bool tokens_exact = ref_out == replay_out;
        const bool hashes_exact = hash_tokens(ref_out) == hash_tokens(replay_out);
        printf("valid=%2u checkpoint=%u replay=%u copy_ms=%.4f gdn_ms=%.4f conv_ms=%.4f "
               "state_bytes=%zu r_exact=%d s_exact=%d pos_exact=%d logits_exact=%d tokens_exact=%d hash_exact=%d\n",
                valid, stats.checkpoint_group, stats.replayed_updates, stats.checkpoint_ms,
                stats.gated_delta_ms, stats.convolution_ms, state.compared_bytes,
                state.r_exact, state.s_exact, pos_exact, logits_exact, tokens_exact, hashes_exact);
        passed &= state_exact && pos_exact && logits_exact && tokens_exact && hashes_exact;
    }
    return passed ? 0 : 1;
}
