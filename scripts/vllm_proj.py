"""vLLM patch serving train_vector.py's MODE project hook exactly: at the output of each decoder layer of the window, every position,
h <- h - ((h - mu_i).u) u. RHABLATION_PROJ is a safetensors file with u [d_model] or [k, d_model] (fp32 unit vectors: one shared, or one per window layer), c [n_layers] (mu_i.u per decoder layer) and
layers [k] (decoder layer indices), and optionally gen_only, which hooks model-generated tokens only: a token past its request's prompt length, or inside a finished assistant turn of the
prompt (generated_mask), so a multi-turn history gets the hook it had when it was written. The mask is refilled each step. Unset, nothing is patched. A decoder layer returns (mlp output, residual) and its output is their sum, so the
correction goes on the mlp output. The MTP draft layers are left alone. Installed by box_proj_setup.sh next to Will's resid-bias plugin, whose register() calls register() here."""
import os

import numpy as np
import torch
from safetensors.torch import load_file

IM_START, IM_END, ASSISTANT, THINK_OPEN = 248045, 248046, 74455, 248068


def generated_mask(ids: list[int]) -> np.ndarray:
    """Which tokens the model wrote in the finished assistant turns of ids: after the template's `<|im_start|>assistant\\n` and `<think>\\n`, through `<|im_end|>`. An open turn at the end is the generation prompt, so it has none."""
    mask = np.zeros(len(ids), dtype=bool)
    ends = np.flatnonzero(np.array(ids) == IM_END)
    for start in np.flatnonzero(np.array(ids) == IM_START):
        if start + 1 >= len(ids) or ids[start + 1] != ASSISTANT: continue
        first = start + 3 + 2 * (start + 3 < len(ids) and ids[start + 3] == THINK_OPEN)
        after = ends[ends > start]
        if len(after): mask[first:after[0] + 1] = True
    return mask


def register() -> None:
    path = os.environ.get("RHABLATION_PROJ", "")
    if not path: return
    from vllm.model_executor.models import qwen3_5, qwen3_next
    saved = load_file(path)
    window = saved["layers"].tolist()
    gen_only = "gen_only" in saved
    mask = []   # the one generated-token mask, made by the first hooked layer on the worker
    orig_init = qwen3_5.Qwen3_5DecoderLayer.__init__
    orig_forward = qwen3_next.Qwen3NextDecoderLayer.forward

    def patch_runner():
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.states import RequestState
        orig_prepare = GPUModelRunner.prepare_inputs
        orig_add = RequestState.add_request

        def patched_add(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
            orig_add(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens)
            if not hasattr(self, "proj_generated"): self.proj_generated = torch.zeros(self.max_num_reqs, self.max_model_len, dtype=torch.bool, device=self.device)
            self.proj_generated[self.req_id_to_index[req_id], :prompt_len] = torch.from_numpy(generated_mask(list(all_token_ids[:prompt_len]))).to(self.device)

        def patched_prepare(self, *args, **kwargs):
            batch = orig_prepare(self, *args, **kwargs)
            n = batch.num_tokens
            req = torch.searchsorted(batch.query_start_loc[1:batch.num_reqs + 1], torch.arange(n, device=batch.positions.device, dtype=batch.query_start_loc.dtype), right=True)
            slot = batch.idx_mapping[req]
            positions = batch.positions[:n]
            mask[0][:n] = ((positions >= self.req_states.prompt_len.gpu[slot]) | self.req_states.proj_generated[slot, positions]).float()
            return batch

        RequestState.add_request = patched_add
        GPUModelRunner.prepare_inputs = patched_prepare

    def patched_init(self, vllm_config, layer_type, prefix=""):
        orig_init(self, vllm_config, layer_type, prefix=prefix)
        self.proj_on = self.layer_idx in window and "mtp" not in prefix
        if self.proj_on:
            device = torch.empty(0).device
            self.register_buffer("proj_u", (saved["u"] if saved["u"].ndim == 1 else saved["u"][window.index(self.layer_idx)]).float().to(device), persistent=False)
            self.register_buffer("proj_c", saved["c"][self.layer_idx].float().to(device), persistent=False)
            if gen_only:
                assert vllm_config.use_v2_model_runner, "the mask is filled in the v2 runner's prepare_inputs"
                if not mask:
                    mask.append(torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens + 1024, device=device))
                    patch_runner()
                self.register_buffer("proj_gen", mask[0], persistent=False)
            print(f"vllm_proj: hook on {prefix}, c {self.proj_c.item():+.3f}", flush=True)

    def patched_forward(self, hidden_states, residual, positions, **kwargs):
        hidden_states, residual = orig_forward(self, hidden_states, residual, positions, **kwargs)
        if getattr(self, "proj_on", False):
            s = (hidden_states + residual).float() @ self.proj_u - self.proj_c
            if gen_only: s = s * self.proj_gen[:s.shape[0]]
            hidden_states = hidden_states - (s[:, None] * self.proj_u).to(hidden_states.dtype)
        return hidden_states, residual

    qwen3_5.Qwen3_5DecoderLayer.__init__ = patched_init
    qwen3_next.Qwen3NextDecoderLayer.forward = patched_forward
