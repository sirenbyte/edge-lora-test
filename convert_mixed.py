"""#2 Asymmetric precision: MoE experts at 4-bit, everything else at 8-bit.

From the survey (gpt-oss MXFP4 idea): experts tolerate aggressive 4-bit,
but attention/router/embeddings need higher precision. So we quantize the
fused experts (switch_mlp) at 4-bit and the rest at 8-bit. Experts dominate
the param count, so total size stays close to uniform-4bit but quality of the
sensitive (attention/router) path is preserved.
"""

from mlx_lm import convert


def predicate(path, module, config=None):
    # path looks like "model.layers.5.feed_forward.switch_mlp.gate_proj"
    if "switch_mlp" in path:          # MoE experts -> aggressive 4-bit
        return {"group_size": 64, "bits": 4}
    return True                        # attn / router / conv / embed -> default 8-bit


if __name__ == "__main__":
    convert(
        hf_path="LiquidAI/LFM2.5-8B-A1B",
        mlx_path="./LFM2.5-8B-A1B-mlx-mixed",
        quantize=True,
        q_bits=8,                      # default for non-experts
        q_group_size=64,
        quant_predicate=predicate,
    )
    print("MIXED CONVERT DONE")
