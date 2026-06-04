"""Vision-unload: load Qwen3.5-4B TEXT-ONLY and keep the ~0.62 GB vision tower
out of RAM until an image actually needs it.

WHY: the 4B is multimodal — `vision_tower.*` (297 tensors, ~0.62 GB of the
2.83 GB checkpoint) only fires on image input. A text/voice assistant pays that
RAM for nothing. Text-only resident ≈ 2.20 GB (fits 6 GB phones comfortably).

ON MLX: `mlx_lm.load()` for model_type `qwen3_5` ALREADY drops the vision tower
(`models/qwen3_5.py::sanitize` skips `vision_tower*` / `model.visual*`). This
module makes that GUARANTEE explicit + verified (asserts no vision params are
resident, reports the real resident weight size), instead of relying on
library-internal behavior that a future mlx-lm bump could change.

ON llama.cpp / mobile: the equivalent is `--no-mmproj` (don't load the
multimodal projector) — see BRAIN_PLAN §6.

ON-DEMAND VISION: when an image arrives, load the full multimodal model via
mlx-vlm separately (see `load_vision_on_demand`), run it for that turn, then
release it. Not wired into the text agent yet (no image input path) — stub +
contract documented below.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm import load

VISION_PREFIXES = ("vision_tower", "visual", "model.visual", "mmproj")


def _iter_named_params(model):
    for name, p in tree_flatten(model.parameters()):
        if isinstance(p, mx.array):
            yield name, p


def resident_weight_bytes(model) -> int:
    """Sum nbytes of every resident parameter array (quantized layers included:
    mlx stores packed uint32 weight + fp16 scales/biases, nbytes counts them)."""
    return sum(p.nbytes for _, p in _iter_named_params(model))


def vision_param_names(model) -> list[str]:
    return [n for n, _ in _iter_named_params(model)
            if any(pref in n for pref in VISION_PREFIXES)]


@dataclass(frozen=True)
class LoadStats:
    params: int
    weight_gb: float
    vision_params: int

    def __str__(self):
        v = "OK (0 resident)" if self.vision_params == 0 else f"⚠️ {self.vision_params} LEAKED"
        # NB: params counts PACKED 4-bit array elems (uint32), not logical 4B.
        return (f"text-only resident: {self.weight_gb:.2f} GB "
                f"({self.params/1e9:.2f}B packed elems), vision={v}")


def load_text_only(model_path: str, *, verify: bool = True):
    """Load the LM text-only. Returns (model, tokenizer, LoadStats).

    With verify=True, raises if any vision tensor slipped into the resident
    model — turning the implicit mlx-lm behavior into a hard guarantee.
    """
    model, tok = load(model_path)
    leaked = vision_param_names(model)
    nbytes = resident_weight_bytes(model)
    nparams = sum(p.size for _, p in _iter_named_params(model))
    stats = LoadStats(params=nparams, weight_gb=nbytes / 1024**3,
                      vision_params=len(leaked))
    if verify and leaked:
        raise RuntimeError(
            f"vision-unload FAILED: {len(leaked)} vision tensors resident "
            f"(e.g. {leaked[:3]}). mlx-lm sanitize may have changed.")
    return model, tok, stats


def load_vision_on_demand(model_path: str):
    """On-demand multimodal load for image turns (lazy, +~0.62 GB while active).

    Contract (when an image input path is added to the agent):
      from mlx_vlm import load as vload, generate as vgen
      vmodel, vproc = vload(model_path)          # full model incl. vision tower
      out = vgen(vmodel, vproc, prompt, image)   # run the image turn
      del vmodel; mx.clear_cache()               # release the 0.62 GB after
    Kept as a stub: the text agent has no image input yet, and pulling mlx-vlm
    in eagerly would defeat the unload. Implement when glasses/camera land.
    """
    raise NotImplementedError(
        "on-demand vision not wired yet (no image input path). "
        "Use mlx-vlm per the docstring contract when adding image turns.")


if __name__ == "__main__":
    BASE = "mlx-community/Qwen3.5-4B-MLX-4bit"
    print(f"loading {BASE} text-only ...", flush=True)
    _, _, stats = load_text_only(BASE)
    print(stats)
    full = 2.83
    print(f"checkpoint on disk = {full:.2f} GB (incl. 0.62 GB vision) -> "
          f"resident {stats.weight_gb:.2f} GB -> saved ~{full - stats.weight_gb:.2f} GB")
