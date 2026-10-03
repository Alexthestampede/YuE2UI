#!/usr/bin/env bash
# YuE2UI installer — ROCm (AMD) focus, falls back to CUDA/NVIDIA pip default.
set -euo pipefail

cd "$(dirname "$0")"

PY=python3.12
if ! command -v "$PY" >/dev/null && [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

echo "==> Python 3.12"
command -v uv >/dev/null || { echo "uv not found — install it: https://docs.astral.sh/uv/ (or install python3.12 + venv manually)"; exit 1; }
[ -d .venv ] || uv venv .venv --python 3.12
source .venv/bin/activate

echo "==> YuE (upstream, needed for the yue2 package + ROCm patches)"
[ -d YuE ] || git clone --depth 1 https://github.com/multimodal-art-projection/YuE.git

echo "==> ROCm detection"
USE_ROCM=0
if command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -q "gfx1[01]"; then
  USE_ROCM=1
  echo "    AMD ROCm GPU found"
elif command -v nvidia-smi >/dev/null; then
  echo "    NVIDIA GPU found"
fi

echo "==> Torch"
if [ "$USE_ROCM" = 1 ]; then
  # Match torch to the host's ROCm major.minor when possible.
  ROCM_VER=$(ls /opt/rocm*/.info/version 2>/dev/null | head -1 | xargs cat 2>/dev/null | cut -d. -f1-2 || true)
  if command -v rpm >/dev/null && rpm -qa 2>/dev/null | grep -o "rocm[0-9.]*" | head -1; then true; fi
  ROCM_WHEEL="${ROCM_VER:-7.1}"
  echo "    installing torch==2.10.0+rocm${ROCM_WHEEL}"
  uv pip install "torch==2.10.0+rocm${ROCM_WHEEL}" --index-url "https://download.pytorch.org/whl/rocm${ROCM_WHEEL}" \
    || uv pip install "torch==2.10.0+rocm7.1" --index-url https://download.pytorch.org/whl/rocm7.1
else
  uv pip install torch==2.10.0
fi

echo "==> yue2 (editable, so ROCm patches below live in real source)"
uv pip install -e ./YuE

echo "==> ROCm patches"
python - <<'EOF'
from pathlib import Path
import sys

cg = Path("YuE/src/yue2/cuda_graph.py")
src = cg.read_text()
patch = '''        if attention_backend == "auto":
            # ROCm: aten._flash_attention_forward rejects seqused_k
            # (mha_varlen_fwd: seqused_k must be nullopt) and the forced
            # CUDNN_ATTENTION backend has no kernel on current ROCm builds.
            # Plain auto-selected SDPA works, including inside graph capture.
            if torch.version.hip:
                attention_backend = "sdpa"
            else:
                attention_backend = "flash" if flash else "cudnn" if fused and torch.backends.cudnn.is_available() else "sdpa"
'''
if 'torch.version.hip' not in src:
    old = '''        if attention_backend == "auto":
            attention_backend = "flash" if flash else "cudnn" if fused and torch.backends.cudnn.is_available() else "sdpa"
'''
    if old not in src:
        print("  ! cuda_graph.py auto-selection not found; YuE upstream may have changed — patch manually", file=sys.stderr)
        sys.exit(1)
    cg.write_text(src.replace(old, patch))
    print("  patched cuda_graph.py (flash/cudnn -> plain SDPA on ROCm)")
else:
    print("  cuda_graph.py already patched")

pl = Path("YuE/src/yue2/pipeline.py")
src = pl.read_text()
if "nar_query_chunk_size" not in src:
    # 1) import os
    if "\nimport os\n" not in src:
        src = src.replace("import dataclasses\nimport json", "import dataclasses\nimport json\nimport os", 1)
    # 2) attribute init after offload_ar
    old = "        self.offload_ar = offload_ar\n"
    new = old + '''        # ROCm: long non-causal SDPA falls back to the math kernel and
        # materializes a full [heads, seq, seq] score matrix (8+ GiB for a
        # ~12k-frame chunk). Tile queries to bound that temporary storage.
        self.nar_query_chunk_size = int(os.environ.get("YUE2_NAR_QUERY_CHUNK", "1024"))
'''
    src = src.replace(old, new, 1)
    # 3) pass through in synthesize call
    old2 = '''                                context=self.generation_config.context, offload_ar=self.offload_ar,
                                cancelled=cancelled, on_progress=report)'''
    new2 = '''                                context=self.generation_config.context, offload_ar=self.offload_ar,
                                query_chunk_size=self.nar_query_chunk_size,
                                cancelled=cancelled, on_progress=report)'''
    if old not in src or old2 not in src:
        print("  ! pipeline.py anchors not found; patch manually", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old2, new2, 1)
    pl.write_text(src)
    print("  patched pipeline.py (NAR query_chunk_size pass-through, prevents 16GiB OOM on long songs)")
else:
    print("  pipeline.py already patched")
EOF

echo "==> UI deps"
uv pip install fastapi uvicorn pydantic

echo "==> Done. Launch with:"
echo "    ./run.sh"