"""Compares the board's C encoder with tokenizers, id for id.

The firmware has to tokenize Chinese on its own once the API accepts an
arbitrary opening, and a silent mismatch would show up as a story that drifts
from what the same prompt produces on the PC. This builds bpe_encode.h with the
host gcc and checks it against tokenizers on a spread of inputs.

    python firmware/esp32_tinystories/tools/check_bpe_parity.py \
        --gcc d:/CodingTrash/tools/w64devkit/w64devkit/bin/gcc.exe
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer

SKETCH = Path(__file__).resolve().parents[1]

SAMPLES = [
    "一天，",
    "小明走进森林，",
    "从前有座山，山里有座庙。",
    "天空忽然乌云密布，闪电划破长空。",
    "hello world",
    "The cat sat on the mat.",
    "  leading spaces",
    "我想吃西瓜🍉",
    "数字 123 和标点！？",
    "夜深了，窗外的雨还在下",
]


def c_literal(s):
    raw = s.encode("utf-8")
    return '"' + "".join("\\x%02x" % b for b in raw) + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(SKETCH.parents[1] / "artifacts/zh/tokenizer.json"))
    ap.add_argument("--gcc", default="gcc")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    src = ["#include <stdio.h>", "#include <string.h>",
           '#include "bpe_encode.h"', ""]
    src.append("static const char *S[] = {" + ",".join(c_literal(s) for s in SAMPLES) + "};")
    src.append(f"static const int NS = {len(SAMPLES)};")
    src.append("""
int main(void) {
  uint16_t ids[256];
  for (int i = 0; i < NS; i++) {
    int n = bpe_encode((const uint8_t *)S[i], (int)strlen(S[i]), ids, 256);
    printf("%d", n);
    for (int k = 0; k < n; k++) printf(" %u", ids[k]);
    printf("\\n");
  }
  return 0;
}""")

    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "parity.c"
        exe = Path(td) / ("parity.exe" if sys.platform == "win32" else "parity")
        c.write_text("\n".join(src), encoding="utf-8")
        subprocess.run([args.gcc, "-O2", "-I", str(SKETCH), "-o", str(exe), str(c)],
                       check=True)
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout

    bad = 0
    for i, line in enumerate(out.strip().splitlines()):
        parts = [int(x) for x in line.split()]
        got = parts[1:1 + parts[0]]
        want = tok.encode(SAMPLES[i]).ids
        if got != want:
            bad += 1
            print(f"MISMATCH {SAMPLES[i]!r}\n  board: {got}\n  pc   : {want}")
        else:
            print(f"ok {SAMPLES[i]!r}  {len(got)} ids")
    if bad:
        raise SystemExit(f"{bad}/{len(SAMPLES)} samples differ")
    print(f"\nall {len(SAMPLES)} samples match tokenizers exactly")


if __name__ == "__main__":
    main()
