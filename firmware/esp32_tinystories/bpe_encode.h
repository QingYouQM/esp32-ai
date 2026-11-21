// Byte-level BPE encoder: text -> token ids, on the board.
//
// The firmware used to only decode (id -> bytes) because the prompt was baked
// in. Serving an HTTP API means the board has to tokenize whatever opening the
// caller sent, so generated/bpe_encode.h carries the merge list and this is the
// matching encoder. It is kept bit-identical to tokenizers' ByteLevel BPE: the
// host tool tools/check_bpe_parity.py compares the two on real sentences.
//
// Merge selection follows the reference: every step takes the adjacent pair
// with the lowest merge rank and collapses all of its occurrences. Prompts are
// short (a few dozen tokens), so a linear scan per step is cheaper than a heap.
#ifndef BPE_ENCODER_H
#define BPE_ENCODER_H

#include <stdint.h>
#include "generated/bpe_encode.h"

static inline int bpe_rank_of(uint16_t a, uint16_t b) {
  uint32_t key = ((uint32_t)a << 16) | b;
  int lo = 0, hi = BPE_MERGE_N - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    if (BPE_KEY[mid] == key) return mid;
    if (BPE_KEY[mid] < key) lo = mid + 1; else hi = mid - 1;
  }
  return -1;
}

// Returns the number of ids written, or -1 when the text needs more than
// `cap` ids (caller decides whether to truncate or refuse).
static inline int bpe_encode(const uint8_t *text, int len, uint16_t *out, int cap) {
  uint16_t sym[256];
  int n = 0;
  for (int i = 0; i < len && n < (int)(sizeof(sym) / sizeof(sym[0])); i++)
    sym[n++] = BPE_BYTE_ID[text[i]];

  while (n > 1) {
    int best = -1, best_rank = 0x7fffffff;
    for (int i = 0; i + 1 < n; i++) {
      int idx = bpe_rank_of(sym[i], sym[i + 1]);
      if (idx >= 0 && BPE_RANK[idx] < best_rank) {
        best_rank = BPE_RANK[idx];
        best = i;
      }
    }
    if (best < 0) break;

    uint16_t a = sym[best], b = sym[best + 1];
    uint16_t merged = BPE_NEW[bpe_rank_of(a, b)];
    int w = 0;
    for (int i = 0; i < n; i++) {
      if (i + 1 < n && sym[i] == a && sym[i + 1] == b) {
        sym[w++] = merged;
        i++;
      } else {
        sym[w++] = sym[i];
      }
    }
    n = w;
  }

  if (n > cap) return -1;
  for (int i = 0; i < n; i++) out[i] = sym[i];
  return n;
}

#endif
