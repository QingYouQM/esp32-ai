// PLE TinyLM inference on the ESP32-S3.
//
// The 28.9M-param model (14.9MB, 4-bit) lives in a flash 'model' partition,
// memory-mapped. Placement follows reads-per-token rather than what happens to
// fit:
//
//   flash   PLE table + token embedding   one row of the 25.2M-parameter
//                                         table per token
//   PSRAM   staged int8 core + head, KV   read once per position
//   SRAM    scratch + norm vectors        touched many times per token
//
// The logits array stays in PSRAM: 25,353 floats is 99 KiB, and the argmax
// reads it once per token.
//
// Same llm.h that is verified against PyTorch on the host; only the platform
// hooks differ here.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

// int8 activations, required by the staged int8 kernel. Not bit-exact against
// the fp32 golden; verify.c must be built without this flag. Validation CE cost
// (runtime/host_verify/ppl.c, 32,768 predictions): 2.4793 -> 2.4796, ppl 11.93 / 11.94.
#define LLM_INT8_ACT 1
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../../runtime/llm.h"
#include "generated/vocab.h"

// On-device 1.54" ST7789 (expansion board's 8-pin TFT socket). display.h now
// carries a 16x16 CJK font (generated/cjk_font.h), so the story renders on the
// panel as well as over serial. Set to 0 for a serial-only run.
#define USE_DISPLAY 1
#if USE_DISPLAY
#include "display.h"
#endif

// On-device web UI: the board raises its own access point, so no router
// credentials live in the firmware and the demo works on any desk. Generation
// moves to its own task (~35 s per story) and the web server keeps serving
// throughout; see loop() below.
#define ENABLE_WIFI 1
#if ENABLE_WIFI
#include <WiFi.h>
#include <WebServer.h>
#define AP_SSID "PLE-TinyLM"
#define AP_PASS "tinystory"
#endif

// Chinese model (artifacts/zh): the head is NOT staged to int8 - core (3.0MB)
// + head (3.15MB) + KV would not fit the 8MB PSRAM. The head runs straight
// from its flash-resident int4 rows instead.
#define STAGE_HEAD_INT8 0

// Device-side context cap. Stories stop at EOT well before this (~130-230
// tokens for 150-200 characters); capping KV at 256 instead of the trained
// 512 is what makes the fp32 KV cache fit alongside the staged core.
#define DEV_SEQ 256

// Repetition penalty (CTRL-style), applied before the argmax: every token
// seen in the prompt or generated so far has its logit divided by this when
// positive and multiplied when negative, once per unique token. Must match
// sample.py's repetition_penalty exactly - acceptance parity relies on it.
// 100 (= 1.0x) disables the penalty entirely. The percent form exists because
// the preprocessor cannot compare float literals in #if.
#define REPETITION_PENALTY_PCT 130
#define REPETITION_PENALTY (REPETITION_PENALTY_PCT / 100.0f)

static const int PROMPT_IDS[] = {524, 258}; // "一天，" (zh-story4 tokenizer)
// Backstop only: the model closes its own story with the end-of-text token
// (id 0) and the loop below stops there, which lands near 120 tokens for a
// 150-200 character Chinese story.
static const int N_GENERATE = 230;

Model model;
Scratch s;

// ---- allocation ------------------------------------------------------------
// Allocations are strict because memory placement is part of the runtime
// configuration. Allocation failure stops initialization.
static size_t psram_used = 0, sram_used = 0;

// The two LLM_Q8_MAX_INPUT int8 activation buffers (matvec_q8, matvec_par),
// which are static and so absent from the totals above. Together these report
// the managed hot set, not total SRAM usage; the free-SRAM figure printed at
// boot is the overall diagnostic.
#define STATIC_SRAM_BYTES (2 * LLM_Q8_MAX_INPUT)

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (p) psram_used += n;
  return p;
}
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) {
    Serial.printf("FATAL: required PSRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  return p;
}
static void *sram_or_die(size_t n, const char *what) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) {
    Serial.printf("FATAL: required SRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  sram_used += n;
  return p;
}

// ---- dual-core int8 matvec -------------------------------------------------
// Serves both hooks. Below ~128 rows the task notify round trip costs more than
// the split saves, so small tensors run single-core.
static TaskHandle_t worker_h, main_h;
static const QT *job_t;
static const int8_t *job_xq;
static float job_xs;
static float *job_y;
static int job_split;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}

static void matvec_par(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
  // The caller need not be setup()'s task: the web UI generates from its own
  // task, so the completion notification has to target whoever called us.
  main_h = xTaskGetCurrentTaskHandle();
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

// Copy RMSNorm weights from mapped flash to internal SRAM.
static void copy_norms_to_sram() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim;
  const float **vecs[3 * 32 + 2];
  int sizes[3 * 32 + 2], n_vec = 0;
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = sram_or_die(bytes, "norm vector");
    memcpy(dst, *vecs[i], bytes);
    *vecs[i] = (const float *)dst;
  }
  Serial.printf("norms  -> SRAM   %d vectors\n", n_vec);
}

static void alloc_scratch() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, S = c->seq_len;
  // hot working set -> internal SRAM
  s.x     = (float *)sram_or_die(D * 4, "x");
  s.h     = (float *)sram_or_die((F > D ? F : D) * 4, "h");
  s.qkv   = (float *)sram_or_die(3 * D * 4, "qkv");
  s.att   = (float *)sram_or_die(D * 4, "att");
  s.g1    = (float *)sram_or_die(F * 4, "g1");
  s.g2    = (float *)sram_or_die((P > F ? P : F) * 4, "g2");
  s.ple   = (float *)sram_or_die(L * P * 4, "ple");
  s.tmpP  = (float *)sram_or_die(L * P * 4, "tmpP");
  s.trow  = (float *)sram_or_die(L * P * 4, "trow");
  s.scores = (float *)sram_or_die(S * 4, "scores");
  // logits: out_vocab floats, 99 KiB here, read once per token. Left in PSRAM
  // rather than spend a fifth of internal SRAM on it.
  s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
  // KV cache: sized for DEV_SEQ, not the trained seq_len (512). At 8 layers x
  // 256 x 192 fp32 this is 1.57MB per cache; at the full 512 it would be 3.15MB
  // each and the Chinese model would not fit PSRAM. llm_forward steps the
  // caches with s.kv_len, set below.
  s.kcache = (float *)ps_or_die((size_t)L * DEV_SEQ * D * 4, "kcache");
  s.vcache = (float *)ps_or_die((size_t)L * DEV_SEQ * D * 4, "vcache");
  s.kv_len = DEV_SEQ;
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

// ---- WiFi AP + web UI -----------------------------------------------------
// The board raises its own access point: no router credentials in the firmware.
// Generation runs in its own task because a story takes ~35 s and the server
// must keep answering while it does.
#if ENABLE_WIFI
static WebServer web(80);
static char *g_text, *g_resp;      // PSRAM: story text and one JSON reply
#define TEXT_CAP 4096
#define RESP_CAP 16384
static volatile int   g_len;
static volatile bool  g_running;
static volatile float g_tok_s, g_ms;

static void generate_story();
static void gen_task(void *);

#include "web_page.h"

// Minimal JSON string escaping: the model emits quotes and newlines, and the
// page reads the text as JSON.
static void json_escape(const char *src, int n, char *dst, int cap) {
  int o = 0;
  for (int i = 0; i < n && o + 7 < cap; i++) {
    unsigned char c = (unsigned char)src[i];
    if (c == '"' || c == '\\') { dst[o++] = '\\'; dst[o++] = (char)c; }
    else if (c < 0x20) o += snprintf(dst + o, cap - o, "\\u%04x", c);
    else dst[o++] = (char)c;
  }
  dst[o] = 0;
}

static void handle_root() { web.send_P(200, "text/html; charset=utf-8", PAGE); }

static void handle_start() {
  if (g_running) { web.send(409, "application/json", "{\"ok\":0}"); return; }
  g_len = 0;
  if (g_text) g_text[0] = 0;
  g_running = true;
  web.send(200, "application/json", "{\"ok\":1}");
  xTaskCreatePinnedToCore(gen_task, "gen", 8192, NULL, 1, NULL, 1);
}

static void handle_text() {
  int since = web.hasArg("since") ? web.arg("since").toInt() : 0;
  if (since < 0) since = 0;
  if (since > g_len) since = g_len;
  int o = snprintf(g_resp, RESP_CAP,
                   "{\"len\":%d,\"done\":%s,\"tps\":%.2f,\"ms\":%.0f,\"text\":\"",
                   g_len, g_running ? "false" : "true",
                   (double)g_tok_s, (double)g_ms);
  json_escape(g_text + since, g_len - since, g_resp + o, RESP_CAP - o - 3);
  strcat(g_resp, "\"}");
  web.send(200, "application/json; charset=utf-8", g_resp);
}

static void start_network() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  web.on("/", handle_root);
  web.on("/start", handle_start);
  web.on("/text", handle_text);
  web.begin();
  Serial.printf("wifi : AP \"%s\" / \"%s\"  ip %s\n", AP_SSID, AP_PASS,
                WiFi.softAPIP().toString().c_str());
  Serial.println("web  : http://192.168.4.1/");
}
#endif

// Emit one token to every active output (serial always; panel when enabled).
static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  // Non-blocking: when no host is draining the USB-CDC buffer (running as a
  // standalone gadget on the display), skip the write instead of stalling the
  // whole generation once the TX buffer fills.
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);
#if USE_DISPLAY
  display_puts(bytes, len);
#endif
#if ENABLE_WIFI
  // The web page polls for everything past the offset it already has, so the
  // story only needs to be accumulated, not pushed.
  if (g_text && g_len + len < TEXT_CAP) {
    memcpy(g_text + g_len, bytes, len);
    g_len += len;
    g_text[g_len] = 0;
  }
#endif
}

// Repetition-penalty bookkeeping: one bit per token id (16384 vocab -> 2KB
// static SRAM). A token is penalized once, however often it appears.
#define VOCAB_BITMAP_BYTES ((VOCAB_N + 7) / 8)
static uint8_t seen[VOCAB_BITMAP_BYTES];

static inline void seen_mark(int tok) {
  seen[tok >> 3] |= (uint8_t)(0x80u >> (tok & 7));
}
static inline int seen_has(int tok) {
  return seen[tok >> 3] & (0x80u >> (tok & 7));
}

// One story from the bundled prompt to EOT. Callable from setup (serial
// only) and from the web task; every output is fed through emit().
static void generate_story() {
#if USE_DISPLAY
  display_home();
#endif
  Serial.print(">>> ");
  int n_prompt = sizeof(PROMPT_IDS) / sizeof(int);
  int pos = 0, tok = 0;
  int64_t decode_us = 0;
  int decoded = 0;

  memset(seen, 0, sizeof(seen));  // fresh story, fresh penalty state
  for (int i = 0; i < n_prompt; i++) {  // prime with the prompt
    tok = PROMPT_IDS[i];
    seen_mark(tok);
    emit(tok);
    llm_forward(&model, tok, pos++, &s);
  }

  llm_profile_reset(&s);

  int64_t t_start = esp_timer_get_time();
  for (int step = 0; step < N_GENERATE && pos < DEV_SEQ; step++) {
#if REPETITION_PENALTY_PCT != 100
    // CTRL-style penalty over every token seen so far. Applied to a fresh
    // logits buffer each step, so penalties never accumulate. Must stay in
    // lockstep with generate(repetition_penalty=...) in src/model.py.
    for (int v = 0; v < model.out_vocab; v++) {
      if (seen_has(v)) {
        s.logits[v] = (s.logits[v] > 0.f)
                          ? s.logits[v] / REPETITION_PENALTY
                          : s.logits[v] * REPETITION_PENALTY;
      }
    }
#endif
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < model.out_vocab; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    if (best == 0) break;  // end-of-text token: the model closed the story
    tok = best;
    seen_mark(tok);
    emit(tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 1) == 0) delay(0);  // yield to the web task every two tokens
  }
  int64_t total_us = esp_timer_get_time() - t_start;

  Serial.printf("\n\n--- %d tokens in %.2f s ---\n", decoded, total_us / 1e6);
  Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
#if USE_DISPLAY
  display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
#if ENABLE_WIFI
  if (decoded > 0) {
    g_tok_s = decoded * 1e6f / (float)total_us;
    g_ms = decode_us / 1000.0f / (float)decoded;
  }
#endif
  blink(0);
}

#if ENABLE_WIFI
static void gen_task(void *) {
  generate_story();
  g_running = false;
  vTaskDelete(NULL);
}
#endif

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 PLE TinyLM ===");

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, model.out_vocab, c->dim, c->n_layers, c->n_heads,
                c->ffn, c->ple_dim, part->size / 1e6);

#if USE_DISPLAY
  display_begin();
#endif

  // The model header states how many logits it produces; vocab.h carries the
  // decode table. If they disagree, every emitted token would be decoded
  // against the wrong table.
  if (VOCAB_N != model.out_vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: vocab.h %d, model %d\n",
                  VOCAB_N, model.out_vocab);
    return;
  }

  alloc_scratch();
  copy_norms_to_sram();
  Serial.printf("hot set-> SRAM   %u B dynamic + %u B static = %u B managed\n",
                (unsigned)sram_used, (unsigned)STATIC_SRAM_BYTES,
                (unsigned)(sram_used + STATIC_SRAM_BYTES));

  // Stage every per-position tensor to int8 in PSRAM.
  int want = llm_core_stage_count(&model);
  int staged = llm_stage_core_int8_alloc(&model, ps);
  if (staged != want) {
    Serial.printf("FATAL: staged %d/%d core tensors\n", staged, want);
    while (1) delay(1000);
  }
#if STAGE_HEAD_INT8
  // The tied head is read per token, not per position, so the core helper does
  // not walk it. Stage it too: it is 85%% of the dense MACs. The English model
  // does this; the Chinese model cannot afford the 3.15MB (see STAGE_HEAD_INT8
  // above) and runs the head straight from flash int4 via the w8==NULL path.
  {
    void *b = ps_or_die(llm_stage_int8_bytes(&model.out_head), "staged head");
    llm_stage_int8(&model.out_head, b);
    ++staged;
  }
#endif
  Serial.printf("weights-> PSRAM  %d tensors int8, %.2f MB allocated\n",
                staged, psram_used / 1048576.0);

  if (xTaskCreatePinnedToCore(worker_main, "mv", 4096, NULL, 2, &worker_h, 0) == pdPASS) {
    // After the worker exists: matvec_par notifies worker_h.
    model.layer_matvec = matvec_par;
    model.head_matvec  = matvec_par;
  } else {
    Serial.println("dual-core worker failed; running single core");
  }

  // FNV-1a over the mapped image. scripts/deploy.sh prints the same value for
  // the file it flashed; the two must agree.
  {
    const uint8_t *img = (const uint8_t *)base;
    uint32_t fp = 2166136261u;
    for (size_t i = 0; i < model.image_bytes; i++) { fp ^= img[i]; fp *= 16777619u; }
    Serial.printf("build: bytes=%u fp=%08x sram=%uB psram=%.2fMB\n",
                  (unsigned)model.image_bytes, fp,
                  (unsigned)(sram_used + STATIC_SRAM_BYTES),
                  psram_used / 1048576.0);
  }
  Serial.printf("free: sram %.0f KB | psram %.2f MB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);

#if ENABLE_WIFI
  g_text = (char *)ps_or_die(TEXT_CAP, "story text");
  g_resp = (char *)ps_or_die(RESP_CAP, "http reply");
  g_text[0] = 0;
  start_network();
#else
  generate_story();
#endif
}

void loop() {
#if ENABLE_WIFI
  web.handleClient();
  delay(2);
#else
  delay(10000);
#endif
}
