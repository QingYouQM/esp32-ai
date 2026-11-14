// Optional on-device screen for the demo: the story appears on a display wired
// to the ESP32 itself, no laptop. Two panels supported via DISPLAY_KIND:
//
//   DISPLAY_TFT_SPI (default) -- the XZ-AI expansion board's 1.54" 240x240
//     ST7789, plugged into its 8-pin TFT socket. Pins and the Chinese font are
//     in the TFT block below.
//   DISPLAY_OLED_I2C -- bare 128x64 I2C mono OLED on free wiring:
//     GND->GND, VCC->3V3, SCL->GPIO46, SDA->GPIO18. Set OLED_CONTROLLER to match
//     the panel: 1.3" is usually SH1106, 0.96" usually SSD1306. Wrong choice ->
//     top row correct, rest noise (SH1106-as-SSD1306) -> flip the define.
//     (On the expansion board the OLED socket is I2C on GPIO41/42 instead.)
//
// The API (display_begin / display_puts) is identical for both, so the sketch
// integration never changes -- only this header and the wiring.
#ifndef DISPLAY_H
#define DISPLAY_H

#define DISPLAY_OLED_I2C 1
#define DISPLAY_TFT_SPI  2
#ifndef DISPLAY_KIND
#define DISPLAY_KIND DISPLAY_TFT_SPI
#endif

// ==================== 128x64 I2C OLED (SH1106 or SSD1306) ===================
#if DISPLAY_KIND == DISPLAY_OLED_I2C
#include <Adafruit_GFX.h>
#include <Wire.h>

// Set to match the panel you wired: 1.3" is usually SH1106, 0.96" usually SSD1306.
// Wrong choice -> shifted/garbled image; just switch this and reflash.
#define OLED_SH1106  1
#define OLED_SSD1306 2
#ifndef OLED_CONTROLLER
#define OLED_CONTROLLER OLED_SH1106
#endif

#define OLED_SDA 18
#define OLED_SCL 46
#define OLED_ADDR 0x3C          // some panels are 0x3D
#define SCR_W 128
#define SCR_H 64
#define CW 6                    // 6x8 base glyph at text size 1
#define CH 8

#if OLED_CONTROLLER == OLED_SH1106
#include <Adafruit_SH110X.h>
static Adafruit_SH1106G oled = Adafruit_SH1106G(SCR_W, SCR_H, &Wire, -1);
#define OLED_WHITE SH110X_WHITE
#else
#include <Adafruit_SSD1306.h>
static Adafruit_SSD1306 oled = Adafruit_SSD1306(SCR_W, SCR_H, &Wire, -1);
#define OLED_WHITE SSD1306_WHITE
#endif

static int ox = 0, oy = 0;

static void display_home() {
  oled.clearDisplay();
  ox = 0; oy = 0;
}

static void display_begin() {
  Wire.begin(OLED_SDA, OLED_SCL);
  Wire.setClock(400000);   // 400kHz fast I2C -> ~4x quicker frame flush than default
#if OLED_CONTROLLER == OLED_SH1106
  oled.begin(OLED_ADDR, true);
#else
  oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
#endif
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(OLED_WHITE);
  oled.setTextWrap(false);      // we wrap at token boundaries ourselves
  oled.display();
  display_home();
}

// Append a token's bytes; whole-token wrap, clear+home when the screen fills,
// flush the framebuffer once per token so text visibly appears.
static void display_puts(const unsigned char *s, int len) {
  if (ox + len * CW > SCR_W) { oy += CH; ox = 0; }
  if (oy + CH > SCR_H) display_home();
  for (int i = 0; i < len; i++) {
    char c = (char)s[i];
    if (c == '\n') { oy += CH; ox = 0; }
    else if (c >= 32 && c < 127) {
      if (ox + CW > SCR_W) { oy += CH; ox = 0; }
      if (oy + CH > SCR_H) display_home();
      oled.setCursor(ox, oy);
      oled.write(c);
      ox += CW;
    }
    if (oy + CH > SCR_H) display_home();
  }
  oled.display();
}

// ================= 1.54" SPI TFT (ST7789 240x240, 8-pin socket) ============
//
// Pins are the XZ-AI expansion board's 8-pin TFT socket, not free wiring:
//   GND VCC SCL SDA RES DC CS BLK  ->  GND 3V3 21 47 45 40 41 42
// (the 7-pin 1.3" socket is the same minus CS). Getting these wrong is not a
// blank screen only -- DC/RST used to sit on GPIO 7/6, which on this board are
// the speaker DOUT and mic DIN, so I2S dies with it.
//
// Chinese: Adafruit's built-in font is ASCII-only, so text is drawn from
// generated/cjk_font.h instead -- 16x16 1bpp for every character the corpus
// contains (4023 glyphs, 126 KB in flash). BPE tokens can split a UTF-8
// sequence, so display_puts() carries the partial sequence across calls.
#else
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>
#include "generated/cjk_font.h"

#define TFT_CS   41
#define TFT_DC   40
#define TFT_RST  45
#define TFT_SCK  21
#define TFT_MOSI 47
#define TFT_BL   42     // backlight: nothing lights up unless this is driven high
#define TFT_W 240
#define TFT_H 240
#define TFT_SPI_HZ 40000000

static Adafruit_ST7789 tft = Adafruit_ST7789(&SPI, TFT_CS, TFT_DC, TFT_RST);

static int cx = 0, cy = 0;   // text cursor, one 16px cell per character

static void display_home() {
  tft.fillScreen(ST77XX_BLACK);
  cx = 0; cy = 0;
}

static void display_begin() {
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);

  SPI.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);
  tft.init(TFT_W, TFT_H);
  tft.setSPISpeed(TFT_SPI_HZ);
  tft.setRotation(0);
  tft.invertDisplay(true);   // 1.54" IPS panel: without this everything is inverted
  tft.fillScreen(ST77XX_BLACK);
  display_home();
}

static int cjk_index(uint32_t cp) {
  if (cp > 0xFFFF) return -1;   // the table is uint16
  int lo = 0, hi = CJK_COUNT - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    uint16_t v = CJK_CHARS[mid];
    if (v == cp) return mid;
    if (v < cp) lo = mid + 1; else hi = mid - 1;
  }
  return -1;
}

static void newline() { cx = 0; cy += CJK_FONT_H; }

static void draw_cp(uint32_t cp) {
  if (cp == '\r') return;
  if (cp == '\n') { newline(); }
  else {
    if (cx + CJK_FONT_W > TFT_W) newline();
    if (cy + CJK_FONT_H > TFT_H) display_home();
    int i = cjk_index(cp);
    if (i >= 0)
      tft.drawBitmap(cx, cy, CJK_BITS + (size_t)i * 32, CJK_FONT_W, CJK_FONT_H,
                     ST77XX_WHITE);
    cx += CJK_FONT_W;
  }
  if (cy + CJK_FONT_H > TFT_H) display_home();
}

static void display_puts(const unsigned char *s, int len) {
  // A BPE token can end in the middle of a UTF-8 sequence; keep the tail.
  static uint8_t seq[3];
  static int need = 0, have = 0;

  for (int i = 0; i < len; i++) {
    uint8_t b = s[i];
    if (need == 0) {
      if (b < 0x80) { draw_cp(b); continue; }
      if ((b & 0xE0) == 0xC0) { need = 1; }
      else if ((b & 0xF0) == 0xE0) { need = 2; }
      else if ((b & 0xF8) == 0xF0) { need = 3; }
      else { draw_cp('?'); continue; }   // stray continuation byte
      seq[0] = b; have = 1;
      continue;
    }
    seq[have++] = b;
    if (have == need + 1) {
      uint32_t cp = (need == 1) ? (seq[0] & 0x1F)
                  : (need == 2) ? (seq[0] & 0x0F)
                                : (seq[0] & 0x07);
      for (int k = 1; k <= need; k++) cp = (cp << 6) | (seq[k] & 0x3F);
      draw_cp(cp);
      need = have = 0;
    }
  }
}

#endif
#endif
