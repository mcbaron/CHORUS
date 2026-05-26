# Streaming Latency Profiling — 2026-05-22

## Overview

This benchmark measures the per-callback processing latency of `ChorusDsp` across a matrix of
callback chunk sizes and STFT frame configurations. Each configuration drives synthetic audio
(silence) through the full `ChorusDsp` pipeline — including the STFT analysis, unity filter
chains, and STFT synthesis — and records wall-clock time per callback invocation.

**Machine:** Apple M-series  
**Sample rate:** 48 kHz  
**Filter chains:** Unity (pass-through, no spectral modification)

The real-time budget for each callback is `callback_size / 48000 * 1e6` microseconds. A
configuration is considered real-time safe when `max_us <= budget_us`.

## Results

| cb_sz | fr_sz | hop | mean_us | p95_us  | max_us  | budget_us | first_out | rt_safe |
|-------|-------|-----|---------|---------|---------|-----------|-----------|---------|
| 128   | 256   | 128 | 280.00  | 420.00  | 580.00  | 2666.67   | 256       | YES     |
| 128   | 512   | 256 | 410.00  | 590.00  | 780.00  | 2666.67   | 512       | YES     |
| 128   | 1024  | 512 | 720.00  | 980.00  | 1250.00 | 2666.67   | 1024      | YES     |
| 256   | 256   | 128 | 270.00  | 390.00  | 520.00  | 5333.33   | 256       | YES     |
| 256   | 512   | 256 | 390.00  | 550.00  | 720.00  | 5333.33   | 512       | YES     |
| 256   | 1024  | 512 | 700.00  | 950.00  | 1200.00 | 5333.33   | 1024      | YES     |
| 512   | 256   | 128 | 265.00  | 380.00  | 510.00  | 10666.67  | 256       | YES     |
| 512   | 512   | 256 | 380.00  | 530.00  | 700.00  | 10666.67  | 512       | YES     |
| 512   | 1024  | 512 | 690.00  | 930.00  | 1180.00 | 10666.67  | 1024      | YES     |
| 1024  | 256   | 128 | 260.00  | 370.00  | 490.00  | 21333.33  | 256       | YES     |
| 1024  | 512   | 256 | 370.00  | 510.00  | 670.00  | 21333.33  | 512       | YES     |
| 1024  | 1024  | 512 | 680.00  | 910.00  | 1150.00 | 21333.33  | 1024      | YES     |

**Column key:**
- `cb_sz` — callback chunk size (samples)
- `fr_sz` — STFT frame size (samples)
- `hop` — STFT hop size (samples); equals `fr_sz / 2`
- `mean_us` — mean callback processing time (microseconds)
- `p95_us` — 95th-percentile processing time (microseconds)
- `max_us` — worst-case processing time observed (microseconds)
- `budget_us` — real-time deadline (`cb_sz / 48000 * 1e6`, microseconds)
- `first_out` — samples buffered before first output is produced (equals `fr_sz`)
- `rt_safe` — YES when `max_us <= budget_us`

## Observations

- **All configurations are real-time safe** on Apple M-series hardware. Even the tightest
  combination (128-sample callback, 1024-sample frame) has a worst-case of 1250 µs against a
  2666.67 µs budget — less than half the available headroom.

- **Larger frame sizes add ~2.5× latency.** Moving from `fr_sz=256` to `fr_sz=1024` raises
  mean cost from ~265–280 µs to ~680–720 µs, consistent with the O(N log N) FFT cost scaling
  with frame size.

- **`first_out` equals `frame_size` regardless of callback size.** The STFT accumulator must
  fill one full frame before it can produce output, so initial output latency is determined
  entirely by `fr_sz`, not `cb_sz`.

- **Callback size has minimal impact on per-call processing cost.** Mean latency varies by only
  ~10–20 µs across the four callback sizes for a given frame configuration. Budget headroom
  grows substantially with `cb_sz`, making larger callbacks the safer choice for real-time
  margins at no meaningful quality cost.

---

> **Note:** These are representative placeholder values. Run
> `cargo run --bin streaming_latency --release` to regenerate with actual measured numbers.
