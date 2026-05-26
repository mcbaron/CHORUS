# Real-Time Buffer Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Rust benchmark binary that measures `ChorusDsp` processing time across DJ-host callback sizes and STFT parameter combinations, then conditionally add a `DspConfig::low_latency()` preset based on profiling results.

**Architecture:** A `[[bin]]` target (`streaming_latency`) in `chorus-dsp` feeds three audio sources (synthetic sine, PinkPanther.wav, TVSong.wav) through `ChorusDsp::process()` in host-realistic callback chunks, times each call with `std::time::Instant`, and reports real-time safety per `(callback_size, frame_size, hop_size)` combination. A scaffolded results document is written to `docs/profiling/`. If profiling confirms `frame_size=512` meets the real-time constraint for all target hosts, `DspConfig::low_latency()` is added to `processor.rs`.

**Tech Stack:** Rust (stable), `hound` (already a dev-dependency), `std::time::Instant`, `ChorusDsp`/`DspConfig` from `chorus-dsp`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `rust/chorus-dsp/Cargo.toml` | Modify | Add `[[bin]]` target for `streaming_latency` |
| `rust/chorus-dsp/src/bin/streaming_latency.rs` | Create | Full benchmark binary: sanity check, timing loop, table output, results file |
| `docs/profiling/2026-05-22-streaming-latency.md` | Create | Scaffolded results template (filled by implementer after running binary) |
| `rust/chorus-dsp/src/processor.rs` | Modify (conditional) | Add `DspConfig::low_latency()` if profiling warrants it |

---

## Task 1: Add `[[bin]]` target to `Cargo.toml`

**Files:**
- Modify: `rust/chorus-dsp/Cargo.toml`

- [ ] **Step 1: Add the bin target**

Open `/Volumes/mcbaron/repos/CHORUS/rust/chorus-dsp/Cargo.toml` and append after the `[dev-dependencies]` block:

```toml
[[bin]]
name = "streaming_latency"
path = "src/bin/streaming_latency.rs"
```

The full file should look like:

```toml
[package]
name = "chorus-dsp"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
rustfft = "6"
num-complex = "0.4"

[dev-dependencies]
approx = "0.5"
ndarray-npy = "0.8"
ndarray = "0.15"
hound = "3"

[[bin]]
name = "streaming_latency"
path = "src/bin/streaming_latency.rs"
```

- [ ] **Step 2: Verify `hound` is a dev-dependency (already confirmed)**

`hound = "3"` is listed under `[dev-dependencies]`. Because `[[bin]]` targets can use dev-dependencies when built with `cargo run`, this is sufficient. No change needed.

- [ ] **Step 3: Create the `src/bin/` directory**

```bash
mkdir -p /Volumes/mcbaron/repos/CHORUS/rust/chorus-dsp/src/bin
```

- [ ] **Step 4: Create a minimal stub so cargo can parse the manifest**

Create `/Volumes/mcbaron/repos/CHORUS/rust/chorus-dsp/src/bin/streaming_latency.rs` with:

```rust
fn main() {
    println!("streaming_latency stub");
}
```

- [ ] **Step 5: Verify the manifest parses**

```
# Run via run-rust-remote skill
cargo build --bin streaming_latency
```

Expected: compiles, prints nothing (binary not run yet).

- [ ] **Step 6: Commit**

```bash
git add rust/chorus-dsp/Cargo.toml rust/chorus-dsp/src/bin/streaming_latency.rs
git commit -m "feat: scaffold streaming_latency bin target"
```

---

## Task 2: Write the sanity-check helper

**Files:**
- Modify: `rust/chorus-dsp/src/bin/streaming_latency.rs`

The sanity check feeds 4096 sine samples through `ChorusDsp` with a given config and verifies:
1. Non-empty output is returned.
2. All output samples are finite (`f64::is_finite`).

- [ ] **Step 1: Replace the stub with the sanity-check skeleton**

Replace the entire file with:

```rust
use chorus_dsp::{ChorusDsp, DspConfig};
use chorus_dsp::transforms::TransformKind;
use chorus_dsp::filters::unity_chains;

fn make_config(frame_size: usize, hop_size: usize) -> DspConfig {
    DspConfig {
        sample_rate: 48_000,
        epsilon: 1e-9,
        filter_chains: unity_chains(),
        transform: TransformKind::Stft {
            frame_size,
            hop_size,
            smoothing_alpha: 0.0,
        },
    }
}

fn sanity_check(frame_size: usize, hop_size: usize) {
    let config = make_config(frame_size, hop_size);
    let mut dsp = ChorusDsp::new(config);

    // Generate 4096 samples of 440 Hz stereo sine at 48 kHz
    let n = 4096usize;
    let input: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let t = i as f64 / 48_000.0;
            let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
            [s, s]
        })
        .collect();

    let output = dsp
        .process(&input)
        .unwrap_or_else(|e| panic!("sanity_check: process() failed for frame_size={frame_size}: {e}"));

    assert!(
        !output.is_empty(),
        "sanity_check: no output produced for frame_size={frame_size}, hop_size={hop_size}. \
         DSP may be misconfigured."
    );

    for (i, &[l, r]) in output.iter().enumerate() {
        assert!(
            l.is_finite() && r.is_finite(),
            "sanity_check: non-finite sample at index {i}: [{l}, {r}] \
             (frame_size={frame_size}, hop_size={hop_size})"
        );
    }

    println!(
        "  [sanity OK] frame_size={frame_size} hop_size={hop_size}: {} output samples, all finite",
        output.len()
    );
}

fn main() {
    println!("=== Sanity Checks ===");
    for &(fs, hs) in &[(256usize, 128usize), (512, 256), (1024, 512)] {
        sanity_check(fs, hs);
    }
    println!("All sanity checks passed.\n");
}
```

- [ ] **Step 2: Compile and run the sanity check**

```
# Run via run-rust-remote skill
cargo run --release --bin streaming_latency
```

Expected output:
```
=== Sanity Checks ===
  [sanity OK] frame_size=256 hop_size=128: ... output samples, all finite
  [sanity OK] frame_size=512 hop_size=256: ... output samples, all finite
  [sanity OK] frame_size=1024 hop_size=512: ... output samples, all finite
All sanity checks passed.
```

- [ ] **Step 3: Commit**

```bash
git add rust/chorus-dsp/src/bin/streaming_latency.rs
git commit -m "feat: add sanity check to streaming_latency binary"
```

---

## Task 3: Write the WAV-loading helper

**Files:**
- Modify: `rust/chorus-dsp/src/bin/streaming_latency.rs`

`hound` is a dev-dependency. The binary uses it for WAV loading. This task adds a `load_wav_stereo` function that normalizes any bit depth to `f64` `[-1.0, 1.0]` and converts mono to stereo.

- [ ] **Step 1: Add `hound` import and `load_wav_stereo` function**

Add at the top of the file (after the existing `use` lines):

```rust
use std::path::Path;
```

Then add the following function before `main()`:

```rust
/// Load a WAV file as stereo f64 samples normalized to [-1.0, 1.0].
/// Mono files are duplicated to both channels.
/// Truncates to `max_samples` stereo frames if the file is longer.
fn load_wav_stereo(path: &Path, max_samples: usize) -> Vec<[f64; 2]> {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("load_wav_stereo: cannot open {}: {e}", path.display()));
    let spec = reader.spec();
    let num_channels = spec.channels as usize;
    assert!(
        num_channels <= 2,
        "load_wav_stereo: expected mono or stereo, got {} channels in {}",
        num_channels,
        path.display()
    );

    let scale = match spec.sample_format {
        hound::SampleFormat::Float => 1.0_f64,
        hound::SampleFormat::Int => {
            1.0 / (1_i64
                .checked_shl(spec.bits_per_sample as u32 - 1)
                .unwrap_or(1) as f64)
        }
    };

    let raw: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("read error") as f64)
            .collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.expect("read error") as f64 * scale)
            .collect(),
    };

    let stereo: Vec<[f64; 2]> = if num_channels == 2 {
        raw.chunks(2).map(|c| [c[0], c[1]]).collect()
    } else {
        raw.iter().map(|&s| [s, s]).collect()
    };

    stereo.into_iter().take(max_samples).collect()
}
```

- [ ] **Step 2: Add `hound` as a regular dependency (bins cannot use dev-deps in release)**

Because `[[bin]]` targets built with `cargo run --release` cannot access `[dev-dependencies]`, move `hound` to `[dependencies]` in `Cargo.toml`:

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
rustfft = "6"
num-complex = "0.4"
hound = "3"

[dev-dependencies]
approx = "0.5"
ndarray-npy = "0.8"
ndarray = "0.15"
hound = "3"
```

Note: `hound` remains in `[dev-dependencies]` too so existing tests continue to work. Cargo handles the duplication gracefully.

- [ ] **Step 3: Verify compilation**

```
# Run via run-rust-remote skill
cargo build --release --bin streaming_latency
```

Expected: compiles without error.

- [ ] **Step 4: Commit**

```bash
git add rust/chorus-dsp/Cargo.toml rust/chorus-dsp/src/bin/streaming_latency.rs
git commit -m "feat: add WAV loading helper to streaming_latency"
```

---

## Task 4: Write the timing measurement core

**Files:**
- Modify: `rust/chorus-dsp/src/bin/streaming_latency.rs`

This task adds the `measure_combo` function that feeds audio in callback-sized chunks, times each `process()` call, and returns a `BenchResult`.

- [ ] **Step 1: Add the `BenchResult` struct and `measure_combo` function before `main()`**

```rust
#[derive(Debug)]
struct BenchResult {
    callback_size: usize,
    frame_size: usize,
    hop_size: usize,
    source_name: String,
    mean_us: f64,
    p95_us: f64,
    max_us: f64,
    callback_budget_us: f64,
    samples_until_first_output: usize,
    real_time_safe: bool,
}

/// Feed `audio` through `ChorusDsp` in `callback_size`-sample chunks.
/// Returns timing statistics and the sample index of first non-empty output.
fn measure_combo(
    audio: &[[f64; 2]],
    callback_size: usize,
    frame_size: usize,
    hop_size: usize,
    source_name: &str,
) -> BenchResult {
    let config = make_config(frame_size, hop_size);
    let mut dsp = ChorusDsp::new(config);

    let mut call_times_us: Vec<f64> = Vec::new();
    let mut samples_until_first_output: Option<usize> = None;
    let mut total_input = 0usize;

    for chunk in audio.chunks(callback_size) {
        let t0 = std::time::Instant::now();
        let output = dsp.process(chunk).expect("process() failed during benchmark");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        call_times_us.push(elapsed_us);
        total_input += chunk.len();

        if samples_until_first_output.is_none() && !output.is_empty() {
            samples_until_first_output = Some(total_input);
        }
    }

    // Sort for percentile calculation (operates on a copy)
    let mut sorted = call_times_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let mean_us = call_times_us.iter().sum::<f64>() / n as f64;
    let p95_us = sorted[(n as f64 * 0.95) as usize];
    let max_us = sorted[n - 1];

    // Budget: samples_in_callback / sample_rate, in microseconds
    let callback_budget_us = callback_size as f64 / 48_000.0 * 1_000_000.0;
    let real_time_safe = p95_us <= callback_budget_us;

    BenchResult {
        callback_size,
        frame_size,
        hop_size,
        source_name: source_name.to_string(),
        mean_us,
        p95_us,
        max_us,
        callback_budget_us,
        samples_until_first_output: samples_until_first_output.unwrap_or(usize::MAX),
        real_time_safe,
    }
}
```

- [ ] **Step 2: Verify compilation**

```
# Run via run-rust-remote skill
cargo build --release --bin streaming_latency
```

Expected: compiles without error. No output yet — `measure_combo` is not called from `main()` in this step.

- [ ] **Step 3: Commit**

```bash
git add rust/chorus-dsp/src/bin/streaming_latency.rs
git commit -m "feat: add BenchResult and measure_combo to streaming_latency"
```

---

## Task 5: Wire the parameter sweep and table output in `main()`

**Files:**
- Modify: `rust/chorus-dsp/src/bin/streaming_latency.rs`

This task replaces `main()` with the full sweep over all audio sources and `(callback_size, frame_size, hop_size)` combinations, printing a results table to stdout.

- [ ] **Step 1: Replace `main()` with the full sweep**

```rust
fn main() {
    // --- Sanity checks ---
    println!("=== Sanity Checks ===");
    for &(fs, hs) in &[(256usize, 128usize), (512, 256), (1024, 512)] {
        sanity_check(fs, hs);
    }
    println!("All sanity checks passed.\n");

    // --- Audio sources ---
    let sample_rate = 48_000usize;
    let max_samples = sample_rate * 5; // 5 seconds

    // Source 1: 440 Hz stereo sine
    let sine: Vec<[f64; 2]> = (0..max_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
            [s, s]
        })
        .collect();

    // Source 2: PinkPanther.wav (path relative to crate root)
    let wav_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/test_tracks_wav");
    let pink = load_wav_stereo(&wav_dir.join("PinkPanther.wav"), max_samples);
    let tv = load_wav_stereo(&wav_dir.join("TVSong.wav"), max_samples);

    let sources: &[(&[[f64; 2]], &str)] = &[
        (sine.as_slice(), "sine_440hz"),
        (pink.as_slice(), "PinkPanther.wav"),
        (tv.as_slice(), "TVSong.wav"),
    ];

    // --- Parameter grid ---
    let callback_sizes: &[usize] = &[64, 128, 256, 512, 1024];
    let stft_params: &[(usize, usize)] = &[
        (256, 128),
        (512, 256),
        (1024, 512),
    ];

    // --- Run sweep ---
    let mut results: Vec<BenchResult> = Vec::new();

    for &(frame_size, hop_size) in stft_params {
        for &callback_size in callback_sizes {
            for &(audio, source_name) in sources {
                let r = measure_combo(audio, callback_size, frame_size, hop_size, source_name);
                results.push(r);
            }
        }
    }

    // --- Print table ---
    println!("=== Results ===");
    println!(
        "{:<14} {:<12} {:<10} {:<22} {:<12} {:<12} {:<12} {:<14} {}",
        "callback_size",
        "frame_size",
        "hop_size",
        "source",
        "mean_us",
        "p95_us",
        "max_us",
        "budget_us",
        "real_time"
    );
    println!("{}", "-".repeat(120));

    for r in &results {
        println!(
            "{:<14} {:<12} {:<10} {:<22} {:<12.1} {:<12.1} {:<12.1} {:<14.1} {}",
            r.callback_size,
            r.frame_size,
            r.hop_size,
            r.source_name,
            r.mean_us,
            r.p95_us,
            r.max_us,
            r.callback_budget_us,
            if r.real_time_safe { "✓ safe" } else { "✗ NOT SAFE" }
        );
    }

    // --- Per-host summary ---
    println!("\n=== Per-Host Summary (worst case across sources) ===");
    let hosts: &[(&str, &[usize])] = &[
        ("Traktor Pro",    &[256, 512]),
        ("Serato DJ Pro",  &[128, 256, 512]),
        ("Mixxx",          &[64, 128, 256, 512]),
        ("VirtualDJ",      &[256, 512, 1024]),
    ];

    for &(host, host_callbacks) in hosts {
        println!("\n{host}:");
        for &(frame_size, hop_size) in stft_params {
            let latency_ms = frame_size as f64 / 48_000.0 * 1000.0;
            // Worst-case: smallest callback (hardest), across all sources
            let worst_safe = host_callbacks.iter().all(|&cb| {
                results
                    .iter()
                    .filter(|r| r.callback_size == cb && r.frame_size == frame_size && r.hop_size == hop_size)
                    .all(|r| r.real_time_safe)
            });
            println!(
                "  frame={frame_size} hop={hop_size} (~{latency_ms:.1}ms): {}",
                if worst_safe { "✓ real-time safe for all callback sizes" } else { "✗ NOT safe for some callback sizes" }
            );
        }
    }

    // --- samples_until_first_output summary ---
    println!("\n=== First Output Latency (samples until first non-empty output) ===");
    println!("{:<12} {:<10} {:<22} {}", "frame_size", "hop_size", "source", "samples_until_first_output");
    println!("{}", "-".repeat(70));
    // Only show sine source for brevity
    for r in results.iter().filter(|r| r.source_name == "sine_440hz" && r.callback_size == 256) {
        println!(
            "{:<12} {:<10} {:<22} {}",
            r.frame_size, r.hop_size, r.source_name, r.samples_until_first_output
        );
    }

    // --- Write results file ---
    write_results_file(&results);

    println!("\nDone. Results written to docs/profiling/2026-05-22-streaming-latency.md");
}
```

- [ ] **Step 2: Add the `write_results_file` stub (implemented in Task 6)**

Add before `main()`:

```rust
fn write_results_file(_results: &[BenchResult]) {
    // implemented in Task 6
}
```

- [ ] **Step 3: Compile and do a quick smoke run**

```
# Run via run-rust-remote skill
cargo run --release --bin streaming_latency
```

Expected: sanity checks pass, full table printed, "Results written to..." line at the end.

- [ ] **Step 4: Commit**

```bash
git add rust/chorus-dsp/src/bin/streaming_latency.rs
git commit -m "feat: add full parameter sweep and table output to streaming_latency"
```

---

## Task 6: Write the results file to `docs/profiling/`

**Files:**
- Modify: `rust/chorus-dsp/src/bin/streaming_latency.rs`
- Create: `docs/profiling/2026-05-22-streaming-latency.md` (written by the binary at runtime)

- [ ] **Step 1: Replace the `write_results_file` stub with the real implementation**

```rust
fn write_results_file(results: &[BenchResult]) {
    use std::fmt::Write as FmtWrite;

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/profiling/2026-05-22-streaming-latency.md");

    // Ensure parent directory exists
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", parent.display()));
    }

    let mut doc = String::new();
    writeln!(doc, "# CHORUS Streaming Latency Profiling Results").unwrap();
    writeln!(doc, "").unwrap();
    writeln!(doc, "**Date:** 2026-05-22  ").unwrap();
    writeln!(doc, "**Machine:** (fill in: hostname, CPU, RAM)  ").unwrap();
    writeln!(doc, "**Rust:** (fill in: `rustc --version`)  ").unwrap();
    writeln!(doc, "**Sample rate:** 48 kHz  ").unwrap();
    writeln!(doc, "**Audio sources:** sine_440hz (5s), PinkPanther.wav (5s), TVSong.wav (5s)  ").unwrap();
    writeln!(doc, "").unwrap();

    writeln!(doc, "## Full Results Table").unwrap();
    writeln!(doc, "").unwrap();
    writeln!(
        doc,
        "| callback_size | frame_size | hop_size | source | mean_us | p95_us | max_us | budget_us | real_time |"
    ).unwrap();
    writeln!(
        doc,
        "|---|---|---|---|---|---|---|---|---|"
    ).unwrap();

    for r in results {
        writeln!(
            doc,
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {} |",
            r.callback_size,
            r.frame_size,
            r.hop_size,
            r.source_name,
            r.mean_us,
            r.p95_us,
            r.max_us,
            r.callback_budget_us,
            if r.real_time_safe { "✓" } else { "✗" }
        ).unwrap();
    }

    writeln!(doc, "").unwrap();
    writeln!(doc, "## Per-Host Recommendations").unwrap();
    writeln!(doc, "").unwrap();

    let hosts: &[(&str, &[usize])] = &[
        ("Traktor Pro",    &[256, 512]),
        ("Serato DJ Pro",  &[128, 256, 512]),
        ("Mixxx",          &[64, 128, 256, 512]),
        ("VirtualDJ",      &[256, 512, 1024]),
    ];
    let stft_params: &[(usize, usize)] = &[(256, 128), (512, 256), (1024, 512)];

    writeln!(doc, "| Host | Recommended frame_size | Recommended hop_size | Latency | Notes |").unwrap();
    writeln!(doc, "|---|---|---|---|---|").unwrap();

    for &(host, host_callbacks) in hosts {
        // Find the smallest frame_size that is real-time safe for all host callback sizes and sources
        let mut recommendation = None;
        for &(frame_size, hop_size) in stft_params {
            let latency_ms = frame_size as f64 / 48_000.0 * 1000.0;
            let all_safe = host_callbacks.iter().all(|&cb| {
                results
                    .iter()
                    .filter(|r| r.callback_size == cb && r.frame_size == frame_size && r.hop_size == hop_size)
                    .all(|r| r.real_time_safe)
            });
            if all_safe && recommendation.is_none() {
                recommendation = Some((frame_size, hop_size, latency_ms));
            }
        }
        match recommendation {
            Some((fs, hs, ms)) => {
                writeln!(
                    doc,
                    "| {} | {} | {} | ~{:.1}ms | All target callback sizes safe |",
                    host, fs, hs, ms
                ).unwrap();
            }
            None => {
                writeln!(
                    doc,
                    "| {} | N/A | N/A | — | No tested config is real-time safe for all callback sizes |",
                    host
                ).unwrap();
            }
        }
    }

    writeln!(doc, "").unwrap();
    writeln!(doc, "## Summary").unwrap();
    writeln!(doc, "").unwrap();
    writeln!(doc, "(Fill in after reviewing the table above.)").unwrap();
    writeln!(doc, "").unwrap();
    writeln!(doc, "- **Default config** (`frame_size=1024`, `hop_size=512`): real-time safe for which hosts?").unwrap();
    writeln!(doc, "- **Low-latency config** (`frame_size=512`, `hop_size=256`): real-time safe for which hosts?").unwrap();
    writeln!(doc, "- **Decision:** Add `DspConfig::low_latency()` preset? (yes / no / conditional)").unwrap();

    std::fs::write(&out_path, &doc)
        .unwrap_or_else(|e| panic!("cannot write results file {}: {e}", out_path.display()));
}
```

- [ ] **Step 2: Run the full binary to generate the results file**

```
# Run via run-rust-remote skill
cargo run --release --bin streaming_latency
```

Expected: binary runs, prints table to stdout, writes `docs/profiling/2026-05-22-streaming-latency.md`.

- [ ] **Step 3: Verify the file was written**

Check that `docs/profiling/2026-05-22-streaming-latency.md` exists and contains a populated Markdown table. The per-host recommendations section should have real data based on measured results.

- [ ] **Step 4: Commit**

```bash
git add rust/chorus-dsp/src/bin/streaming_latency.rs docs/profiling/2026-05-22-streaming-latency.md
git commit -m "feat: write profiling results to docs/profiling in streaming_latency"
```

---

## Task 7: Scaffold the profiling results template (pre-run placeholder)

**Files:**
- Create: `docs/profiling/2026-05-22-streaming-latency.md`

This task creates a human-readable template that the binary will overwrite when run. It serves as documentation of intent and format before the binary is executed.

> **Note:** If Task 6 was completed first (binary was run), skip this task — the file already exists with real data.

- [ ] **Step 1: Create the scaffolded template**

Create `/Volumes/mcbaron/repos/CHORUS/docs/profiling/2026-05-22-streaming-latency.md`:

```markdown
# CHORUS Streaming Latency Profiling Results

**Date:** 2026-05-22
**Machine:** (fill in after running: hostname, CPU, RAM)
**Rust:** (fill in: `rustc --version`)
**Sample rate:** 48 kHz
**Audio sources:** sine_440hz (5s), PinkPanther.wav (5s), TVSong.wav (5s)

> This file is auto-generated by `cargo run --release --bin streaming_latency`.
> Run the binary to populate with real measurements.

## Full Results Table

| callback_size | frame_size | hop_size | source | mean_us | p95_us | max_us | budget_us | real_time |
|---|---|---|---|---|---|---|---|---|
| (run binary to populate) | | | | | | | | |

## Per-Host Recommendations

| Host | Recommended frame_size | Recommended hop_size | Latency | Notes |
|---|---|---|---|---|
| Traktor Pro | TBD | TBD | TBD | |
| Serato DJ Pro | TBD | TBD | TBD | |
| Mixxx | TBD | TBD | TBD | |
| VirtualDJ | TBD | TBD | TBD | |

## Summary

(Fill in after reviewing the table above.)

- **Default config** (`frame_size=1024`, `hop_size=512`): real-time safe for which hosts?
- **Low-latency config** (`frame_size=512`, `hop_size=256`): real-time safe for which hosts?
- **Decision:** Add `DspConfig::low_latency()` preset? (yes / no / conditional)
```

- [ ] **Step 2: Commit**

```bash
git add docs/profiling/2026-05-22-streaming-latency.md
git commit -m "docs: scaffold streaming latency profiling results template"
```

---

## Task 8: Conditional — Add `DspConfig::low_latency()` if profiling warrants it

**Files:**
- Modify: `rust/chorus-dsp/src/processor.rs`

**When to perform this task:** Only if the profiling results in `docs/profiling/2026-05-22-streaming-latency.md` show that `frame_size=512, hop_size=256` is real-time safe for all target callback sizes across all hosts tested, AND that it provides a meaningful latency improvement over the default.

**When to skip this task:** If `frame_size=512` fails the real-time constraint for any target host/callback combination, or if profiling shows no improvement over `frame_size=1024`.

- [ ] **Step 1: Review profiling results**

Open `docs/profiling/2026-05-22-streaming-latency.md`. Check:
- Does `frame_size=512, hop_size=256` show `✓` for all target callback sizes (64, 128, 256, 512, 1024) across all three audio sources?
- Is the p95 callback time well within budget (< 50% of budget), suggesting a meaningful safety margin?

If yes to both, proceed. Otherwise stop here and note the finding.

- [ ] **Step 2: Add `DspConfig::low_latency()` to `processor.rs`**

In `/Volumes/mcbaron/repos/CHORUS/rust/chorus-dsp/src/processor.rs`, add the following `impl` block after the existing `impl Default for DspConfig`:

```rust
impl DspConfig {
    /// Low-latency preset suitable for Traktor Pro / Serato DJ Pro (~10ms algorithmic latency).
    ///
    /// Uses `frame_size=512, hop_size=256` giving ~10.7ms latency at 48kHz vs ~21.3ms for the
    /// default. Profiling confirmed real-time safety for all target DJ host callback sizes.
    pub fn low_latency() -> Self {
        Self {
            sample_rate: 48_000,
            epsilon: 1e-9,
            filter_chains: crate::filters::unity_chains(),
            transform: crate::transforms::TransformKind::Stft {
                frame_size: 512,
                hop_size: 256,
                smoothing_alpha: 0.0,
            },
        }
    }
}
```

- [ ] **Step 3: Write a test for the new preset**

In the `#[cfg(test)]` block in `processor.rs`, add:

```rust
#[test]
fn low_latency_preset_produces_output() {
    let input: Vec<[f64; 2]> = (0..2048).map(|i| {
        let t = i as f64 / 48_000.0;
        let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
        [s, s]
    }).collect();
    let mut dsp = ChorusDsp::new(DspConfig::low_latency());
    let output = dsp.process(&input).unwrap();
    // frame_size=512: first output arrives after 512 samples, so 2048 input should produce output
    assert!(!output.is_empty(), "low_latency preset produced no output for 2048 input samples");
    for &[l, r] in &output {
        assert!(l.is_finite() && r.is_finite(), "non-finite sample in low_latency output");
    }
}
```

- [ ] **Step 4: Run tests**

```
# Run via run-rust-remote skill
cargo test --lib
```

Expected: all existing tests pass plus `low_latency_preset_produces_output`.

- [ ] **Step 5: Commit**

```bash
git add rust/chorus-dsp/src/processor.rs
git commit -m "feat: add DspConfig::low_latency() preset (frame_size=512, hop_size=256)"
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|---|---|
| Benchmark harness as `[[bin]]` at `benches/streaming_latency.rs` | Task 1 (placed in `src/bin/` per Rust conventions; spec says `benches/` but that conflicts with Rust's `[[bench]]` convention — `src/bin/` is correct for `[[bin]]`) |
| Three audio sources: sine 440Hz, PinkPanther.wav, TVSong.wav | Task 5 |
| Feed in `callback_size`-sample chunks via `ChorusDsp::process()` | Task 4 |
| Measure mean, p95, max callback time (μs) | Task 4 |
| Record `samples_until_first_output` | Task 4 |
| Pass/fail: `p95 ≤ callback_budget` | Task 4 |
| Parameter grid: `callback_size ∈ [64,128,256,512,1024]`, STFT params `(256/128, 512/256, 1024/512)` | Task 5 |
| Print results table to stdout | Task 5 |
| Write `docs/profiling/YYYY-MM-DD-streaming-latency.md` | Task 6 |
| Per-host recommendations in output file | Task 6 |
| Sanity check: 4096 sine samples → non-empty output, all finite | Task 2 |
| `DspConfig::low_latency()` conditional on profiling | Task 8 |
| Scaffolded results template | Task 7 |

### Placeholder Scan

- Task 7 contains "TBD" in the Markdown template file content — this is intentional, it is the template content that the binary replaces with real data when run. Not a plan placeholder.
- `write_results_file` stub in Task 5 Step 2 is replaced in Task 6 Step 1. Cross-task dependency is explicit.

### Type Consistency

- `BenchResult` defined in Task 4, used in Task 5 and Task 6 — field names consistent throughout.
- `make_config(frame_size, hop_size)` defined in Task 2, called in Task 4 `measure_combo` — signature consistent.
- `load_wav_stereo(path: &Path, max_samples: usize) -> Vec<[f64; 2]>` defined in Task 3, called in Task 5 — consistent.
- `DspConfig::low_latency()` defined and tested within Task 8 — no cross-task type dependency.

### Note on `benches/` vs `src/bin/`

The spec says `benches/streaming_latency.rs` but also says to add it as a `[[bin]]` target. In Rust, `benches/` is for `[[bench]]` criterion-style targets. A `[[bin]]` target belongs in `src/bin/`. The plan uses `src/bin/streaming_latency.rs` which is correct. The `cargo run --release --bin streaming_latency` invocation specified in the spec works correctly from `src/bin/`.
