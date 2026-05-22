use std::panic::{catch_unwind, AssertUnwindSafe};

use chorus_dsp::{ChorusDsp, DspConfig};

/// Opaque handle wrapping `ChorusDsp`. C callers hold a `*mut ChorusHandle`.
pub struct ChorusHandle {
    dsp: ChorusDsp,
}

/// Allocate a new `ChorusDsp` with default config, overriding `sample_rate`.
///
/// Returns a non-null opaque pointer on success, or null if a panic occurs
/// during construction.
///
/// # Note
/// `sample_rate` is stored in the config but transform constructors currently
/// default to 48 kHz internally. Pass `48_000` for now; other values are accepted
/// but the DSP will still operate at 48 kHz.
///
/// # Safety
/// The returned pointer must be freed with `chorus_destroy`. It must not be
/// used after `chorus_destroy` is called.
#[no_mangle]
pub unsafe extern "C" fn chorus_create(sample_rate: u32) -> *mut ChorusHandle {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let config = DspConfig { sample_rate, ..DspConfig::default() };
        let dsp = ChorusDsp::new(config);
        Box::into_raw(Box::new(ChorusHandle { dsp }))
    }));

    match result {
        Ok(ptr) => ptr,
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a handle previously returned by `chorus_create`.
///
/// No-op if `handle` is null.
///
/// # Safety
/// `handle` must be a valid pointer returned by `chorus_create` that has not
/// already been destroyed.
#[no_mangle]
pub unsafe extern "C" fn chorus_destroy(handle: *mut ChorusHandle) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !handle.is_null() {
            drop(Box::from_raw(handle));
        }
    }));
}

/// Process interleaved stereo audio.
///
/// `input`  — pointer to `frames * 2` interleaved f32 samples (L, R, L, R, …).
/// `output` — pointer to a buffer of at least `frames * 2` f32 samples that
///            will receive the processed interleaved stereo output.
/// `frames` — number of stereo frames to process.
///
/// Return codes:
///  0  — success
/// -1  — null pointer argument or integer overflow
/// -2  — empty input (frames == 0) or DSP error
/// -3  — internal panic
///
/// Because the STFT introduces latency, the number of output frames produced
/// may be less than `frames`. In that case the remaining output samples are
/// zeroed. The caller should expect a warm-up period.
///
/// # Safety
/// - `input`  must point to at least `frames * 2` initialised f32 values.
/// - `output` must point to a writable buffer of at least `frames * 2` f32 values.
/// - `handle` must be a valid, non-null pointer returned by `chorus_create`.
/// - No other thread may access `handle` concurrently.
#[no_mangle]
pub unsafe extern "C" fn chorus_process_interleaved(
    handle: *mut ChorusHandle,
    input: *const f32,
    output: *mut f32,
    frames: usize,
) -> i32 {
    if handle.is_null() || input.is_null() || output.is_null() {
        return -1;
    }

    if frames == 0 {
        return -2;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Check for overflow: frames * 2 must fit in usize.
        let sample_count = match frames.checked_mul(2) {
            Some(n) => n,
            None => return -1,
        };

        // Build a Vec<[f64; 2]> from the interleaved f32 input.
        let input_slice = std::slice::from_raw_parts(input, sample_count);
        let stereo_in: Vec<[f64; 2]> = input_slice
            .chunks_exact(2)
            .map(|ch| [ch[0] as f64, ch[1] as f64])
            .collect();

        // Prepare the output slice for writing.
        let output_slice = std::slice::from_raw_parts_mut(output, sample_count);

        // Process through ChorusDsp.
        let dsp = &mut (*handle).dsp;
        match dsp.process(&stereo_in) {
            Ok(processed) => {
                // STFT latency: processed.len() may be <= frames.
                let write_frames = processed.len().min(frames);
                for (i, frame) in processed[..write_frames].iter().enumerate() {
                    output_slice[i * 2] = frame[0] as f32;
                    output_slice[i * 2 + 1] = frame[1] as f32;
                }
                // Zero-fill any frames not yet produced (warm-up latency).
                for i in write_frames..frames {
                    output_slice[i * 2] = 0.0;
                    output_slice[i * 2 + 1] = 0.0;
                }
                0i32
            }
            Err(_) => -2i32,
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => -3,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_destroy_round_trip() {
        unsafe {
            let handle = chorus_create(48_000);
            assert!(!handle.is_null(), "chorus_create should return non-null");
            chorus_destroy(handle);
        }
    }

    #[test]
    fn destroy_null_is_safe() {
        unsafe {
            chorus_destroy(std::ptr::null_mut()); // must not panic
        }
    }

    #[test]
    fn process_null_handle_returns_minus_one() {
        let input = vec![0.0f32; 4];
        let mut output = vec![0.0f32; 4];
        let ret = unsafe {
            chorus_process_interleaved(
                std::ptr::null_mut(),
                input.as_ptr(),
                output.as_mut_ptr(),
                2,
            )
        };
        assert_eq!(ret, -1);
    }

    #[test]
    fn process_null_input_returns_minus_one() {
        unsafe {
            let handle = chorus_create(48_000);
            let mut output = vec![0.0f32; 4];
            let ret = chorus_process_interleaved(
                handle,
                std::ptr::null(),
                output.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, -1);
            chorus_destroy(handle);
        }
    }

    #[test]
    fn process_null_output_returns_minus_one() {
        unsafe {
            let handle = chorus_create(48_000);
            let input = vec![0.0f32; 4];
            let ret = chorus_process_interleaved(
                handle,
                input.as_ptr(),
                std::ptr::null_mut(),
                2,
            );
            assert_eq!(ret, -1);
            chorus_destroy(handle);
        }
    }

    #[test]
    fn process_zero_frames_returns_minus_two() {
        unsafe {
            let handle = chorus_create(48_000);
            let input = vec![0.0f32; 0];
            let mut output = vec![0.0f32; 0];
            let ret = chorus_process_interleaved(
                handle,
                input.as_ptr(),
                output.as_mut_ptr(),
                0,
            );
            assert_eq!(ret, -2);
            chorus_destroy(handle);
        }
    }

    #[test]
    fn process_valid_input_returns_zero() {
        unsafe {
            let handle = chorus_create(48_000);
            assert!(!handle.is_null());

            // 2048 frames of a 440 Hz sine at 48 kHz.
            let frames = 2048usize;
            let input: Vec<f32> = (0..frames)
                .flat_map(|i| {
                    let t = i as f64 / 48_000.0;
                    let s = (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32 * 0.5;
                    [s, s]
                })
                .collect();
            let mut output = vec![0.0f32; frames * 2];

            let ret = chorus_process_interleaved(
                handle,
                input.as_ptr(),
                output.as_mut_ptr(),
                frames,
            );
            assert_eq!(ret, 0, "expected success return code");

            chorus_destroy(handle);
        }
    }

    #[test]
    fn process_twice_on_same_handle_succeeds() {
        unsafe {
            let handle = chorus_create(48_000);
            assert!(!handle.is_null());

            let input = vec![0.0f32; 2048 * 2];
            let mut output = vec![0.0f32; 2048 * 2];

            let first = chorus_process_interleaved(
                handle,
                input.as_ptr(),
                output.as_mut_ptr(),
                2048,
            );
            let second = chorus_process_interleaved(
                handle,
                input.as_ptr(),
                output.as_mut_ptr(),
                2048,
            );

            assert_eq!(first, 0);
            assert_eq!(second, 0);
            chorus_destroy(handle);
        }
    }

    #[test]
    fn output_zero_filled_during_warmup() {
        // 3 frames is far below the default STFT frame size (1024),
        // so the DSP returns 0 output frames during warm-up.
        unsafe {
            let frames = 3usize;
            let handle = chorus_create(48_000);
            assert!(!handle.is_null());
            let input = vec![1.0f32; frames * 2];
            let mut output = vec![999.0f32; frames * 2];
            let result = chorus_process_interleaved(
                handle,
                input.as_ptr(),
                output.as_mut_ptr(),
                frames,
            );
            assert_eq!(result, 0);
            for &sample in &output {
                assert_eq!(
                    sample, 0.0,
                    "expected zero-fill during warm-up, got {sample}"
                );
            }
            chorus_destroy(handle);
        }
    }
}
