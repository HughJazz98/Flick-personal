//! Lightweight linked-stereo dynamics processors for the realtime callback.
//!
//! Signal order: upward compressor -> downward compressor -> lookahead limiter.
//!
//! - no allocations in the audio callback (all buffers are sized in `new`,
//!   which runs on the command thread / on sample-rate changes)
//! - one shared envelope per frame for stereo coherence
//! - the upward compressor lifts quiet passages without touching loud ones
//! - the downward compressor is a soft feed-forward design
//! - the limiter looks ahead ~1.5 ms so its gain is already down when a peak
//!   arrives (min-hold + moving-average envelope: no overshoot by design)

const EPSILON: f32 = 1.0e-6;
const LIMITER_LOOKAHEAD_MS: f32 = 1.5;
const MAX_CHANNELS: usize = 8;
const UPWARD_DETECTOR_MS: f32 = 50.0;
const UPWARD_ATTACK_MS: f32 = 5.0;
/// Below this level nothing is lifted, so silence, fade-outs and dither/tape
/// hiss stay where they are.
const UPWARD_FLOOR_DB: f32 = -70.0;
/// Width of the ramp above the floor over which the lift fades in.
const UPWARD_FLOOR_RAMP_DB: f32 = 12.0;

#[derive(Debug, Clone, Copy)]
pub struct CompressorSettings {
    pub enabled: bool,
    pub threshold_db: f32,
    pub ratio: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub makeup_gain_db: f32,
}

impl CompressorSettings {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            threshold_db: -18.0,
            ratio: 3.0,
            attack_ms: 12.0,
            release_ms: 140.0,
            makeup_gain_db: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UpwardCompressorSettings {
    pub enabled: bool,
    /// Signals below this level get lifted.
    pub threshold_db: f32,
    /// Upward ratio: 2.0 means a signal 10 dB under the threshold ends up
    /// 5 dB under it.
    pub ratio: f32,
    /// Hard cap on the lift so very quiet material isn't dragged way up.
    pub max_boost_db: f32,
    /// How quickly the lift comes back in after a loud passage.
    pub release_ms: f32,
}

impl UpwardCompressorSettings {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            threshold_db: -36.0,
            ratio: 2.0,
            max_boost_db: 8.0,
            release_ms: 300.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LimiterSettings {
    pub enabled: bool,
    pub input_gain_db: f32,
    pub ceiling_db: f32,
    pub release_ms: f32,
}

impl LimiterSettings {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            input_gain_db: 0.0,
            ceiling_db: -0.8,
            release_ms: 80.0,
        }
    }
}

pub struct DynamicsChain {
    sample_rate: u32,
    upward: UpwardCompressorSettings,
    compressor: CompressorSettings,
    limiter: LimiterSettings,

    upward_detector_coeff: f32,
    upward_attack_coeff: f32,
    upward_release_coeff: f32,
    upward_level: f32,
    upward_gain_db: f32,

    compressor_attack_coeff: f32,
    compressor_release_coeff: f32,
    compressor_gain_db: f32,

    limiter_release_coeff: f32,
    limiter_input_gain_lin: f32,
    limiter_ceiling_lin: f32,
    /// Lookahead window in frames (also the audio delay).
    lookahead: usize,
    /// Ring of per-frame required gains (min-hold window).
    required_gain: Vec<f32>,
    /// Ring of released envelope values (moving-average window).
    smoothing: Vec<f32>,
    smoothing_sum: f64,
    /// Interleaved delay line, `lookahead * MAX_CHANNELS` samples.
    delay: Vec<f32>,
    ring_pos: usize,
    limiter_envelope: f32,
}

impl DynamicsChain {
    pub fn new(sample_rate: u32) -> Self {
        let sample_rate = sample_rate.max(1);
        let lookahead = ((LIMITER_LOOKAHEAD_MS / 1000.0) * sample_rate as f32)
            .round()
            .max(1.0) as usize;
        let mut chain = Self {
            sample_rate,
            upward: UpwardCompressorSettings::disabled(),
            compressor: CompressorSettings::disabled(),
            limiter: LimiterSettings::disabled(),
            upward_detector_coeff: 0.0,
            upward_attack_coeff: 0.0,
            upward_release_coeff: 0.0,
            upward_level: 0.0,
            upward_gain_db: 0.0,
            compressor_attack_coeff: 0.0,
            compressor_release_coeff: 0.0,
            compressor_gain_db: 0.0,
            limiter_release_coeff: 0.0,
            limiter_input_gain_lin: 1.0,
            limiter_ceiling_lin: db_to_linear(-0.8),
            lookahead,
            required_gain: vec![1.0; lookahead],
            smoothing: vec![1.0; lookahead],
            smoothing_sum: lookahead as f64,
            delay: vec![0.0; lookahead * MAX_CHANNELS],
            ring_pos: 0,
            limiter_envelope: 1.0,
        };
        chain.refresh_coefficients();
        chain
    }

    /// Limiter latency in frames while the limiter is enabled.
    pub fn limiter_latency_frames(&self) -> usize {
        self.lookahead.saturating_sub(1)
    }

    pub fn is_active(&self) -> bool {
        self.upward.enabled || self.compressor.enabled || self.limiter.enabled
    }

    pub fn set_upward_compressor(
        &mut self,
        enabled: bool,
        threshold_db: f32,
        ratio: f32,
        max_boost_db: f32,
        release_ms: f32,
    ) {
        self.upward = UpwardCompressorSettings {
            enabled,
            threshold_db: finite_or(threshold_db, -36.0).clamp(-60.0, -10.0),
            ratio: finite_or(ratio, 2.0).clamp(1.0, 4.0),
            max_boost_db: finite_or(max_boost_db, 8.0).clamp(0.0, 18.0),
            release_ms: finite_or(release_ms, 300.0).clamp(50.0, 1000.0),
        };
        self.refresh_coefficients();
        if !enabled {
            self.upward_gain_db = 0.0;
            self.upward_level = 0.0;
        }
    }

    pub fn set_compressor(
        &mut self,
        enabled: bool,
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        makeup_gain_db: f32,
    ) {
        self.compressor = CompressorSettings {
            enabled,
            threshold_db: threshold_db.clamp(-36.0, 0.0),
            ratio: ratio.clamp(1.0, 12.0),
            attack_ms: attack_ms.clamp(1.0, 100.0),
            release_ms: release_ms.clamp(20.0, 500.0),
            makeup_gain_db: makeup_gain_db.clamp(-12.0, 12.0),
        };
        self.refresh_coefficients();
        if !enabled {
            self.compressor_gain_db = 0.0;
        }
    }

    pub fn set_limiter(
        &mut self,
        enabled: bool,
        input_gain_db: f32,
        ceiling_db: f32,
        release_ms: f32,
    ) {
        let was_enabled = self.limiter.enabled;
        self.limiter = LimiterSettings {
            enabled,
            input_gain_db: input_gain_db.clamp(0.0, 12.0),
            ceiling_db: ceiling_db.clamp(-12.0, 0.0),
            release_ms: release_ms.clamp(20.0, 300.0),
        };
        self.refresh_coefficients();
        if enabled != was_enabled {
            self.reset_limiter_state();
        }
    }

    pub fn process(&mut self, buf: &mut [f32], channels: usize) {
        if channels == 0 || !self.is_active() {
            return;
        }

        for frame in buf.chunks_exact_mut(channels) {
            if self.upward.enabled {
                self.process_upward_frame(frame);
            }
            if self.compressor.enabled {
                self.process_compressor_frame(frame);
            }
            if self.limiter.enabled {
                self.process_limiter_frame(frame);
            }
        }
    }

    fn process_upward_frame(&mut self, frame: &mut [f32]) {
        let peak = frame_peak(frame);

        // Peak-hold detector: instant rise, ~50 ms fall. Keeps the level
        // reading steady through waveform zero crossings so the lift doesn't
        // wobble on bass notes.
        self.upward_level = if peak > self.upward_level {
            peak
        } else {
            self.upward_detector_coeff * self.upward_level
                + (1.0 - self.upward_detector_coeff) * peak
        };

        let level_db = linear_to_db(self.upward_level);
        let desired_db = upward_boost_db(&self.upward, level_db);

        // Pulling the lift back (a loud part arriving) is fast; bringing it
        // back in is the user-facing release.
        let coeff = if desired_db < self.upward_gain_db {
            self.upward_attack_coeff
        } else {
            self.upward_release_coeff
        };
        self.upward_gain_db = coeff * self.upward_gain_db + (1.0 - coeff) * desired_db;

        if self.upward_gain_db > 0.001 {
            let gain = db_to_linear(self.upward_gain_db);
            for sample in frame.iter_mut() {
                *sample *= gain;
            }
        }
    }

    fn process_compressor_frame(&mut self, frame: &mut [f32]) {
        let peak = frame_peak(frame);

        let desired_gain_db = if peak <= EPSILON {
            self.compressor.makeup_gain_db
        } else {
            let input_db = linear_to_db(peak);
            if input_db <= self.compressor.threshold_db {
                self.compressor.makeup_gain_db
            } else {
                let compressed_db = self.compressor.threshold_db
                    + (input_db - self.compressor.threshold_db) / self.compressor.ratio;
                self.compressor.makeup_gain_db + (compressed_db - input_db)
            }
        };

        let coeff = if desired_gain_db < self.compressor_gain_db {
            self.compressor_attack_coeff
        } else {
            self.compressor_release_coeff
        };

        self.compressor_gain_db = coeff * self.compressor_gain_db + (1.0 - coeff) * desired_gain_db;

        let gain = db_to_linear(self.compressor_gain_db);
        for sample in frame.iter_mut() {
            *sample *= gain;
        }
    }

    /// Lookahead peak limiter.
    ///
    /// For each incoming frame the gain needed to keep it under the ceiling is
    /// written into a ring. The minimum over the last `lookahead` frames is
    /// taken (min-hold), a release curve is applied on the way back up, and
    /// the result is averaged over another `lookahead` frames. Audio is
    /// delayed by `lookahead - 1` frames, so by the time a frame comes out of
    /// the delay line every gain value in the averaging window is already at
    /// or below what that frame needed — the gain is fully down when the peak
    /// arrives, with a smooth ~1.5 ms ramp instead of a step.
    fn process_limiter_frame(&mut self, frame: &mut [f32]) {
        let channels = frame.len().min(MAX_CHANNELS);
        let len = self.lookahead;
        let pos = self.ring_pos;
        let input_gain = self.limiter_input_gain_lin;

        let mut peak = 0.0f32;
        for sample in frame.iter().take(channels) {
            peak = peak.max((sample * input_gain).abs());
        }
        let required = if peak > self.limiter_ceiling_lin && peak > EPSILON {
            self.limiter_ceiling_lin / peak
        } else {
            1.0
        };
        self.required_gain[pos] = required;

        let held = self
            .required_gain
            .iter()
            .fold(1.0f32, |min_gain, &gain| min_gain.min(gain));

        self.limiter_envelope = if held < self.limiter_envelope {
            held
        } else {
            self.limiter_release_coeff * self.limiter_envelope
                + (1.0 - self.limiter_release_coeff) * held
        };

        self.smoothing_sum += self.limiter_envelope as f64 - self.smoothing[pos] as f64;
        self.smoothing[pos] = self.limiter_envelope;
        let gain = (self.smoothing_sum / len as f64) as f32;

        // Delay line: read the frame written `len - 1` frames ago (the slot
        // after the write position), then write the current frame.
        let read_slot = (pos + 1) % len;
        let base_write = pos * MAX_CHANNELS;
        let base_read = read_slot * MAX_CHANNELS;
        let total_gain = gain * input_gain;
        let ceiling = self.limiter_ceiling_lin;
        for ch in 0..channels {
            let incoming = frame[ch];
            let delayed = if len == 1 {
                incoming
            } else {
                self.delay[base_read + ch]
            };
            self.delay[base_write + ch] = incoming;
            // Final clamp is only a guard against float rounding.
            frame[ch] = (delayed * total_gain).clamp(-ceiling, ceiling);
        }
        // Channels beyond MAX_CHANNELS are not delayed; silence them rather
        // than emit misaligned audio.
        for sample in frame.iter_mut().skip(channels) {
            *sample = 0.0;
        }

        self.ring_pos = read_slot;
        if self.ring_pos == 0 {
            // Re-sum once per window to stop f64 drift from accumulating.
            self.smoothing_sum = self.smoothing.iter().map(|&g| g as f64).sum();
        }
    }

    fn reset_limiter_state(&mut self) {
        self.required_gain.fill(1.0);
        self.smoothing.fill(1.0);
        self.smoothing_sum = self.lookahead as f64;
        self.delay.fill(0.0);
        self.ring_pos = 0;
        self.limiter_envelope = 1.0;
    }

    fn refresh_coefficients(&mut self) {
        self.upward_detector_coeff = time_to_coeff(UPWARD_DETECTOR_MS, self.sample_rate);
        self.upward_attack_coeff = time_to_coeff(UPWARD_ATTACK_MS, self.sample_rate);
        self.upward_release_coeff = time_to_coeff(self.upward.release_ms, self.sample_rate);
        self.compressor_attack_coeff = time_to_coeff(self.compressor.attack_ms, self.sample_rate);
        self.compressor_release_coeff = time_to_coeff(self.compressor.release_ms, self.sample_rate);
        self.limiter_release_coeff = time_to_coeff(self.limiter.release_ms, self.sample_rate);
        self.limiter_input_gain_lin = db_to_linear(self.limiter.input_gain_db);
        self.limiter_ceiling_lin = db_to_linear(self.limiter.ceiling_db);
    }
}

/// Static upward gain curve (dB of lift for a given detector level).
fn upward_boost_db(settings: &UpwardCompressorSettings, level_db: f32) -> f32 {
    if level_db >= settings.threshold_db || level_db <= UPWARD_FLOOR_DB {
        return 0.0;
    }
    let below = settings.threshold_db - level_db;
    let boost = (below * (1.0 - 1.0 / settings.ratio)).min(settings.max_boost_db);
    let ramp = ((level_db - UPWARD_FLOOR_DB) / UPWARD_FLOOR_RAMP_DB).clamp(0.0, 1.0);
    boost * ramp
}

#[inline]
fn frame_peak(frame: &[f32]) -> f32 {
    frame
        .iter()
        .fold(0.0f32, |max_peak, sample| max_peak.max(sample.abs()))
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn time_to_coeff(milliseconds: f32, sample_rate: u32) -> f32 {
    let seconds = (milliseconds.max(0.1)) / 1000.0;
    (-1.0 / (seconds * sample_rate as f32)).exp()
}

fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.max(EPSILON).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: u32 = 48_000;

    fn stereo_sine(freq: f32, amp: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|n| {
                let s = amp * (2.0 * std::f32::consts::PI * freq * n as f32 / FS as f32).sin();
                [s, s]
            })
            .collect()
    }

    fn peak(buf: &[f32]) -> f32 {
        buf.iter().fold(0.0f32, |m, s| m.max(s.abs()))
    }

    #[test]
    fn everything_disabled_is_bit_exact() {
        let mut chain = DynamicsChain::new(FS);
        let input = stereo_sine(440.0, 0.9, 4096);
        let mut buf = input.clone();
        chain.process(&mut buf, 2);
        assert_eq!(buf, input);
    }

    #[test]
    fn lookahead_limiter_never_overshoots_ceiling() {
        let mut chain = DynamicsChain::new(FS);
        chain.set_limiter(true, 0.0, -1.0, 60.0);
        let ceiling = db_to_linear(-1.0);
        // Quiet bed with sudden full-scale-plus spikes: worst case for a
        // limiter without lookahead.
        let mut buf = stereo_sine(220.0, 0.1, FS as usize);
        for spike in (1000..buf.len()).step_by(9_600) {
            buf[spike] = 1.8;
            buf[spike + 1] = -1.8;
        }
        chain.process(&mut buf, 2);
        let p = peak(&buf);
        assert!(p <= ceiling + 1.0e-5, "peak {p} > ceiling {ceiling}");
    }

    #[test]
    fn lookahead_limiter_brings_gain_down_before_the_peak() {
        let mut chain = DynamicsChain::new(FS);
        chain.set_limiter(true, 0.0, -1.0, 60.0);
        let latency = chain.limiter_latency_frames();
        let frames = 4800;
        let mut buf = vec![0.5f32; frames * 2];
        let hit = 2400;
        buf[hit * 2] = 1.5;
        buf[hit * 2 + 1] = 1.5;
        chain.process(&mut buf, 2);
        // The frame just before the (delayed) peak is already attenuated.
        let before = buf[(hit + latency - 1) * 2];
        assert!(before < 0.5, "gain not ramping down before peak: {before}");
        let at_peak = buf[(hit + latency) * 2];
        assert!(at_peak <= db_to_linear(-1.0) + 1.0e-5);
    }

    #[test]
    fn limiter_leaves_quiet_signal_alone_apart_from_delay() {
        let mut chain = DynamicsChain::new(FS);
        chain.set_limiter(true, 0.0, -1.0, 60.0);
        let latency = chain.limiter_latency_frames();
        let input = stereo_sine(440.0, 0.3, 4096);
        let mut buf = input.clone();
        chain.process(&mut buf, 2);
        for n in 0..(4096 - latency) {
            let expected = input[n * 2];
            let got = buf[(n + latency) * 2];
            assert!((expected - got).abs() < 1.0e-6, "frame {n}: {expected} vs {got}");
        }
    }

    #[test]
    fn upward_lifts_quiet_and_leaves_loud_alone() {
        let mut quiet_chain = DynamicsChain::new(FS);
        quiet_chain.set_upward_compressor(true, -30.0, 2.0, 8.0, 300.0);
        let amp_quiet = db_to_linear(-40.0);
        let mut quiet = stereo_sine(500.0, amp_quiet, FS as usize * 2);
        quiet_chain.process(&mut quiet, 2);
        let tail = &quiet[quiet.len() - FS as usize..];
        let lift_db = linear_to_db(peak(tail) / amp_quiet);
        // 10 dB under threshold at 2:1 -> ~5 dB lift.
        assert!((lift_db - 5.0).abs() < 0.8, "lift {lift_db}");

        let mut loud_chain = DynamicsChain::new(FS);
        loud_chain.set_upward_compressor(true, -30.0, 2.0, 8.0, 300.0);
        let input = stereo_sine(500.0, 0.5, FS as usize);
        let mut loud = input.clone();
        loud_chain.process(&mut loud, 2);
        assert_eq!(loud, input);
    }

    #[test]
    fn upward_respects_max_boost_and_floor() {
        let s = UpwardCompressorSettings {
            enabled: true,
            threshold_db: -30.0,
            ratio: 4.0,
            max_boost_db: 6.0,
            release_ms: 300.0,
        };
        assert!((upward_boost_db(&s, -50.0) - 6.0).abs() < 1.0e-4);
        assert_eq!(upward_boost_db(&s, -80.0), 0.0);
        assert_eq!(upward_boost_db(&s, -20.0), 0.0);
        assert!(upward_boost_db(&s, -66.0) < 6.0);
    }

    #[test]
    fn upward_pulls_back_fast_when_loud_part_arrives() {
        let mut chain = DynamicsChain::new(FS);
        chain.set_upward_compressor(true, -30.0, 2.0, 8.0, 300.0);
        let mut quiet = stereo_sine(500.0, db_to_linear(-40.0), FS as usize);
        chain.process(&mut quiet, 2);
        let mut loud = stereo_sine(500.0, 0.5, 2400); // 50 ms
        chain.process(&mut loud, 2);
        // After ~25 ms of loud signal the lift should be essentially gone.
        let late = &loud[1200 * 2..];
        assert!(peak(late) < 0.5 * db_to_linear(0.5), "peak {}", peak(late));
    }

    #[test]
    fn toggling_limiter_resets_delay() {
        let mut chain = DynamicsChain::new(FS);
        chain.set_limiter(true, 0.0, -1.0, 60.0);
        let mut buf = stereo_sine(440.0, 0.9, 512);
        chain.process(&mut buf, 2);
        chain.set_limiter(false, 0.0, -1.0, 60.0);
        assert!(chain.delay.iter().all(|&s| s == 0.0));
    }
}
