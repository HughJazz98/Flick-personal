//! Soft saturation ("drive") stage for the realtime callback.
//!
//! Sits right after the EQ/preamp so that preamp overs are rounded off
//! smoothly instead of being hard-clipped by later stages.
//!
//! - `drive` (0..1) moves the soft-knee threshold down from -0.9 dBFS (only
//!   peaks get rounded) to about -10 dBFS (audible saturation).
//!   Below the knee the signal is untouched; above it a tanh curve bends the
//!   waveform toward an asymptote at full scale, so the output never exceeds
//!   ±1.0 no matter how hot the input is.
//! - `warmth` (0..1) adds a small second-order (even-harmonic) term before the
//!   knee, the "tube-ish" colouring. It scales with level, so quiet passages
//!   stay clean. A DC blocker removes the offset that term introduces.
//!
//! No allocations in `process`; per-channel state lives in fixed arrays.

const MAX_CHANNELS: usize = 8;
const KNEE_AT_ZERO_DRIVE: f32 = 0.90;
const KNEE_AT_FULL_DRIVE: f32 = 0.30;
const MAX_WARMTH_COEFF: f32 = 0.20;
const DC_BLOCK_HZ: f32 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaturationSettings {
    pub enabled: bool,
    pub drive: f32,
    pub warmth: f32,
}

impl SaturationSettings {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            drive: 0.25,
            warmth: 0.3,
        }
    }
}

pub struct Saturator {
    sample_rate: u32,
    settings: SaturationSettings,
    knee: f32,
    inv_span: f32,
    span: f32,
    warmth_coeff: f32,
    dc_coeff: f32,
    dc_x1: [f32; MAX_CHANNELS],
    dc_y1: [f32; MAX_CHANNELS],
}

impl Saturator {
    pub fn new(sample_rate: u32) -> Self {
        let mut saturator = Self {
            sample_rate: sample_rate.max(1),
            settings: SaturationSettings::disabled(),
            knee: KNEE_AT_ZERO_DRIVE,
            inv_span: 0.0,
            span: 0.0,
            warmth_coeff: 0.0,
            dc_coeff: 0.0,
            dc_x1: [0.0; MAX_CHANNELS],
            dc_y1: [0.0; MAX_CHANNELS],
        };
        saturator.refresh();
        saturator
    }

    pub fn reconfigure_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate.max(1);
        self.refresh();
        self.reset_state();
    }

    pub fn settings(&self) -> SaturationSettings {
        self.settings
    }

    pub fn set(&mut self, enabled: bool, drive: f32, warmth: f32) {
        let was_enabled = self.settings.enabled;
        self.settings = SaturationSettings {
            enabled,
            drive: sanitize_unit(drive),
            warmth: sanitize_unit(warmth),
        };
        self.refresh();
        if enabled && !was_enabled {
            self.reset_state();
        }
    }

    pub fn process(&mut self, buf: &mut [f32], channels: usize) {
        if !self.settings.enabled || channels == 0 {
            return;
        }
        let use_warmth = self.warmth_coeff > 0.0;
        let tracked = channels.min(MAX_CHANNELS);

        for frame in buf.chunks_exact_mut(channels) {
            for (ch, sample) in frame.iter_mut().enumerate() {
                let mut x = *sample;
                if !x.is_finite() {
                    x = 0.0;
                }
                if use_warmth {
                    // Even-harmonic colouring; DC is removed below.
                    x += self.warmth_coeff * x * x;
                }
                let mut y = self.shape(x);
                if use_warmth && ch < tracked {
                    // One-pole DC blocker: y[n] = x[n] - x[n-1] + R * y[n-1]
                    let out = y - self.dc_x1[ch] + self.dc_coeff * self.dc_y1[ch];
                    self.dc_x1[ch] = y;
                    self.dc_y1[ch] = out;
                    y = out;
                }
                *sample = y;
            }
        }
    }

    #[inline]
    fn shape(&self, x: f32) -> f32 {
        let magnitude = x.abs();
        if magnitude <= self.knee {
            return x;
        }
        let over = (magnitude - self.knee) * self.inv_span;
        let shaped = self.knee + self.span * over.tanh();
        shaped.copysign(x)
    }

    fn refresh(&mut self) {
        let drive = self.settings.drive;
        self.knee = KNEE_AT_ZERO_DRIVE + (KNEE_AT_FULL_DRIVE - KNEE_AT_ZERO_DRIVE) * drive;
        self.span = 1.0 - self.knee;
        self.inv_span = 1.0 / self.span;
        self.warmth_coeff = MAX_WARMTH_COEFF * self.settings.warmth;
        let fs = self.sample_rate as f32;
        self.dc_coeff = (-2.0 * std::f32::consts::PI * DC_BLOCK_HZ / fs).exp();
    }

    fn reset_state(&mut self) {
        self.dc_x1 = [0.0; MAX_CHANNELS];
        self.dc_y1 = [0.0; MAX_CHANNELS];
    }
}

fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, amp: f32, fs: u32, frames: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|n| {
                let s = amp * (2.0 * std::f32::consts::PI * freq * n as f32 / fs as f32).sin();
                [s, s]
            })
            .collect()
    }

    #[test]
    fn disabled_is_bit_exact() {
        let mut sat = Saturator::new(48_000);
        let input = sine(440.0, 1.4, 48_000, 2048);
        let mut buf = input.clone();
        sat.process(&mut buf, 2);
        assert_eq!(buf, input);
    }

    #[test]
    fn hot_input_never_exceeds_full_scale() {
        for drive in [0.0, 0.25, 0.5, 1.0] {
            let mut sat = Saturator::new(48_000);
            sat.set(true, drive, 0.0);
            let mut buf = sine(440.0, 2.0, 48_000, 4096);
            sat.process(&mut buf, 2);
            let peak = buf.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!(peak <= 1.0, "drive {drive}: peak {peak}");
        }
    }

    #[test]
    fn signal_below_knee_is_untouched_without_warmth() {
        let mut sat = Saturator::new(48_000);
        sat.set(true, 0.25, 0.0); // knee = 0.75
        let input = sine(1000.0, 0.5, 48_000, 2048);
        let mut buf = input.clone();
        sat.process(&mut buf, 2);
        assert_eq!(buf, input);
    }

    #[test]
    fn shaper_is_continuous_at_the_knee() {
        let mut sat = Saturator::new(48_000);
        sat.set(true, 0.5, 0.0);
        let k = sat.knee;
        let below = sat.shape(k - 1.0e-4);
        let above = sat.shape(k + 1.0e-4);
        assert!((above - below).abs() < 5.0e-4);
    }

    #[test]
    fn warmth_output_has_no_dc_after_settling() {
        let fs = 48_000;
        let mut sat = Saturator::new(fs);
        sat.set(true, 0.25, 1.0);
        let mut buf = sine(200.0, 0.7, fs, fs as usize * 2);
        sat.process(&mut buf, 2);
        let tail = &buf[buf.len() - fs as usize..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(mean.abs() < 1.0e-3, "dc {mean}");
    }

    #[test]
    fn non_finite_input_is_silenced() {
        let mut sat = Saturator::new(48_000);
        sat.set(true, 0.5, 0.0);
        let mut buf = vec![f32::NAN, f32::INFINITY];
        sat.process(&mut buf, 2);
        assert!(buf.iter().all(|s| s.is_finite()));
    }
}
