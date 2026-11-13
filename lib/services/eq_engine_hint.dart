/// Cross-service flag: parametric EQ wants the Rust engine so the full
/// variable-band DSP chain runs instead of the device's fixed-band AudioEffect.
class EqEngineHint {
  EqEngineHint._();

  static bool parametricPeqActive = false;

  /// Saturation or the upward compressor is on. These exist only in the Rust
  /// engine, so unlike parametric EQ (which the Android path approximates)
  /// they pull playback onto it on every non-Bluetooth route.
  static bool rustOnlyDspActive = false;
}
