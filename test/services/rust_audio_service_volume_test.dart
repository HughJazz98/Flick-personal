import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';

import 'package:flick/services/rust_audio_service.dart';

/// Mirrors `volume_to_gain` in rust/src/audio/engine.rs (below unity).
double rustCurveGain(double v) =>
    v <= 0.0 ? 0.0 : (v <= 1.0 ? math.pow(10, 2.0 * (v - 1.0)).toDouble() : v);

void main() {
  test('Rust engine ends up at the same linear gain ExoPlayer applies', () {
    for (final linear in [1.0, 0.9, 0.7, 0.5, 0.25, 0.05]) {
      final gain = rustCurveGain(RustAudioService.linearToRustVolume(linear));
      expect(gain, closeTo(linear, 1e-9), reason: 'slider $linear');
    }
  });

  test('extended volume and silence pass through', () {
    expect(RustAudioService.linearToRustVolume(1.5), 1.5);
    expect(RustAudioService.linearToRustVolume(0.0), 0.0);
  });
}
