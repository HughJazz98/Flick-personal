import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

/// Cross-service flag: parametric EQ wants the Rust engine so the full
/// variable-band DSP chain runs instead of the device's fixed-band AudioEffect.
class EqEngineHint {
  EqEngineHint._();

  static bool parametricPeqActive = false;

  /// Saturation or the upward compressor is on. These exist only in the Rust
  /// engine, so unlike parametric EQ (which the Android path approximates)
  /// they pull playback onto it on every non-Bluetooth route.
  /// Null until the equalizer has applied its state once this session.
  static bool? rustOnlyDspActive;

  /// The live flag once the equalizer has applied its state; before that
  /// (app start, when the route is first resolved) the persisted EQ state.
  static Future<bool> rustOnlyDspRequested() async {
    final live = rustOnlyDspActive;
    if (live != null) return live;
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString('eq_state_v1');
      if (raw == null || raw.isEmpty) return false;
      final json = jsonDecode(raw) as Map<String, dynamic>;
      if (json['enabled'] == false) return false;
      bool on(String key) => (json[key] as Map?)?['enabled'] == true;
      return on('saturation') || on('upwardCompressor');
    } catch (_) {
      return false;
    }
  }
}
