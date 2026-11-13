//! Announces the Rust engine's Android audio session to third-party effect
//! apps (Wavelet, Poweramp EQ, Sound Beautifier, ...).
//!
//! Those apps don't hook "Flick", they hook an audio session: a player sends
//! `AudioEffect.ACTION_OPEN_AUDIO_EFFECT_CONTROL_SESSION` with its session id,
//! and the effect app attaches its EQ/bass boost to that session. ExoPlayer
//! does this, which is why such apps work on the Standard engine. The shared
//! Oboe stream now asks Android to allocate a session and this module
//! broadcasts it on open and `..._CLOSE_...` when the stream goes away.

#![cfg_attr(not(target_os = "android"), allow(dead_code))]

/// `AudioEffect.CONTENT_TYPE_MUSIC`.
const CONTENT_TYPE_MUSIC: i32 = 0;
const ACTION_OPEN: &str = "android.media.action.OPEN_AUDIO_EFFECT_CONTROL_SESSION";
const ACTION_CLOSE: &str = "android.media.action.CLOSE_AUDIO_EFFECT_CONTROL_SESSION";
const EXTRA_AUDIO_SESSION: &str = "android.media.extra.AUDIO_SESSION";
const EXTRA_PACKAGE_NAME: &str = "android.media.extra.PACKAGE_NAME";
const EXTRA_CONTENT_TYPE: &str = "android.media.extra.CONTENT_TYPE";

/// Tell effect apps a session opened. `session` <= 0 means none was allocated.
pub fn announce_open(session: i32) {
    announce(true, session);
}

/// Tell effect apps the session is gone so they detach.
pub fn announce_close(session: i32) {
    announce(false, session);
}

/// Runs on its own short-lived thread: the caller is the engine's stream
/// setup/teardown path, which must never block on, or be unwound by, a JNI
/// problem. Any panic in here is contained to that thread.
#[cfg(target_os = "android")]
fn announce(open: bool, session: i32) {
    if session <= 0 {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("flick-effect-session".into())
        .spawn(move || {
            if std::panic::catch_unwind(|| announce_blocking(open, session)).is_err() {
                log::warn!("[EffectSession] announce panicked for session {session}");
            }
        });
    if let Err(error) = spawned {
        log::warn!("[EffectSession] could not spawn announce thread: {error}");
    }
}

#[cfg(target_os = "android")]
fn announce_blocking(open: bool, session: i32) {
    let ctx = ndk_context::android_context();
    if ctx.context().is_null() {
        log::warn!("[EffectSession] no Android context; cannot announce session {session}");
        return;
    }
    let vm = unsafe { jni::JavaVM::from_raw(ctx.vm().cast()) };
    let result = vm.attach_current_thread(|env| {
        let context =
            unsafe { jni::objects::JObject::from_raw(env, ctx.context() as jni::sys::jobject) };
        let outcome = send_broadcast(env, &context, open, session);
        let _ = env.exception_clear();
        outcome
    });
    match result {
        Ok(()) => log::info!(
            "[EffectSession] {} audio effect session {session}",
            if open { "opened" } else { "closed" }
        ),
        Err(error) => log::warn!(
            "[EffectSession] failed to announce session {session} ({}): {error}",
            if open { "open" } else { "close" }
        ),
    }
}

#[cfg(not(target_os = "android"))]
fn announce(_open: bool, _session: i32) {}

/// Builds and sends the Intent.
#[cfg(target_os = "android")]
fn send_broadcast<'local>(
    env: &mut jni::Env<'local>,
    context: &jni::objects::JObject<'local>,
    open: bool,
    session: i32,
) -> Result<(), jni::errors::Error> {
    use jni::objects::JValue;
    use jni::{jni_sig, jni_str};

    let action = env.new_string(if open { ACTION_OPEN } else { ACTION_CLOSE })?;
    let intent = env.new_object(
        jni_str!("android/content/Intent"),
        jni_sig!("(Ljava/lang/String;)V"),
        &[JValue::Object(&action)],
    )?;

    let package = env
        .call_method(
            context,
            jni_str!("getPackageName"),
            jni_sig!("()Ljava/lang/String;"),
            &[],
        )?
        .l()?;

    let session_key = env.new_string(EXTRA_AUDIO_SESSION)?;
    env.call_method(
        &intent,
        jni_str!("putExtra"),
        jni_sig!("(Ljava/lang/String;I)Landroid/content/Intent;"),
        &[JValue::Object(&session_key), JValue::Int(session)],
    )?;

    let package_key = env.new_string(EXTRA_PACKAGE_NAME)?;
    env.call_method(
        &intent,
        jni_str!("putExtra"),
        jni_sig!("(Ljava/lang/String;Ljava/lang/String;)Landroid/content/Intent;"),
        &[JValue::Object(&package_key), JValue::Object(&package)],
    )?;

    let content_key = env.new_string(EXTRA_CONTENT_TYPE)?;
    env.call_method(
        &intent,
        jni_str!("putExtra"),
        jni_sig!("(Ljava/lang/String;I)Landroid/content/Intent;"),
        &[JValue::Object(&content_key), JValue::Int(CONTENT_TYPE_MUSIC)],
    )?;

    env.call_method(
        context,
        jni_str!("sendBroadcast"),
        jni_sig!("(Landroid/content/Intent;)V"),
        &[JValue::Object(&intent)],
    )?;
    Ok(())
}
