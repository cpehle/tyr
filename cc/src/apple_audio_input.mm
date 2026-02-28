#include <lean/lean.h>

#include <deque>
#include <mutex>
#include <thread>
#include <chrono>
#include <string>

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>

namespace {

std::mutex g_audio_mu;
std::deque<float> g_audio_fifo;
AudioQueueRef g_input_queue = nullptr;
bool g_running = false;
uint64_t g_target_rate = 16000;
uint64_t g_target_channels = 1;

static lean_object* mk_io_error(const std::string& msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

static void input_callback(
    void* /*inUserData*/,
    AudioQueueRef inAQ,
    AudioQueueBufferRef inBuffer,
    const AudioTimeStamp* /*inStartTime*/,
    UInt32 inNumPackets,
    const AudioStreamPacketDescription* /*inPacketDesc*/) {
  if (inNumPackets > 0 && inBuffer->mAudioData != nullptr) {
    const size_t n = static_cast<size_t>(inBuffer->mAudioDataByteSize / sizeof(float));
    const float* p = static_cast<const float*>(inBuffer->mAudioData);
    std::lock_guard<std::mutex> lock(g_audio_mu);
    for (size_t i = 0; i < n; ++i) {
      g_audio_fifo.push_back(p[i]);
    }
    // Keep ~30s cap to avoid unbounded growth.
    const size_t cap = static_cast<size_t>(g_target_rate * g_target_channels * 30);
    while (g_audio_fifo.size() > cap) {
      g_audio_fifo.pop_front();
    }
  }

  if (g_running) {
    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, nullptr);
  }
}

static void stop_queue() {
  if (g_input_queue != nullptr) {
    g_running = false;
    AudioQueueStop(g_input_queue, true);
    AudioQueueDispose(g_input_queue, true);
    g_input_queue = nullptr;
  }
}

} // namespace

extern "C" {

lean_object* lean_tyr_audio_input_start(
    uint64_t sample_rate,
    uint64_t channels,
    uint64_t buffer_ms,
    lean_object* /*w*/) {
  if (sample_rate == 0 || channels == 0) {
    return mk_io_error("audio_input_start: sample_rate/channels must be > 0");
  }

  stop_queue();
  {
    std::lock_guard<std::mutex> lock(g_audio_mu);
    g_audio_fifo.clear();
  }

  g_target_rate = sample_rate;
  g_target_channels = channels;

  AudioStreamBasicDescription fmt{};
  fmt.mSampleRate = static_cast<Float64>(sample_rate);
  fmt.mFormatID = kAudioFormatLinearPCM;
  fmt.mFormatFlags = kLinearPCMFormatFlagIsFloat | kAudioFormatFlagIsPacked;
  fmt.mFramesPerPacket = 1;
  fmt.mChannelsPerFrame = static_cast<UInt32>(channels);
  fmt.mBitsPerChannel = 32;
  fmt.mBytesPerFrame = fmt.mChannelsPerFrame * sizeof(float);
  fmt.mBytesPerPacket = fmt.mFramesPerPacket * fmt.mBytesPerFrame;

  OSStatus st = AudioQueueNewInput(
      &fmt,
      input_callback,
      nullptr,
      nullptr,
      kCFRunLoopCommonModes,
      0,
      &g_input_queue);
  if (st != noErr || g_input_queue == nullptr) {
    g_input_queue = nullptr;
    return mk_io_error("audio_input_start: AudioQueueNewInput failed");
  }

  const uint64_t ms = (buffer_ms == 0) ? 100 : buffer_ms;
  uint64_t frames = (sample_rate * ms) / 1000;
  if (frames < 256) frames = 256;
  UInt32 bytes = static_cast<UInt32>(frames * channels * sizeof(float));

  for (int i = 0; i < 3; ++i) {
    AudioQueueBufferRef buf = nullptr;
    st = AudioQueueAllocateBuffer(g_input_queue, bytes, &buf);
    if (st != noErr || buf == nullptr) {
      stop_queue();
      return mk_io_error("audio_input_start: AudioQueueAllocateBuffer failed");
    }
    buf->mAudioDataByteSize = bytes;
    st = AudioQueueEnqueueBuffer(g_input_queue, buf, 0, nullptr);
    if (st != noErr) {
      stop_queue();
      return mk_io_error("audio_input_start: AudioQueueEnqueueBuffer failed");
    }
  }

  st = AudioQueueStart(g_input_queue, nullptr);
  if (st != noErr) {
    stop_queue();
    return mk_io_error("audio_input_start: AudioQueueStart failed (check microphone permission)");
  }

  g_running = true;
  return lean_io_result_mk_ok(lean_box(0));
}

lean_object* lean_tyr_audio_input_read(
    uint64_t max_samples,
    uint64_t block_ms,
    lean_object* /*w*/) {
  if (max_samples == 0) {
    return lean_io_result_mk_ok(lean_mk_empty_array());
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(block_ms);
  while (true) {
    {
      std::lock_guard<std::mutex> lock(g_audio_mu);
      if (!g_audio_fifo.empty() || !g_running) break;
    }
    if (block_ms == 0 || std::chrono::steady_clock::now() >= deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  lean_object* out = lean_mk_empty_array();
  {
    std::lock_guard<std::mutex> lock(g_audio_mu);
    size_t n = std::min(static_cast<size_t>(max_samples), g_audio_fifo.size());
    for (size_t i = 0; i < n; ++i) {
      out = lean_array_push(out, lean_box_float(static_cast<double>(g_audio_fifo.front())));
      g_audio_fifo.pop_front();
    }
  }
  return lean_io_result_mk_ok(out);
}

lean_object* lean_tyr_audio_input_stop(lean_object* /*w*/) {
  stop_queue();
  std::lock_guard<std::mutex> lock(g_audio_mu);
  g_audio_fifo.clear();
  return lean_io_result_mk_ok(lean_box(0));
}

} // extern "C"

#else

extern "C" {

lean_object* lean_tyr_audio_input_start(uint64_t, uint64_t, uint64_t, lean_object* /*w*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string("audio input is only supported on macOS")));
}

lean_object* lean_tyr_audio_input_read(uint64_t, uint64_t, lean_object* /*w*/) {
  return lean_io_result_mk_ok(lean_mk_empty_array());
}

lean_object* lean_tyr_audio_input_stop(lean_object* /*w*/) {
  return lean_io_result_mk_ok(lean_box(0));
}

} // extern "C"

#endif
