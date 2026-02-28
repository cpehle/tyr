/*
 * apple_vl_processor.mm
 *
 * Apple-only image/video preprocessing for Qwen3.5-VL patch inputs.
 * No fallback path: requires macOS system media frameworks.
 */

#include <lean/lean.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#ifdef __APPLE__
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

// Defined in tyr.cpp
lean_object *fromTorchTensor(torch::Tensor t);

static lean_object* mk_io_error(const std::string& msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

static bool load_image_rgb_f32(
    const std::string& path,
    int& width,
    int& height,
    std::vector<float>& rgb,
    std::string& err) {
  @autoreleasepool {
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    if (url == nil) {
      err = "Invalid image path URL";
      return false;
    }

    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (src == nullptr) {
      err = "Failed to open image with ImageIO";
      return false;
    }

    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (img == nullptr) {
      err = "Failed to decode image";
      return false;
    }

    width = static_cast<int>(CGImageGetWidth(img));
    height = static_cast<int>(CGImageGetHeight(img));
    if (width <= 0 || height <= 0) {
      CGImageRelease(img);
      err = "Decoded image has invalid dimensions";
      return false;
    }

    std::vector<uint8_t> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
      rgba.data(),
      width,
      height,
      8,
      width * 4,
      colorSpace,
      kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
    );
    CGColorSpaceRelease(colorSpace);
    if (ctx == nullptr) {
      CGImageRelease(img);
      err = "Failed to create bitmap context";
      return false;
    }

    CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), img);
    CGContextRelease(ctx);
    CGImageRelease(img);

    rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    for (size_t i = 0; i < static_cast<size_t>(width) * static_cast<size_t>(height); ++i) {
      rgb[i * 3 + 0] = static_cast<float>(rgba[i * 4 + 0]) / 255.0f;
      rgb[i * 3 + 1] = static_cast<float>(rgba[i * 4 + 1]) / 255.0f;
      rgb[i * 3 + 2] = static_cast<float>(rgba[i * 4 + 2]) / 255.0f;
    }
  }
  return true;
}

static bool load_video_rgb_f32_frames(
    const std::string& path,
    uint64_t max_frames,
    int& width,
    int& height,
    std::vector<std::vector<float>>& frames,
    std::string& err) {
  @autoreleasepool {
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    if (url == nil) {
      err = "Invalid video path URL";
      return false;
    }

    AVURLAsset* asset = [AVURLAsset URLAssetWithURL:url options:nil];
    if (asset == nil) {
      err = "Failed to open video asset";
      return false;
    }

    NSArray<AVAssetTrack*>* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
    if (tracks.count == 0) {
      err = "No video track found";
      return false;
    }
    AVAssetTrack* track = tracks.firstObject;

    NSError* nsErr = nil;
    AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:&nsErr];
    if (reader == nil) {
      err = nsErr != nil ? std::string([[nsErr localizedDescription] UTF8String]) : "Failed to create AVAssetReader";
      return false;
    }

    NSDictionary* outputSettings = @{
      (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
    };
    AVAssetReaderTrackOutput* output = [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:outputSettings];
    output.alwaysCopiesSampleData = NO;
    if (![reader canAddOutput:output]) {
      err = "Cannot add AVAssetReader track output";
      return false;
    }
    [reader addOutput:output];

    if (![reader startReading]) {
      err = "AVAssetReader failed to start";
      return false;
    }

    frames.clear();
    width = 0;
    height = 0;
    while (reader.status == AVAssetReaderStatusReading) {
      CMSampleBufferRef sample = [output copyNextSampleBuffer];
      if (sample == nullptr) {
        break;
      }

      CVImageBufferRef imgBuf = CMSampleBufferGetImageBuffer(sample);
      if (imgBuf == nullptr) {
        CFRelease(sample);
        continue;
      }
      CVPixelBufferRef px = (CVPixelBufferRef)imgBuf;
      CVPixelBufferLockBaseAddress(px, kCVPixelBufferLock_ReadOnly);

      int fw = static_cast<int>(CVPixelBufferGetWidth(px));
      int fh = static_cast<int>(CVPixelBufferGetHeight(px));
      if (fw <= 0 || fh <= 0) {
        CVPixelBufferUnlockBaseAddress(px, kCVPixelBufferLock_ReadOnly);
        CFRelease(sample);
        continue;
      }
      if (width == 0 && height == 0) {
        width = fw;
        height = fh;
      }
      if (fw != width || fh != height) {
        CVPixelBufferUnlockBaseAddress(px, kCVPixelBufferLock_ReadOnly);
        CFRelease(sample);
        err = "Video frames have inconsistent dimensions";
        return false;
      }

      uint8_t* base = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(px));
      size_t bpr = CVPixelBufferGetBytesPerRow(px);

      std::vector<float> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
      for (int y = 0; y < height; ++y) {
        uint8_t* row = base + static_cast<size_t>(y) * bpr;
        for (int x = 0; x < width; ++x) {
          uint8_t b = row[x * 4 + 0];
          uint8_t g = row[x * 4 + 1];
          uint8_t r = row[x * 4 + 2];
          size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
          rgb[idx + 0] = static_cast<float>(r) / 255.0f;
          rgb[idx + 1] = static_cast<float>(g) / 255.0f;
          rgb[idx + 2] = static_cast<float>(b) / 255.0f;
        }
      }

      CVPixelBufferUnlockBaseAddress(px, kCVPixelBufferLock_ReadOnly);
      CFRelease(sample);

      frames.push_back(std::move(rgb));
      if (max_frames > 0 && frames.size() >= max_frames) {
        break;
      }
    }

    if (reader.status == AVAssetReaderStatusFailed) {
      err = "AVAssetReader failed while decoding video";
      return false;
    }

    if (frames.empty()) {
      err = "No decodable frames found in video";
      return false;
    }
  }
  return true;
}

static bool patchify_rgb_frames(
    const std::vector<std::vector<float>>& frames,
    int width,
    int height,
    uint64_t in_channels,
    uint64_t patch_size,
    uint64_t temporal_patch_size,
    std::vector<float>& out,
    uint64_t& n_patches,
    std::string& err) {
  if (in_channels != 3) {
    err = "Apple media preprocessor currently supports in_channels=3 only";
    return false;
  }
  if (patch_size == 0 || temporal_patch_size == 0) {
    err = "patch_size and temporal_patch_size must be > 0";
    return false;
  }
  if (frames.empty()) {
    err = "No frames available for patchify";
    return false;
  }

  int h_eff = (height / static_cast<int>(patch_size)) * static_cast<int>(patch_size);
  int w_eff = (width / static_cast<int>(patch_size)) * static_cast<int>(patch_size);
  if (h_eff <= 0 || w_eff <= 0) {
    err = "Image/video dimensions are smaller than patch_size";
    return false;
  }
  int y_off = (height - h_eff) / 2;
  int x_off = (width - w_eff) / 2;

  uint64_t groups =
      (static_cast<uint64_t>(frames.size()) + temporal_patch_size - 1) / temporal_patch_size;
  uint64_t patches_per_group =
      static_cast<uint64_t>(h_eff / static_cast<int>(patch_size)) *
      static_cast<uint64_t>(w_eff / static_cast<int>(patch_size));
  n_patches = groups * patches_per_group;

  uint64_t patch_dim = in_channels * temporal_patch_size * patch_size * patch_size;
  out.assign(static_cast<size_t>(n_patches) * static_cast<size_t>(patch_dim), 0.0f);

  uint64_t patch_idx = 0;
  for (uint64_t g = 0; g < groups; ++g) {
    for (int py = 0; py < h_eff / static_cast<int>(patch_size); ++py) {
      for (int px = 0; px < w_eff / static_cast<int>(patch_size); ++px) {
        size_t base = static_cast<size_t>(patch_idx) * static_cast<size_t>(patch_dim);
        size_t k = 0;
        for (uint64_t c = 0; c < in_channels; ++c) {
          for (uint64_t t = 0; t < temporal_patch_size; ++t) {
            uint64_t frame_idx = g * temporal_patch_size + t;
            if (frame_idx >= frames.size()) {
              frame_idx = static_cast<uint64_t>(frames.size() - 1);
            }
            const auto& frame = frames[static_cast<size_t>(frame_idx)];
            for (uint64_t dy = 0; dy < patch_size; ++dy) {
              for (uint64_t dx = 0; dx < patch_size; ++dx) {
                int y = y_off + py * static_cast<int>(patch_size) + static_cast<int>(dy);
                int x = x_off + px * static_cast<int>(patch_size) + static_cast<int>(dx);
                size_t pix = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3 + static_cast<size_t>(c);
                out[base + k] = frame[pix];
                ++k;
              }
            }
          }
        }
        ++patch_idx;
      }
    }
  }

  return true;
}

extern "C" {

lean_object* lean_torch_media_load_image_patchified(
    b_lean_obj_arg path_obj,
    uint64_t in_channels,
    uint64_t patch_size,
    uint64_t temporal_patch_size,
    lean_object* /*w*/) {
  const char* path_c = lean_string_cstr(path_obj);
  std::string path(path_c);

  int width = 0;
  int height = 0;
  std::vector<float> rgb;
  std::string err;
  if (!load_image_rgb_f32(path, width, height, rgb, err)) {
    return mk_io_error("loadImagePatchified failed: " + err);
  }

  std::vector<std::vector<float>> frames;
  frames.push_back(std::move(rgb));

  std::vector<float> patches;
  uint64_t n_patches = 0;
  if (!patchify_rgb_frames(
        frames, width, height, in_channels, patch_size, temporal_patch_size,
        patches, n_patches, err)) {
    return mk_io_error("loadImagePatchified failed: " + err);
  }

  uint64_t patch_dim = in_channels * temporal_patch_size * patch_size * patch_size;
  auto t = torch::from_blob(
      patches.data(),
      {static_cast<int64_t>(n_patches), static_cast<int64_t>(patch_dim)},
      torch::TensorOptions().dtype(torch::kFloat32)).clone();
  return lean_io_result_mk_ok(fromTorchTensor(t));
}

lean_object* lean_torch_media_load_video_patchified(
    b_lean_obj_arg path_obj,
    uint64_t in_channels,
    uint64_t patch_size,
    uint64_t temporal_patch_size,
    uint64_t max_frames,
    lean_object* /*w*/) {
  const char* path_c = lean_string_cstr(path_obj);
  std::string path(path_c);

  int width = 0;
  int height = 0;
  std::vector<std::vector<float>> frames;
  std::string err;
  if (!load_video_rgb_f32_frames(path, max_frames, width, height, frames, err)) {
    return mk_io_error("loadVideoPatchified failed: " + err);
  }

  std::vector<float> patches;
  uint64_t n_patches = 0;
  if (!patchify_rgb_frames(
        frames, width, height, in_channels, patch_size, temporal_patch_size,
        patches, n_patches, err)) {
    return mk_io_error("loadVideoPatchified failed: " + err);
  }

  uint64_t patch_dim = in_channels * temporal_patch_size * patch_size * patch_size;
  auto t = torch::from_blob(
      patches.data(),
      {static_cast<int64_t>(n_patches), static_cast<int64_t>(patch_dim)},
      torch::TensorOptions().dtype(torch::kFloat32)).clone();
  return lean_io_result_mk_ok(fromTorchTensor(t));
}

} // extern "C"

#else

extern "C" {

lean_object* lean_torch_media_load_image_patchified(
    b_lean_obj_arg /*path_obj*/,
    uint64_t /*in_channels*/,
    uint64_t /*patch_size*/,
    uint64_t /*temporal_patch_size*/,
    lean_object* /*w*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("Apple media path requires macOS build")));
}

lean_object* lean_torch_media_load_video_patchified(
    b_lean_obj_arg /*path_obj*/,
    uint64_t /*in_channels*/,
    uint64_t /*patch_size*/,
    uint64_t /*temporal_patch_size*/,
    uint64_t /*max_frames*/,
    lean_object* /*w*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("Apple media path requires macOS build")));
}

} // extern "C"

#endif

