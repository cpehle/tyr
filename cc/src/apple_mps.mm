/*
 * apple_mps.mm
 *
 * Apple-only MPS availability probing with an Objective-C autorelease pool.
 * This avoids flaky pure-C++ startup checks on newer macOS / libtorch combos.
 */

#include <torch/torch.h>
#include <torch/mps.h>

#include <chrono>
#include <exception>
#include <iostream>
#include <thread>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

static __attribute__((noinline,optnone)) bool tyr_has_metal_device() {
#ifdef __APPLE__
  @autoreleasepool {
    id<MTLDevice> metal_device = MTLCreateSystemDefaultDevice();
    return metal_device != nil;
  }
#else
  return false;
#endif
}

static __attribute__((noinline,optnone)) bool tyr_has_supported_macos_version(int debug_flag) {
#ifdef __APPLE__
  @autoreleasepool {
    NSOperatingSystemVersion version = [[NSProcessInfo processInfo] operatingSystemVersion];
    if (debug_flag != 0) {
      std::cerr << "[tyr:mps] macOS version="
                << version.majorVersion << "."
                << version.minorVersion << "."
                << version.patchVersion << std::endl;
    }
    return version.majorVersion >= 14;
  }
#else
  (void)debug_flag;
  return false;
#endif
}

extern "C" __attribute__((noinline,optnone)) bool tyr_apple_mps_is_available(int debug_flag) {
#ifdef __APPLE__
  @autoreleasepool {
    const bool debug = debug_flag != 0;
    if (!tyr_has_supported_macos_version(debug_flag)) {
      if (debug) {
        std::cerr << "[tyr:mps] backend requires macOS 14.0+" << std::endl;
      }
      return false;
    }
    const bool has_metal_device = tyr_has_metal_device();
    if (debug) {
      std::cerr << "[tyr:mps] MTLCreateSystemDefaultDevice()=" << has_metal_device << std::endl;
    }
    if (!has_metal_device) {
      return false;
    }

    // Give the libtorch MPS backend a brief startup window before the first
    // availability query. On recent macOS releases an immediate first probe can
    // transiently return false and poison subsequent checks in the same process.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    auto warmup_mps_runtime = [&](const char* label) -> bool {
      try {
        const bool available = torch::mps::is_available();
        if (debug) {
          std::cerr << "[tyr:mps] " << label
                    << " torch::mps::is_available()=" << available << std::endl;
        }
        return available;
      } catch (const c10::Error& e) {
        if (debug) {
          std::cerr << "[tyr:mps] " << label
                    << " torch::mps::is_available() c10::Error: " << e.what() << std::endl;
        }
        return false;
      } catch (const std::exception& e) {
        if (debug) {
          std::cerr << "[tyr:mps] " << label
                    << " torch::mps::is_available() std::exception: " << e.what() << std::endl;
        }
        return false;
      } catch (...) {
        if (debug) {
          std::cerr << "[tyr:mps] " << label
                    << " torch::mps::is_available() failed with unknown exception" << std::endl;
        }
        return false;
      }
    };

    auto probe_mps_allocation = [&](const char* label) -> bool {
      try {
        auto probe = torch::ones({1}, torch::TensorOptions().device(torch::kMPS));
        const bool probe_ok = probe.device().is_mps();
        if (debug) {
          std::cerr << "[tyr:mps] " << label
                    << " probe allocation succeeded, device.is_mps()=" << probe_ok << std::endl;
        }
        return probe_ok;
      } catch (const c10::Error& e) {
        if (debug) {
          std::cerr << "[tyr:mps] " << label << " probe allocation c10::Error: " << e.what()
                    << std::endl;
        }
        return false;
      } catch (const std::exception& e) {
        if (debug) {
          std::cerr << "[tyr:mps] " << label << " probe allocation std::exception: " << e.what()
                    << std::endl;
        }
        return false;
      } catch (...) {
        if (debug) {
          std::cerr << "[tyr:mps] " << label
                    << " probe allocation failed with unknown exception" << std::endl;
        }
        return false;
      }
    };

    try {
      constexpr int kMaxAttempts = 8;
      constexpr auto kRetryDelay = std::chrono::milliseconds(100);
      constexpr auto kPostWarmupDelay = std::chrono::milliseconds(100);
      for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        const bool warmed = warmup_mps_runtime(attempt == 0 ? "initial-warmup" : "retry-warmup");
        if (warmed) {
          if (debug) {
            std::cerr << "[tyr:mps] sleeping after warmup for "
                      << kPostWarmupDelay.count() << "ms before allocation probe" << std::endl;
          }
          std::this_thread::sleep_for(kPostWarmupDelay);
        }
        const bool probe_ok = probe_mps_allocation(attempt == 0 ? "initial" : "retry");
        if (probe_ok) {
          const bool confirmed = warmup_mps_runtime("post-probe");
          if (debug) {
            std::cerr << "[tyr:mps] post-probe confirmed=" << confirmed << std::endl;
            std::cerr << "[tyr:mps] final availability=1 (probe)" << std::endl;
          }
          return true;
        }
        (void)warmed;
        if (attempt + 1 < kMaxAttempts) {
          std::this_thread::sleep_for(kRetryDelay);
        }
      }
      if (debug) {
        std::cerr << "[tyr:mps] final availability=0" << std::endl;
      }
      return false;
    } catch (const c10::Error& e) {
      if (debug) {
        std::cerr << "[tyr:mps] torch::mps::is_available() c10::Error: " << e.what()
                  << std::endl;
      }
      return probe_mps_allocation("exception-fallback");
    } catch (const std::exception& e) {
      if (debug) {
        std::cerr << "[tyr:mps] torch::mps::is_available() std::exception: " << e.what()
                  << std::endl;
      }
      return probe_mps_allocation("exception-fallback");
    }
  }
#else
  (void)debug_flag;
  return false;
#endif
}
