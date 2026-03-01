/*
 * tyr_media_stub.cpp
 *
 * Cross-platform stubs for Apple-only media preprocessing FFI entrypoints.
 * Linux/other non-Apple builds must still provide these symbols because Lean
 * references them unconditionally from extern declarations.
 */

#include <lean/lean.h>

#ifndef __APPLE__
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
    uint64_t /*frame_stride*/,
    lean_object* /*w*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
      lean_mk_string("Apple media path requires macOS build")));
}

} // extern "C"
#endif
