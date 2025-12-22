// Wrapper to export Tyr initialization symbol with default visibility
// The Lean-generated initializer has hidden visibility, so we use dlsym to find it

#include <lean/lean.h>
#include <dlfcn.h>
#include <stdio.h>

typedef lean_object* (*init_fn_t)(uint8_t, lean_object*);

// This is the symbol the Lean plugin loader looks for
// It has default visibility so it will be exported from the dylib
extern "C" __attribute__((visibility("default")))
lean_object* initialize_Tyr(uint8_t builtin, lean_object* w) {
    // Find the hidden Lean-generated initializer
    // We renamed it to initialize_Tyr_impl during compilation
    static init_fn_t impl = nullptr;
    if (!impl) {
        // Try to find it via dlsym with RTLD_DEFAULT (searches all loaded images)
        impl = (init_fn_t)dlsym(RTLD_DEFAULT, "initialize_Tyr_impl");
        if (!impl) {
            fprintf(stderr, "Error: Could not find initialize_Tyr_impl\n");
            return lean_io_result_mk_error(lean_mk_io_user_error(
                lean_mk_string("Failed to find Tyr initializer implementation")));
        }
    }
    return impl(builtin, w);
}
