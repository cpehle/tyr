/*
 * tyr_execution.cpp - Sandboxed Code Execution FFI
 *
 * Provides sandboxed execution of Python/shell code with:
 * - Timeout protection via alarm/SIGALRM
 * - Memory limits via setrlimit
 * - Process isolation via fork
 * - Stdout/stderr capture via pipes
 *
 * Based on nanochat's execution.py approach.
 */

#include <lean/lean.h>

#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <fcntl.h>
#include <poll.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

extern "C" {

// ============================================================================
// Helper Functions
// ============================================================================

static lean_object* mk_lean_string(const std::string& s) {
    return lean_mk_string(s.c_str());
}

static lean_object* mk_io_error(const std::string& msg) {
    lean_object* err = lean_mk_io_user_error(lean_mk_string(msg.c_str()));
    return lean_io_result_mk_error(err);
}

static lean_object* mk_io_ok(lean_object* value) {
    return lean_io_result_mk_ok(value);
}

// ExitStatus enum values (must match Lean definition order)
enum ExitStatus {
    EXIT_SUCCESS_STATUS = 0,
    EXIT_FAILURE_STATUS = 1,
    EXIT_TIMEOUT = 2,
    EXIT_MEMORY_LIMIT = 3,
    EXIT_SIGNALED = 4,
    EXIT_ERROR = 5
};

// Create ExecResult structure
// Structure: { status : ExitStatus, exitCode : Int32, stdout : String,
//              stderr : String, execTimeMs : UInt64 }
static lean_object* mk_exec_result(ExitStatus status, int32_t exit_code,
                                   const std::string& stdout_str,
                                   const std::string& stderr_str,
                                   uint64_t exec_time_ms) {
    lean_object* obj = lean_alloc_ctor(0, 3, 8 + 4);  // 3 object fields, 12 scalar bytes

    // Object fields
    lean_ctor_set(obj, 0, lean_box(status));              // status (enum)
    lean_ctor_set(obj, 1, mk_lean_string(stdout_str));    // stdout
    lean_ctor_set(obj, 2, mk_lean_string(stderr_str));    // stderr

    // Scalar fields (Int32 + UInt64)
    lean_ctor_set_uint32(obj, sizeof(void*) * 3, static_cast<uint32_t>(exit_code));
    lean_ctor_set_uint64(obj, sizeof(void*) * 3 + 4, exec_time_ms);

    return obj;
}

// Read from file descriptor into string (non-blocking)
static std::string read_fd(int fd, size_t max_size = 1024 * 1024) {
    std::string result;
    char buffer[4096];

    while (result.size() < max_size) {
        ssize_t n = read(fd, buffer, sizeof(buffer));
        if (n <= 0) break;
        result.append(buffer, n);
    }

    return result;
}

// Create temporary file with content
static std::string create_temp_file(const std::string& content, const std::string& suffix) {
    std::string template_path = fs::temp_directory_path().string() + "/tyr_exec_XXXXXX" + suffix;
    std::vector<char> path_buf(template_path.begin(), template_path.end());
    path_buf.push_back('\0');

    // Use mkstemps to create file with suffix
    int fd = mkstemps(path_buf.data(), static_cast<int>(suffix.length()));
    if (fd < 0) {
        return "";
    }

    // Write content
    write(fd, content.data(), content.size());
    close(fd);

    return std::string(path_buf.data());
}

// Set resource limits for child process
static void set_resource_limits(uint64_t memory_limit) {
    // Memory limit (virtual memory)
    struct rlimit mem_limit;
    mem_limit.rlim_cur = memory_limit;
    mem_limit.rlim_max = memory_limit;
    setrlimit(RLIMIT_AS, &mem_limit);

    // CPU time limit (backup for timeout)
    struct rlimit cpu_limit;
    cpu_limit.rlim_cur = 60;  // 60 seconds max
    cpu_limit.rlim_max = 60;
    setrlimit(RLIMIT_CPU, &cpu_limit);

    // Disable core dumps
    struct rlimit core_limit;
    core_limit.rlim_cur = 0;
    core_limit.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &core_limit);

    // Limit number of processes (prevent fork bombs)
    struct rlimit nproc_limit;
    nproc_limit.rlim_cur = 10;
    nproc_limit.rlim_max = 10;
    setrlimit(RLIMIT_NPROC, &nproc_limit);
}

// ============================================================================
// Main Execution Function
// ============================================================================

static lean_object* exec_sandboxed_impl(const std::string& command,
                                        const std::vector<std::string>& args,
                                        uint64_t timeout_ms,
                                        uint64_t memory_limit) {
    // Create pipes for stdout/stderr capture
    int stdout_pipe[2], stderr_pipe[2];
    if (pipe(stdout_pipe) < 0 || pipe(stderr_pipe) < 0) {
        return mk_io_error("Failed to create pipes");
    }

    auto start_time = std::chrono::steady_clock::now();

    pid_t pid = fork();

    if (pid < 0) {
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        return mk_io_error("Failed to fork process");
    }

    if (pid == 0) {
        // Child process

        // Set up stdout/stderr redirection
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        // Set resource limits
        set_resource_limits(memory_limit);

        // Build argument array
        std::vector<const char*> argv;
        argv.push_back(command.c_str());
        for (const auto& arg : args) {
            argv.push_back(arg.c_str());
        }
        argv.push_back(nullptr);

        // Execute
        execvp(command.c_str(), const_cast<char* const*>(argv.data()));

        // If exec fails
        _exit(127);
    }

    // Parent process
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    // Set non-blocking mode
    fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

    // Wait for child with timeout
    std::string stdout_str, stderr_str;
    int status = 0;
    bool timed_out = false;

    // Poll for output and child completion
    while (true) {
        // Check elapsed time
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        if (static_cast<uint64_t>(elapsed_ms) > timeout_ms) {
            // Timeout - kill child
            kill(pid, SIGKILL);
            timed_out = true;
            waitpid(pid, &status, 0);
            break;
        }

        // Check if child has exited
        int result = waitpid(pid, &status, WNOHANG);
        if (result > 0) {
            // Child exited
            break;
        } else if (result < 0) {
            // Error
            break;
        }

        // Read available output
        char buf[4096];
        ssize_t n;
        while ((n = read(stdout_pipe[0], buf, sizeof(buf))) > 0) {
            stdout_str.append(buf, n);
        }
        while ((n = read(stderr_pipe[0], buf, sizeof(buf))) > 0) {
            stderr_str.append(buf, n);
        }

        // Small sleep to avoid busy waiting
        usleep(1000);  // 1ms
    }

    // Read any remaining output
    char buf[4096];
    ssize_t n;
    while ((n = read(stdout_pipe[0], buf, sizeof(buf))) > 0) {
        stdout_str.append(buf, n);
    }
    while ((n = read(stderr_pipe[0], buf, sizeof(buf))) > 0) {
        stderr_str.append(buf, n);
    }

    close(stdout_pipe[0]);
    close(stderr_pipe[0]);

    // Calculate execution time
    auto end_time = std::chrono::steady_clock::now();
    auto exec_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Determine exit status
    ExitStatus exit_status;
    int32_t exit_code = 0;

    if (timed_out) {
        exit_status = EXIT_TIMEOUT;
        exit_code = -1;
    } else if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
        if (exit_code == 0) {
            exit_status = EXIT_SUCCESS_STATUS;
        } else {
            exit_status = EXIT_FAILURE_STATUS;
        }
    } else if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        exit_code = -sig;
        if (sig == SIGXCPU || sig == SIGKILL) {
            // Likely resource limit
            exit_status = EXIT_MEMORY_LIMIT;
        } else {
            exit_status = EXIT_SIGNALED;
        }
    } else {
        exit_status = EXIT_ERROR;
        exit_code = -1;
    }

    return mk_io_ok(mk_exec_result(exit_status, exit_code, stdout_str, stderr_str,
                                   static_cast<uint64_t>(exec_time_ms)));
}

// ============================================================================
// FFI Implementation: lean_exec_python_sandboxed
// ============================================================================

lean_object* lean_exec_python_sandboxed(b_lean_obj_arg code, b_lean_obj_arg config,
                                        lean_object* /* world */) {
    std::string code_str = lean_string_cstr(code);

    // Extract config fields
    // Config structure: { timeout : UInt64, memoryLimit : UInt64, workDir : String, envVars : Array }
    uint64_t timeout_ms = lean_ctor_get_uint64(config, sizeof(void*) * 2);
    uint64_t memory_limit = lean_ctor_get_uint64(config, sizeof(void*) * 2 + 8);

    // Create temp file with code
    std::string temp_file = create_temp_file(code_str, ".py");
    if (temp_file.empty()) {
        return mk_io_error("Failed to create temporary file");
    }

    // Execute Python
    auto result = exec_sandboxed_impl("python3", {temp_file}, timeout_ms, memory_limit);

    // Clean up temp file
    fs::remove(temp_file);

    return result;
}

// ============================================================================
// FFI Implementation: lean_exec_shell_sandboxed
// ============================================================================

lean_object* lean_exec_shell_sandboxed(b_lean_obj_arg command, b_lean_obj_arg config,
                                       lean_object* /* world */) {
    std::string cmd_str = lean_string_cstr(command);

    // Extract config fields
    uint64_t timeout_ms = lean_ctor_get_uint64(config, sizeof(void*) * 2);
    uint64_t memory_limit = lean_ctor_get_uint64(config, sizeof(void*) * 2 + 8);

    // Execute via shell
    return exec_sandboxed_impl("/bin/sh", {"-c", cmd_str}, timeout_ms, memory_limit);
}

// ============================================================================
// FFI Implementation: lean_exec_is_available
// ============================================================================

lean_object* lean_exec_is_available(lean_object* /* world */) {
    // Check if we can fork and have python3
    bool available = true;

#ifdef _WIN32
    // Windows not supported for sandboxed execution
    available = false;
#else
    // Check for python3
    if (system("which python3 > /dev/null 2>&1") != 0) {
        available = false;
    }
#endif

    return mk_io_ok(lean_box(available ? 1 : 0));
}

} // extern "C"
