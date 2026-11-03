// graceful_exit.h
//
// Closes the Tenstorrent card on every exit path. A program that dies with the
// device open leaves kernels running and the card busy, and the next open can
// fail. run_guarded() catches exceptions (every TT_FATAL throws), so the stack
// unwinds and ~MeshDevice() closes the device. SIGINT and SIGTERM are blocked,
// so they never interrupt a device call: the run finishes, closes the device
// and exits 1. Threads started before main() still take them, so a handler
// records them there instead of killing the process. A hung device call still
// needs kill -9 and `tt-smi -r <card>`.

#pragma once

#include <csignal>
#include <exception>
#include <iostream>

namespace turboquant {

inline volatile std::sig_atomic_t g_stop_signal = 0;

inline int run_guarded(int (*body)(int, char**), int argc, char** argv) {
    auto record = [](int sig) { g_stop_signal = sig; };
    std::signal(SIGINT, record);
    std::signal(SIGTERM, record);
    sigset_t stop;
    sigemptyset(&stop);
    sigaddset(&stop, SIGINT);
    sigaddset(&stop, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &stop, nullptr);  // inherited by every thread tt-metal starts

    int rc = 1;
    try {
        rc = body(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[TurboQuant] error: " << e.what() << "\n";
    }
    pthread_sigmask(SIG_UNBLOCK, &stop, nullptr);  // records a signal that arrived meanwhile
    if (g_stop_signal) {
        std::cerr << "[TurboQuant] stopped by signal " << g_stop_signal << "\n";
        return 1;
    }
    return rc;
}

}  // namespace turboquant
