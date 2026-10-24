// Emscripten has no CPU-affinity APIs. verilated_threads.cpp (always linked, even for
// single-threaded models, since verilated.cpp references VlThreadPool) calls these to
// pin worker threads. In WASM we run with 0 worker threads, so "unsupported" is correct.
#include <cerrno>
#include <pthread.h>
#include <sched.h>

extern "C" {

int pthread_getaffinity_np(pthread_t, size_t, cpu_set_t*) { return ENOSYS; }
int pthread_setaffinity_np(pthread_t, size_t, const cpu_set_t*) { return ENOSYS; }
int sched_getcpu(void) { return 0; }

}
