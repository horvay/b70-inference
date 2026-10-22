/* Host-side waits on pinned flags written by the GPU (called through ctypes, which releases the GIL). */
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <immintrin.h>

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Wait until *p >= v. Returns *p, -1 on timeout, -2 if *abort_flag became nonzero. Spins briefly, then yields. */
int64_t wait_ge(volatile int64_t* p, int64_t v, int64_t timeout_ns, volatile int64_t* abort_flag) {
    int64_t t0 = now_ns();
    int64_t spins = 0;
    for (;;) {
        int64_t x = *p;
        if (x >= v) return x;
        if (abort_flag && *abort_flag) return -2;
        if (++spins < 4000) {
            _mm_pause();
        } else {
            sched_yield();
            if ((spins & 255) == 0 && now_ns() - t0 > timeout_ns) return -1;
        }
    }
}

/* Pager fast path (called without the GIL): wait until ready >= step. If the layer's staging copies have landed
   (stage_flag >= step), point the staged pieces at their VRAM copies (ptr[keys[i]] = addrs[i]). Then check that all
   three pieces of each of the n picked experts are resident (vslot >= 0 or rslot >= 0 at key = base_key + 3 * expert
   + piece). If so, publish done = step and return 1; return 0 if some piece is missing (the caller reads it from
   NVMe and publishes done itself), -1 on timeout, -2 on abort. Flips applied: *flipped = 1. */
int64_t fast_release(volatile int64_t* ready, int64_t step, int64_t timeout_ns, volatile int64_t* abort_flag,
                     const volatile int64_t* picks, int64_t n, const int64_t* vslot, const int64_t* rslot,
                     int64_t base_key, volatile int64_t* done, const volatile int64_t* stage_flag, int64_t n_stage,
                     const int64_t* stage_keys, const int64_t* stage_addrs, int64_t* ptr, int64_t* flipped) {
    int64_t r = wait_ge(ready, step, timeout_ns, abort_flag);
    if (r < 0) return r;
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    *flipped = 0;
    if (stage_flag && n_stage > 0 && *stage_flag >= step) {
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        for (int64_t i = 0; i < n_stage; ++i) ptr[stage_keys[i]] = stage_addrs[i];
        *flipped = 1;
    }
    for (int64_t i = 0; i < n; ++i) {
        int64_t k = base_key + 3 * picks[i];
        for (int p = 0; p < 3; ++p)
            if (vslot[k + p] < 0 && rslot[k + p] < 0) return 0;
    }
    __atomic_store_n(done, step, __ATOMIC_RELEASE);
    return 1;
}
