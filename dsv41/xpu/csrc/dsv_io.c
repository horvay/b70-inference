/* Parallel O_DIRECT reads of expert pieces (called through ctypes, which releases the GIL): no Python per piece.
   O_DIRECT into USM host memory fails, so each thread reads into its own aligned bounce buffer and copies out. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

typedef struct {
    int fd;
    const int64_t* offs;
    const int64_t* lens;
    const int64_t* dsts;
    int64_t n;
    int64_t chunk;            /* bytes per pread */
    volatile int64_t next;    /* work counter */
    volatile int64_t errors;
} job_t;

static void* worker(void* arg) {
    job_t* j = (job_t*)arg;
    void* bounce = NULL;
    if (posix_memalign(&bounce, 4096, (size_t)j->chunk) != 0) {
        __sync_fetch_and_add(&j->errors, 1);
        return NULL;
    }
    for (;;) {
        int64_t i = __sync_fetch_and_add(&j->next, 1);
        if (i >= j->n) break;
        int64_t off = j->offs[i], len = j->lens[i];
        char* dst = (char*)(intptr_t)j->dsts[i];
        int64_t got = 0;
        while (got < len) {
            int64_t want = len - got < j->chunk ? len - got : j->chunk;
            ssize_t r = pread(j->fd, bounce, (size_t)want, off + got);
            if (r <= 0) {
                __sync_fetch_and_add(&j->errors, 1);
                break;
            }
            memcpy(dst + got, bounce, (size_t)r);
            got += r;
        }
    }
    free(bounce);
    return NULL;
}

/* Read n pieces: piece i = bytes [offs[i], offs[i] + lens[i]) of fd into the host address dsts[i]. offs, lens and the
   chunk must be multiples of 4096 (O_DIRECT). Returns the number of failed pieces (0 = all read). */
int64_t read_pieces(int fd, const int64_t* offs, const int64_t* lens, const int64_t* dsts, int64_t n, int threads,
                    int64_t chunk) {
    if (n <= 0) return 0;
    job_t j = {fd, offs, lens, dsts, n, chunk, 0, 0};
    if (threads > n) threads = (int)n;
    if (threads > 64) threads = 64;
    pthread_t th[64];
    int started = 0;
    for (int t = 0; t < threads; t++)
        if (pthread_create(&th[t], NULL, worker, &j) == 0) started++;
    if (started == 0) worker(&j);
    for (int t = 0; t < started; t++) pthread_join(th[t], NULL);
    return j.errors;
}
