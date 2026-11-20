/* membw v2 — 内存带宽测量（修正串行依赖链）
 * v1 的 bug: s += a[i] 是 FP 加法链，每 8 字节等 4 周期延迟，测的是加法器不是内存
 * v2: 8 个独立整数累加器 + XOR，消除依赖链，-O3 展开
 * 编译: gcc -O3 -march=native -pthread -funroll-loops -o membw membw.c
 * 用法: ./membw read <GiB> <线程数> | ./membw copy <GiB> <线程数> | ./membw latency <GiB>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

typedef struct {
    int       id;
    uint64_t *src;
    uint64_t *dst;
    size_t    n;      /* uint64 个数 */
    double    bytes;
    double    dt;
} worker_t;

static void pin(int id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(id, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

/* 纯读：8 路独立 XOR，无依赖链，1 周期可发射一次 load */
static void *do_read(void *p) {
    worker_t *w = (worker_t *)p;
    pin(w->id);
    const uint64_t *s = w->src;
    size_t n = w->n;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    double t0 = now_s();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 ^= s[i];     a1 ^= s[i + 1]; a2 ^= s[i + 2]; a3 ^= s[i + 3];
        a4 ^= s[i + 4]; a5 ^= s[i + 5]; a6 ^= s[i + 6]; a7 ^= s[i + 7];
    }
    for (; i < n; i++) a0 ^= s[i];
    double t1 = now_s();
    w->dt = t1 - t0;
    w->bytes = (double)(n * sizeof(uint64_t));
    if ((a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7) == 0xDEADBEEFCAFEBABEULL)
        fprintf(stderr, ".");
    return NULL;
}

static void *do_copy(void *p) {
    worker_t *w = (worker_t *)p;
    pin(w->id);
    double t0 = now_s();
    memcpy(w->dst, w->src, w->n * sizeof(uint64_t));
    double t1 = now_s();
    w->dt = t1 - t0;
    w->bytes = (double)(w->n * sizeof(uint64_t)) * 2.0;  /* 读 + 写 */
    return NULL;
}

/* 指针追逐：测单次 DRAM 访问延迟（LLC 全失效） */
static void *do_latency(void *p) {
    worker_t *w = (worker_t *)p;
    pin(0);
    uint64_t *s = w->src;
    size_t n = w->n;
    uint64_t idx = 0;
    double t0 = now_s();
    for (size_t i = 0; i < n / 4; i++) idx = s[idx];
    double t1 = now_s();
    w->dt = t1 - t0;
    w->bytes = (double)(n / 4);
    if (idx == 0xFFFFFFFFFFFFFFFFULL) fprintf(stderr, ".");
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s <read|copy|latency> <GiB> [线程数]\n", argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    double gib = atof(argv[2]);
    int nth = (argc >= 4) ? atoi(argv[3]) : 4;
    if (nth < 1 || nth > 16) { fprintf(stderr, "线程数 1..16\n"); return 2; }

    size_t chunk = (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
    size_t total = chunk * nth;
    int is_copy = (strcmp(mode, "copy") == 0);
    int is_lat  = (strcmp(mode, "latency") == 0);

    size_t alloc = is_copy ? total * 2 : total;
    char *mem = mmap(NULL, alloc, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (mem == MAP_FAILED) { perror("mmap"); return 1; }
    memset(mem, 1, alloc);

    uint64_t *a = (uint64_t *)mem;
    uint64_t *b = is_copy ? (uint64_t *)(mem + total) : NULL;
    size_t per = chunk / sizeof(uint64_t);   /* 每线程的 uint64 个数 */

    /* latency 模式：把页链成一个随机环，破坏预取 */
    if (is_lat) {
        size_t npages = per / 8;              /* 每页 8 个 uint64 */
        size_t *ord = malloc(npages * sizeof(size_t));
        for (size_t i = 0; i < npages; i++) ord[i] = i;
        srand(12345);
        for (size_t i = npages - 1; i > 0; i--) {
            size_t j = (size_t)rand() % (i + 1);
            size_t t = ord[i]; ord[i] = ord[j]; ord[j] = t;
        }
        /* 每页第一个 uint64 存下一页的元素下标，形成一条覆盖所有页的单环 */
        for (size_t i = 0; i < npages; i++) {
            uint64_t *cur = a + ord[i] * 8;
            *cur = (uint64_t)(ord[(i + 1) % npages] * 8);
        }
        free(ord);
        uint64_t idx = 0;
        double t0 = now_s();
        for (size_t i = 0; i < npages; i++) idx = a[idx];
        double t1 = now_s();
        double dt = t1 - t0;
        if (idx == 0xFFFFFFFFFFFFFFFFULL) fprintf(stderr, ".");
        printf("latency  %4.1f GiB 工作集  %.1f ns/次  (%.1f M 随机访问/s)\n",
               gib, dt / npages * 1e9, npages / dt / 1e6);
        munmap(mem, alloc);
        return 0;
    }

    pthread_t th[16];
    worker_t w[16];
    double wall0 = now_s();
    for (int i = 0; i < nth; i++) {
        w[i].id = i;
        w[i].n = per;
        w[i].src = a + (size_t)i * per;
        w[i].dst = is_copy ? (b + (size_t)i * per) : NULL;
        pthread_create(&th[i], NULL, is_copy ? do_copy : do_read, &w[i]);
    }
    double sum_bytes = 0;
    for (int i = 0; i < nth; i++) { pthread_join(th[i], NULL); sum_bytes += w[i].bytes; }
    double wall = now_s() - wall0;

    printf("%-4s  %4.1f GiB/线程 × %d线程  %6.2f GB/s  (%.2f s, %.1f GB/s/线程)\n",
           mode, gib, nth, sum_bytes / wall / 1e9, wall, sum_bytes / wall / 1e9 / nth);

    munmap(mem, alloc);
    return 0;
}
