#!/usr/bin/env python3
# 21 nov 2026 — da run.c (llama2.c di karpathy, 350e04f) a run_mos.c per MorphOS sul G5:
#  - niente mmap (MorphOS non ce l'ha): il file del modello si legge tutto in memoria con fread
#  - big-endian: i file .bin sono little-endian (tutto a 32 bit) -> si girano i byte al caricamento
#  - tempo con gettimeofday (clock_gettime non c'è in libnix)
#  - -DFORZA_SWAP: gira i byte anche su x86, per provare la strada big-endian su un PC x86
#    con file pre-girati (stesso risultato = la conversione è giusta)
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

def sost(vecchio, nuovo):
    global s
    n = s.count(vecchio)
    if n != 1:
        sys.exit(f"trovato {n} volte: {vecchio[:60]!r}")
    s = s.replace(vecchio, nuovo)

sost('''#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif''', '''#if defined _WIN32
    #include "win.h"
#elif defined(__MORPHOS__) || defined(SENZA_MMAP)
    #include <unistd.h>
    #include <sys/time.h>
    #include <sys/types.h>
    #define SENZA_MMAP 1
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include <stdint.h>

// i file .bin di llama2.c sono little-endian: sul PowerPC (big-endian) si girano i byte
#if defined(FORZA_SWAP) || defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    #define GIRA_BYTE 1
#endif
static char* dati_grezzi = NULL;   // buffer del modello (SENZA_MMAP), allineato a mano
static void gira32(void* p, size_t n) {   // n parole da 32 bit
#ifdef GIRA_BYTE
    uint32_t* w = (uint32_t*)p;
    for (size_t i = 0; i < n; i++) {
        uint32_t x = w[i];
        w[i] = (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
    }
#else
    (void)p; (void)n;
#endif
}''')

sost('''    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size''', '''    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    gira32(config, sizeof(Config) / 4);
    // negative vocab size''')

sost('''    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\\n"); exit(EXIT_FAILURE); }
''', '''#ifdef SENZA_MMAP
    // tutto il file in memoria (MorphOS non ha mmap)
    *fd = -1;
    dati_grezzi = (char*)malloc(*file_size + 32);
    if (!dati_grezzi) { fprintf(stderr, "malloc di %ld byte fallita\\n", (long)*file_size); exit(EXIT_FAILURE); }
#ifdef NON_ALLINEARE
    *data = (float*)dati_grezzi;   // solo per la prova: pesi dove capita (Config = 28 byte)
#else
    { char* p = dati_grezzi; while (((uintptr_t)(p + sizeof(Config))) & 15) p += 4; *data = (float*)p; }
#endif
    fprintf(stderr, "[pesi a %p, resto modulo 16 = %d, modulo 128 = %d]\\n", (void*)((char*)*data + sizeof(Config)), (int)(((uintptr_t)*data + sizeof(Config)) & 15), (int)(((uintptr_t)*data + sizeof(Config)) & 127));
    fseek(file, 0, SEEK_SET);
    if (fread(*data, 1, *file_size, file) != (size_t)*file_size) { fprintf(stderr, "lettura del modello fallita\\n"); exit(EXIT_FAILURE); }
    fclose(file);
    gira32(*data, *file_size / 4);   // nel file è tutto a 32 bit (int32 + float32)
#else
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\\n"); exit(EXIT_FAILURE); }
#endif
''')

sost('''    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }''', '''#ifdef SENZA_MMAP
    free(dati_grezzi);
#else
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
#endif''')

sost('''    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\\n"); exit(EXIT_FAILURE); }''',
     '''    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\\n"); exit(EXIT_FAILURE); }
    gira32(&t->max_token_length, 1);''')
sost('''        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\\n"); exit(EXIT_FAILURE); }''',
     '''        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\\n"); exit(EXIT_FAILURE); }
        gira32(t->vocab_scores + i, 1); gira32(&len, 1);''')

sost('''    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;''', '''#ifdef SENZA_MMAP
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
#else
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
#endif''')

sost('''void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;''', '''#ifdef __ALTIVEC__
#include <altivec.h>
// AltiVec (G4/G5): 4 float per istruzione, vec_madd = moltiplica-somma fusa.
// Righe di W allineate a 16 byte (buffer allineato + n multiplo di 4); x copiato in un buffer allineato.
static float* x_allineato = NULL; static int x_cap = 0; static char* x_grezzo = NULL;
static void matmul_scalare(float* xout, float* x, float* w, int n, int d);
void matmul(float* xout, float* x, float* w, int n, int d) {
    static int detto = 0;
    if ((n & 3) || ((uintptr_t)w & 15)) {
        if (!(detto & 2)) { fprintf(stderr, "[matmul scalare: n=%d w=%p]\\n", n, (void*)w); detto |= 2; }
        matmul_scalare(xout, x, w, n, d); return;
    }
    if (!(detto & 1)) { fprintf(stderr, "[matmul AltiVec attivo]\\n"); detto |= 1; }
    if (n > x_cap) {
        free(x_grezzo); x_grezzo = (char*)malloc(n * sizeof(float) + 16);
        x_allineato = (float*)(((uintptr_t)x_grezzo + 15) & ~(uintptr_t)15); x_cap = n;
    }
    memcpy(x_allineato, x, n * sizeof(float));
    const vector float zero = (vector float){0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < d; i++) {
        const float* r = w + (size_t)i * n;
        vector float a0 = zero, a1 = zero, a2 = zero, a3 = zero;
        int j = 0;
        for (; j + 16 <= n; j += 16) {
            a0 = vec_madd(vec_ld(0, r + j),      vec_ld(0, x_allineato + j),      a0);
            a1 = vec_madd(vec_ld(16, r + j),     vec_ld(16, x_allineato + j),     a1);
            a2 = vec_madd(vec_ld(32, r + j),     vec_ld(32, x_allineato + j),     a2);
            a3 = vec_madd(vec_ld(48, r + j),     vec_ld(48, x_allineato + j),     a3);
        }
        for (; j < n; j += 4) a0 = vec_madd(vec_ld(0, r + j), vec_ld(0, x_allineato + j), a0);
        a0 = vec_add(vec_add(a0, a1), vec_add(a2, a3));
        float t[4] __attribute__((aligned(16)));
        vec_st(a0, 0, t);
        xout[i] = (t[0] + t[1]) + (t[2] + t[3]);
    }
}
static void matmul_scalare(float* xout, float* x, float* w, int n, int d) {
#else
void matmul(float* xout, float* x, float* w, int n, int d) {
#endif
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;''')

open(dst, 'w').write(s)
print("scritto", dst)
