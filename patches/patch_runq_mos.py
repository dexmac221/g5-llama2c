#!/usr/bin/env python3
# 21 nov 2026 — da runq.c (llama2.c int8, 350e04f) a runq_mos.c per MorphOS sul G5:
#  - niente mmap: file letto in un buffer allineato a 16 byte (le righe int8 servono allineate per vec_ld)
#  - big-endian: si girano SOLO intestazione, norme fp32 e scale fp32; i pesi int8 restano come sono
#  - gettimeofday al posto di clock_gettime
#  - matmul AltiVec: vec_msum (vmsummbm, int8 con segno x uint8 -> int32, 16 prodotti per istruzione).
#    vmsummbm vuole un operando SENZA segno: x+128 (xor 0x80) e poi si toglie 128*somma(w),
#    con la somma dei pesi fatta al volo da vec_sum4s (stesso trucco di VNNI vpdpbusd, che è u8 x s8)
#  - -DFORZA_SWAP: gira i byte anche su x86 (prova con file big-endian); -DVERIFICA: confronta
#    AltiVec e scalare sulle prime chiamate e stampa lo scarto
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

def sost(vecchio, nuovo):
    global s
    n = s.count(vecchio)
    if n != 1:
        sys.exit(f"trovato {n} volte: {vecchio[:70]!r}")
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
#if defined(FORZA_SWAP) || defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    #define GIRA_BYTE 1
#endif
static void gira32(void* p, size_t n) {   // n parole da 32 bit, little-endian -> ordine della macchina
#ifdef GIRA_BYTE
    uint32_t* w = (uint32_t*)p;
    for (size_t i = 0; i < n; i++) {
        uint32_t x = w[i];
        w[i] = (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
    }
#else
    (void)p; (void)n;
#endif
}
static char* dati_grezzi = NULL;''')

# scale girate mentre si mappano i tensori (ogni tensore passa di qui una volta sola)
sost('''        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;''', '''        /* map scale factors */
        res[i].s = (float*)p;
        gira32(res[i].s, size_each / GS);
        p = (float*)p + size_each / GS;''')
sost('''    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;''', '''    float* fptr = (float*) ptr; // cast our pointer to float*
    gira32(fptr, (2 * p->n_layers + 1) * p->dim);   // le tre norme fp32, di fila
    w->rms_att_weight = fptr;''')

sost('''    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }''',
     '''    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    gira32(&magic_number, 1);''')
sost('''    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }''',
     '''    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    gira32(&version, 1);''')
sost('''    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // read in flags''', '''    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    gira32(config, sizeof(Config) / 4);
    // read in flags''')
sost('''    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }''',
     '''    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    gira32(&group_size, 1);''')
sost('''    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\\n"); exit(EXIT_FAILURE); }''', '''#ifdef SENZA_MMAP
    *fd = -1;
    dati_grezzi = (char*)malloc(*file_size + 32);
    if (!dati_grezzi) { fprintf(stderr, "malloc di %ld byte fallita\\n", (long)*file_size); exit(EXIT_FAILURE); }
    *data = (float*)(((uintptr_t)dati_grezzi + 15) & ~(uintptr_t)15);
    fseek(file, 0, SEEK_SET);
    if (fread(*data, 1, *file_size, file) != (size_t)*file_size) { fprintf(stderr, "lettura del modello fallita\\n"); exit(EXIT_FAILURE); }
    fclose(file);
#else
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\\n"); exit(EXIT_FAILURE); }
#endif''')
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
sost('''        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\\n"); exit(EXIT_FAILURE); }''',
     '''        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\\n"); exit(EXIT_FAILURE); }
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

sost('''void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
''', '''#ifdef __ALTIVEC__
#include <altivec.h>
static void matmul_scalare(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);
#ifdef PROFILO
static void matmul_vero(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);
void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) { unsigned long long t0 = tb(); matmul_vero(xout, x, w, n, d); tb_mm += tb() - t0; }
#define matmul matmul_vero
#endif
static unsigned char* xu = NULL; static char* xu_grezzo = NULL; static int xu_cap = 0;
void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    static int detto = 0;
    if ((GS & 15) || (n % GS) || ((uintptr_t)w->q & 15)) {
        if (!(detto & 2)) { fprintf(stderr, "[matmul int8 scalare: GS=%d n=%d q=%p]\\n", GS, n, (void*)w->q); detto |= 2; }
        matmul_scalare(xout, x, w, n, d); return;
    }
    if (!(detto & 1)) { fprintf(stderr, "[matmul int8 AltiVec vec_msum, GS=%d]\\n", GS); detto |= 1; }
    if (n > xu_cap) {
        free(xu_grezzo); xu_grezzo = (char*)malloc(n + 16);
        xu = (unsigned char*)(((uintptr_t)xu_grezzo + 15) & ~(uintptr_t)15); xu_cap = n;
    }
    for (int j = 0; j < n; j++) xu[j] = (unsigned char)x->q[j] ^ 0x80;   // x + 128, senza segno
    const vector signed int zero = vec_splat_s32(0);
    const vector unsigned int sette = vec_splat_u32(7);
    const int ng = n / GS;
#ifdef QUATTRO
    // versione 2: 4 gruppi alla volta; somme orizzontali con merge (trasposta 4x4), int->float con
    // vec_ctf e scale moltiplicate in vettore. Restano scalari solo i gruppi avanzati (ng % 4).
    static float* xs = NULL; static char* xs_grezzo = NULL; static int xs_cap = 0;
    if (ng + 4 > xs_cap) {
        free(xs_grezzo); xs_grezzo = (char*)malloc((ng + 4) * sizeof(float) + 16);
        xs = (float*)(((uintptr_t)xs_grezzo + 15) & ~(uintptr_t)15); xs_cap = ng + 4;
    }
    memcpy(xs, x->s, ng * sizeof(float));
    const vector float zf = (vector float){0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < d; i++) {
        const signed char* r = (const signed char*)w->q + (size_t)i * n;
        const float* ws = w->s + (size_t)i * ng;
        vector float vv = zf;
        int g = 0;
        for (; g + 4 <= ng; g += 4) {
#ifdef PREFETCH
            for (int off = 0; off < 4 * GS; off += 128) __builtin_prefetch(r + g * GS + off + PREFETCH, 0, 0);   // dcbt
#endif
            vector signed int a[4];
            for (int u = 0; u < 4; u++) {
                const signed char* rg = r + (g + u) * GS;
                const unsigned char* xg = xu + (g + u) * GS;
                vector signed int acc = zero, sw = zero;
                for (int k = 0; k < GS; k += 16) {
                    vector signed char wv = vec_ld(k, rg);
                    acc = vec_msum(wv, vec_ld(k, xg), acc);
                    sw = vec_sum4s(wv, sw);
                }
                a[u] = vec_sub(acc, vec_sl(sw, sette));
            }
            vector signed int u0 = vec_add(vec_mergeh(a[0], a[2]), vec_mergel(a[0], a[2]));
            vector signed int u1 = vec_add(vec_mergeh(a[1], a[3]), vec_mergel(a[1], a[3]));
            vector signed int sum = vec_add(vec_mergeh(u0, u1), vec_mergel(u0, u1));   // [g, g+1, g+2, g+3]
            const float* pw = ws + g;   // scale della riga: possono essere disallineate
            vector float wsv = vec_perm(vec_ld(0, pw), vec_ld(15, pw), vec_lvsl(0, pw));
            vector float sc = vec_madd(wsv, vec_ld(0, xs + g), zf);
            vv = vec_madd(vec_ctf(sum, 0), sc, vv);
        }
        float t4[4] __attribute__((aligned(16)));
        vec_st(vv, 0, t4);
        float val = (t4[0] + t4[1]) + (t4[2] + t4[3]);
        for (; g < ng; g++) {
#else
    for (int i = 0; i < d; i++) {
        const signed char* r = (const signed char*)w->q + (size_t)i * n;
        const float* ws = w->s + (size_t)i * ng;
        float val = 0.0f;
        for (int g = 0; g < ng; g++) {
#endif
            const signed char* rg = r + g * GS;
            const unsigned char* xg = xu + g * GS;
            vector signed int acc = zero, sw = zero;
            for (int k = 0; k < GS; k += 16) {
                vector signed char wv = vec_ld(k, rg);
                acc = vec_msum(wv, vec_ld(k, xg), acc);   // somma di w*(x+128), 4 alla volta per parola
                sw = vec_sum4s(wv, sw);                   // somma di w
            }
            acc = vec_sums(vec_sub(acc, vec_sl(sw, sette)), zero);   // - 128*somma(w), poi tutto in [3]
            int32_t t[4] __attribute__((aligned(16)));
            vec_st(acc, 0, t);
            val += ((float)t[3]) * ws[g] * x->s[g];
        }
        xout[i] = val;
    }
#ifdef VERIFICA
    static int volte = 0;
    if (volte < 3) {
        volte++;
        float* ref = (float*)malloc(d * sizeof(float));
        matmul_scalare(ref, x, w, n, d);
        float peggio = 0.0f;
        for (int i = 0; i < d; i++) { float e = fabsf(ref[i] - xout[i]); if (e > peggio) peggio = e; }
        fprintf(stderr, "[verifica %d: n=%d d=%d scarto max AltiVec/scalare = %g]\\n", volte, n, d, peggio);
        free(ref);
    }
#endif
}
static void matmul_scalare(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
#else
void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
#endif
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
''')

# --- PROFILO: tempi con il timebase del PowerPC (mftb), solo con -DPROFILO
sost('''void quantize(QuantizedTensor *qx, float* x, int n) {''', '''#ifdef PROFILO
static unsigned long long tb_mm = 0, tb_q = 0, tb_fw = 0, tb_att = 0, tb_ffn = 0;
static inline unsigned long long tb(void) { return __builtin_ppc_get_timebase(); }
static void quantize_vera(QuantizedTensor *qx, float* x, int n);
void quantize(QuantizedTensor *qx, float* x, int n) { unsigned long long t0 = tb(); quantize_vera(qx, x, n); tb_q += tb() - t0; }
static void quantize_vera(QuantizedTensor *qx, float* x, int n) {
#else
void quantize(QuantizedTensor *qx, float* x, int n) {
#endif''')
sost('''float* forward(Transformer* transformer, int token, int pos) {''', '''#ifdef PROFILO
#undef matmul
static float* forward_vero(Transformer* transformer, int token, int pos);
float* forward(Transformer* transformer, int token, int pos) { unsigned long long t0 = tb(); float* r = forward_vero(transformer, token, pos); tb_fw += tb() - t0; return r; }
static float* forward_vero(Transformer* transformer, int token, int pos) {
#else
float* forward(Transformer* transformer, int token, int pos) {
#endif''')
sost('''        fprintf(stderr, "achieved tok/s: %f\\n", (pos-1) / (double)(end-start)*1000);''', '''        fprintf(stderr, "achieved tok/s: %f\\n", (pos-1) / (double)(end-start)*1000);
#ifdef PROFILO
        fprintf(stderr, "[profilo su tutto il forward: matmul %.1f%%, quantize %.1f%%, RoPE+attenzione %.1f%%, SwiGLU %.1f%%, resto %.1f%%]\\n",
                100.0 * tb_mm / tb_fw, 100.0 * tb_q / tb_fw, 100.0 * tb_att / tb_fw, 100.0 * tb_ffn / tb_fw,
                100.0 * (tb_fw - tb_mm - tb_q - tb_att - tb_ffn) / tb_fw);
#endif''')

# --- VELOCE: parti fuori dalle matmul (21 nov sera, dopo il profilo: matmul 79%, quantize 7%, resto 14%)
sost('''void malloc_run_state(RunState* s, Config* p) {''', '''#ifdef VELOCE
// buffer di stato allineati a 16 byte (per vec_ld nell'attenzione); il puntatore vero sta subito prima
static void* calloc16(size_t n, size_t sz) {
    char* g = (char*)calloc(n * sz + 16 + sizeof(void*), 1);
    if (!g) return NULL;
    char* a = (char*)(((uintptr_t)(g + sizeof(void*)) + 15) & ~(uintptr_t)15);
    ((void**)a)[-1] = g;
    return a;
}
static void free16(void* a) { if (a) free(((void**)a)[-1]); }
#define calloc calloc16
#endif
void malloc_run_state(RunState* s, Config* p) {''')
sost('''void free_run_state(RunState* s) {''', '''#ifdef VELOCE
#undef calloc
#define free free16
#endif
void free_run_state(RunState* s) {''')
sost('''    free(s->value_cache);
}''', '''    free(s->value_cache);
}
#ifdef VELOCE
#undef free
#endif''')

# quantize: round() di libnix è lento; stesso risultato con +-0,5 in double e troncamento (esatto per |q| <= 127)
sost('''            int8_t quantized = (int8_t) round(quant_value); // round and clamp''', '''#ifdef VELOCE
            double dq = (double)quant_value;
            int8_t quantized = (int8_t)(int)(dq + (dq >= 0.0 ? 0.5 : -0.5));   // = round(), arrotonda lontano da zero
#else
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
#endif''')

# RoPE: cos/sin dipendono solo da pos e da i % head_size -> una volta per token invece che per ogni strato
sost('''        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);''', '''#ifdef PROFILO
        unsigned long long t_att0 = tb();
#endif
#ifdef VELOCE
        static float* rope_c = NULL; static float* rope_s = NULL; static int rope_pos = -1, rope_hs = 0;
        if (rope_hs != head_size) {
            free(rope_c); free(rope_s);
            rope_c = (float*)malloc(head_size / 2 * sizeof(float)); rope_s = (float*)malloc(head_size / 2 * sizeof(float));
            rope_hs = head_size; rope_pos = -1;
        }
        if (rope_pos != pos) {
            for (int j = 0; j < head_size; j += 2) {
                float freq = 1.0f / powf(10000.0f, j / (float)head_size);
                float val = pos * freq;
                rope_c[j / 2] = cosf(val);
                rope_s[j / 2] = sinf(val);
            }
            rope_pos = pos;
        }
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float fcr = rope_c[head_dim / 2];
            float fci = rope_s[head_dim / 2];
#else
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
#endif''')

# attenzione: prodotti q.k e somma pesata dei v con AltiVec (head_size multiplo di 16, buffer allineati)
sost('''                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }''', '''#if defined(VELOCE) && defined(__ALTIVEC__)
                float score = 0.0f;
                if ((head_size & 15) == 0 && ((((uintptr_t)q) | ((uintptr_t)k)) & 15) == 0) {
                    vector float z = (vector float){0.0f, 0.0f, 0.0f, 0.0f}, a0 = z, a1 = z, a2 = z, a3 = z;
                    for (int i = 0; i < head_size; i += 16) {
                        a0 = vec_madd(vec_ld(0, q + i),  vec_ld(0, k + i),  a0);
                        a1 = vec_madd(vec_ld(16, q + i), vec_ld(16, k + i), a1);
                        a2 = vec_madd(vec_ld(32, q + i), vec_ld(32, k + i), a2);
                        a3 = vec_madd(vec_ld(48, q + i), vec_ld(48, k + i), a3);
                    }
                    float tt[4] __attribute__((aligned(16)));
                    vec_st(vec_add(vec_add(a0, a1), vec_add(a2, a3)), 0, tt);
                    score = (tt[0] + tt[1]) + (tt[2] + tt[3]);
                } else {
                    for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                }
#else
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
#endif''')
sost('''                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }''', '''#if defined(VELOCE) && defined(__ALTIVEC__)
                if ((head_size & 3) == 0 && ((((uintptr_t)xb) | ((uintptr_t)v)) & 15) == 0) {
                    vector float av = (vector float){a, a, a, a};
                    for (int i = 0; i < head_size; i += 4)
                        vec_st(vec_madd(av, vec_ld(0, v + i), vec_ld(0, xb + i)), 0, xb + i);
                } else {
                    for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
                }
#else
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
#endif''')
sost('''        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);''', '''#ifdef PROFILO
        tb_att += tb() - t_att0;
#endif
        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);''')
sost('''        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {''', '''        // SwiGLU non-linearity
#ifdef PROFILO
        unsigned long long t_ffn0 = tb();
#endif
        for (int i = 0; i < hidden_dim; i++) {''')
sost('''        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);''', '''#ifdef PROFILO
        tb_ffn += tb() - t_ffn0;
#endif
        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);''')

# --- V3: pesi resi SENZA segno al caricamento (w ^ 0x80 = w + 128), attivazioni con segno:
#     vmsummbm(x con segno, w+128 senza segno) = x.w + 128*somma(x) -> la correzione dipende solo da x,
#     si calcola una volta per chiamata e vale per tutte le righe: niente vec_sum4s nel ciclo interno.
#     GS fissato a tempo di compilazione (32 o 64) perché il compilatore srotoli il ciclo.
sost('''    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}''', '''    memory_map_weights(weights, config, weights_ptr, shared_classifier);
#ifdef V3
    {
        Config* p = config; TransformerWeights* w = weights; int hs = p->dim / p->n_heads;
        gira_segno(w->q_tokens, 1, p->vocab_size * p->dim);   // la tabella fp32 è già stata ricavata prima
        gira_segno(w->wq, p->n_layers, p->dim * (p->n_heads * hs));
        gira_segno(w->wk, p->n_layers, p->dim * (p->n_kv_heads * hs));
        gira_segno(w->wv, p->n_layers, p->dim * (p->n_kv_heads * hs));
        gira_segno(w->wo, p->n_layers, (p->n_heads * hs) * p->dim);
        gira_segno(w->w1, p->n_layers, p->dim * p->hidden_dim);
        gira_segno(w->w2, p->n_layers, p->hidden_dim * p->dim);
        gira_segno(w->w3, p->n_layers, p->dim * p->hidden_dim);
        if (w->wcls != w->q_tokens) gira_segno(w->wcls, 1, p->dim * p->vocab_size);
        fprintf(stderr, "[V3: pesi int8 resi senza segno (+128) al caricamento]\\n");
    }
#endif
}''')
sost('''void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {''', '''#ifdef V3
static void gira_segno(QuantizedTensor* t, int n, int size_each) {
    for (int i = 0; i < n; i++) {
        uint32_t* q = (uint32_t*)t[i].q;   // le righe sono allineate: 4 byte alla volta
        for (int j = 0; j < size_each / 4; j++) q[j] ^= 0x80808080u;
    }
}
#endif
void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {''')
# lo scalare deve sapere che i pesi sono girati (serve per VERIFICA e per il caso non allineato, e su x86)
sost('''                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);''', '''#ifdef V3
                ival += ((int32_t) x->q[j + k]) * ((int32_t)(int8_t)(w->q[in + j + k] ^ 0x80));
#else
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
#endif''')
# il kernel V3
sost('''    if (!(detto & 1)) { fprintf(stderr, "[matmul int8 AltiVec vec_msum, GS=%d]\\n", GS); detto |= 1; }''', '''    if (!(detto & 1)) { fprintf(stderr, "[matmul int8 AltiVec vec_msum, GS=%d]\\n", GS); detto |= 1; }
#ifdef V3
    if (GS == 64) matmul_v3(xout, x, w, n, d, 64);
    else if (GS == 32) matmul_v3(xout, x, w, n, d, 32);
    else matmul_v3(xout, x, w, n, d, GS);
#ifdef VERIFICA
    static int volte3 = 0;
    if (volte3 < 3) {
        volte3++;
        float* ref = (float*)malloc(d * sizeof(float));
        matmul_scalare(ref, x, w, n, d);
        float peggio = 0.0f;
        for (int i = 0; i < d; i++) { float e = fabsf(ref[i] - xout[i]); if (e > peggio) peggio = e; }
        fprintf(stderr, "[verifica V3 %d: n=%d d=%d scarto max = %g]\\n", volte3, n, d, peggio);
        free(ref);
    }
#endif
    return;
#endif''')
sost('''static unsigned char* xu = NULL; static char* xu_grezzo = NULL; static int xu_cap = 0;''', '''static unsigned char* xu = NULL; static char* xu_grezzo = NULL; static int xu_cap = 0;
#ifdef V3
static signed char* x3 = NULL; static char* x3_grezzo = NULL; static int x3_cap = 0;
static int32_t* corr3 = NULL; static float* xs3 = NULL; static char* c3_grezzo = NULL; static int c3_cap = 0;
static inline __attribute__((always_inline))
void matmul_v3(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d, const int gs) {
    const int ng = n / gs;
    if (n > x3_cap) {
        free(x3_grezzo); x3_grezzo = (char*)malloc(n + 16);
        x3 = (signed char*)(((uintptr_t)x3_grezzo + 15) & ~(uintptr_t)15); x3_cap = n;
    }
    if (ng + 4 > c3_cap) {
        free(c3_grezzo); c3_grezzo = (char*)malloc(2 * (ng + 4) * 4 + 32);
        corr3 = (int32_t*)(((uintptr_t)c3_grezzo + 15) & ~(uintptr_t)15);
        xs3 = (float*)(corr3 + ((ng + 4 + 3) & ~3)); c3_cap = ng + 4;
    }
    memcpy(x3, x->q, n);
    for (int g = 0; g < ng; g++) {
        int32_t sx = 0;
        for (int k = 0; k < gs; k++) sx += x->q[g * gs + k];
        corr3[g] = 128 * sx;          // vmsummbm(x, w+128) - 128*somma(x) = x.w
        xs3[g] = x->s[g];
    }
    const vector signed int zero = vec_splat_s32(0);
    const vector float zf = (vector float){0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < d; i++) {
        const unsigned char* r = (const unsigned char*)w->q + (size_t)i * n;
        const float* ws = w->s + (size_t)i * ng;
        vector float vv = zf;
        int g = 0;
        for (; g + 4 <= ng; g += 4) {
            vector signed int a[4];
            for (int u = 0; u < 4; u++) {
                const unsigned char* rg = r + (g + u) * gs;
                const signed char* xg = x3 + (g + u) * gs;
                vector signed int acc = zero;
                for (int k = 0; k < gs; k += 16) acc = vec_msum(vec_ld(k, xg), vec_ld(k, rg), acc);
                a[u] = acc;
            }
            vector signed int u0 = vec_add(vec_mergeh(a[0], a[2]), vec_mergel(a[0], a[2]));
            vector signed int u1 = vec_add(vec_mergeh(a[1], a[3]), vec_mergel(a[1], a[3]));
            vector signed int sum = vec_sub(vec_add(vec_mergeh(u0, u1), vec_mergel(u0, u1)), vec_ld(0, corr3 + g));
            const float* pw = ws + g;
            vector float wsv = vec_perm(vec_ld(0, pw), vec_ld(15, pw), vec_lvsl(0, pw));
            vector float sc = vec_madd(wsv, vec_ld(0, xs3 + g), zf);
            vv = vec_madd(vec_ctf(sum, 0), sc, vv);
        }
        float t4[4] __attribute__((aligned(16)));
        vec_st(vv, 0, t4);
        float val = (t4[0] + t4[1]) + (t4[2] + t4[3]);
        for (; g < ng; g++) {
            vector signed int acc = zero;
            for (int k = 0; k < gs; k += 16) acc = vec_msum(vec_ld(k, x3 + g * gs), vec_ld(k, r + g * gs), acc);
            acc = vec_sums(acc, zero);
            int32_t t[4] __attribute__((aligned(16)));
            vec_st(acc, 0, t);
            val += ((float)(t[3] - corr3[g])) * ws[g] * xs3[g];
        }
        xout[i] = val;
    }
}
#endif''')

open(dst, 'w').write(s)
print("scritto", dst)
