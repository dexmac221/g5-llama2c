#!/usr/bin/env python3
# 22 nov 2026 — da runq_leo.c a runq_leo2.c (revisione di Claude web):
#  -DFONDI   : q/k/v e w1/w3 in UNA regione parallela (97 -> 61 regioni per token), corpo identico a matmul_v3
#  -DPROFILO : sotto macOS cronometro mach_absolute_time() + millisecondi per token di ogni parte
import re, sys
s = open('src/runq_leo.c').read()
def sost(a, b):
    global s
    n = s.count(a)
    if n != 1: sys.exit(f"trovato {n} volte: {a[:70]!r}")
    s = s.replace(a, b)

# 1) cronometro su macOS
sost("static inline unsigned long long tb(void) { return __builtin_ppc_get_timebase(); }",
     "#ifdef __APPLE__\n#include <mach/mach_time.h>\nstatic inline unsigned long long tb(void) { return mach_absolute_time(); }\n#else\n"
     "static inline unsigned long long tb(void) { return __builtin_ppc_get_timebase(); }\n#endif\nstatic long n_fw = 0;")
sost("float* forward(Transformer* transformer, int token, int pos) { unsigned long long t0 = tb(); float* r = forward_vero(transformer, token, pos); tb_fw += tb() - t0; return r; }",
     "float* forward(Transformer* transformer, int token, int pos) { unsigned long long t0 = tb(); float* r = forward_vero(transformer, token, pos); tb_fw += tb() - t0; n_fw++; return r; }")
sost("""                100.0 * (tb_fw - tb_mm - tb_q - tb_att - tb_ffn) / tb_fw);
#endif""", """                100.0 * (tb_fw - tb_mm - tb_q - tb_att - tb_ffn) / tb_fw);
#ifdef __APPLE__
        { mach_timebase_info_data_t ti; mach_timebase_info(&ti); double k = (double)ti.numer / ti.denom / 1e6 / n_fw;
          fprintf(stderr, "[ms per forward: matmul %.2f, quantize %.2f, RoPE+attenzione %.2f, SwiGLU %.2f, resto %.2f, totale %.2f]\\n",
                  tb_mm * k, tb_q * k, tb_att * k, tb_ffn * k, (tb_fw - tb_mm - tb_q - tb_att - tb_ffn) * k, tb_fw * k); }
#endif
#endif""")

# 2) matmul_multi: stesso corpo di matmul_v3, righe di piu' matrici con lo stesso ingresso in una regione sola
ini = s.index("void matmul_v3(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d, const int gs) {")
fin = s.index("\n}\n", ini) + 3
corpo = s[ini:fin]
multi = corpo.replace("void matmul_v3(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d, const int gs) {",
                      "void matmul_multi_g(QuantizedTensor *x, int n, int nm, QuantizedTensor **wv, float **outv, const int *dv, const int gs) {")
vecchio = """    int i;
    #pragma omp parallel for private(i) schedule(static)
    for (i = 0; i < d; i++) {
"""
assert multi.count(vecchio) == 1
multi = multi.replace(vecchio, """    int inizio[4], tot = 0;
    for (int m = 0; m < nm; m++) { inizio[m] = tot; tot += dv[m]; }
    int tt;
    #pragma omp parallel for private(tt) schedule(static)
    for (tt = 0; tt < tot; tt++) {
        int m = 0; while (m + 1 < nm && tt >= inizio[m + 1]) m++;
        const int i = tt - inizio[m];
        QuantizedTensor *w = wv[m]; float *xout = outv[m];
""")
multi = ("#ifdef FONDI\nstatic inline __attribute__((always_inline))\n" + multi +
         "static void matmul_multi(QuantizedTensor *x, int n, int nm, QuantizedTensor **wv, float **outv, const int *dv) {\n"
         "#ifdef PROFILO\n    unsigned long long t0 = tb();\n#endif\n"
         "    if (GS == 64) matmul_multi_g(x, n, nm, wv, outv, dv, 64);\n"
         "    else if (GS == 32) matmul_multi_g(x, n, nm, wv, outv, dv, 32);\n"
         "    else matmul_multi_g(x, n, nm, wv, outv, dv, GS);\n"
         "#ifdef PROFILO\n    tb_mm += tb() - t0;\n#endif\n}\n#endif\n")
s = s[:fin] + multi + s[fin:]

# 3) nel forward
sost("""        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
""", """#ifdef FONDI
        { QuantizedTensor *mw[3] = {w->wq + l, w->wk + l, w->wv + l}; float *mo[3] = {s->q, s->k, s->v};
          int md[3] = {dim, kv_dim, kv_dim}; matmul_multi(&s->xq, dim, 3, mw, mo, md); }
#else
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
#endif
""")
sost("""        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
""", """#ifdef FONDI
        { QuantizedTensor *mw[2] = {w->w1 + l, w->w3 + l}; float *mo[2] = {s->hb, s->hb2};
          int md[2] = {hidden_dim, hidden_dim}; matmul_multi(&s->xq, dim, 2, mw, mo, md); }
#else
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
#endif
""")
s = s.replace("// 22 nov: + OpenMP nella matmul_v3 (Leopard, 2 CPU).", "// 22 nov: + OpenMP nella matmul_v3 (Leopard, 2 CPU); runq_leo2: + FONDI (regioni fuse) e PROFILO con mach_absolute_time.", 1)
open('src/runq_leo2.c', 'w').write(s)
print("scritto src/runq_leo2.c")
