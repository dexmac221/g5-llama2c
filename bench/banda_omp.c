/* 22 nov 2026 — banda di lettura del G5 sotto Leopard con 1 o 2 thread (OpenMP): 256 MB, 5 passate, la migliore. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <altivec.h>
#include <omp.h>
static double ora(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec + t.tv_usec*1e-6; }
int main(void){
    size_t n = 256u<<20; char* g = malloc(n + 16); float* a = (float*)(((uintptr_t)g + 15) & ~(uintptr_t)15);
    long m = (long)(n / 4), i;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < m; i++) a[i] = (float)(i & 7);   /* prima scrittura dal thread che poi legge */
    double meglio = 1e9; float tot = 0;
    for (int r = 0; r < 5; r++) {
        double t0 = ora(); float parz = 0;
        #pragma omp parallel reduction(+:parz)
        {
            int nt = omp_get_num_threads(), id = omp_get_thread_num();
            long blocco = (m / nt) & ~15L, ini = id * blocco, fin = (id == nt - 1) ? m : ini + blocco;
            vector float s0 = (vector float){0,0,0,0}, s1 = s0, s2 = s0, s3 = s0;
            for (long k = ini; k < fin; k += 16) {
                s0 = vec_add(s0, vec_ld(0, a + k)); s1 = vec_add(s1, vec_ld(16, a + k));
                s2 = vec_add(s2, vec_ld(32, a + k)); s3 = vec_add(s3, vec_ld(48, a + k));
            }
            float t[4] __attribute__((aligned(16))); vec_st(vec_add(vec_add(s0,s1),vec_add(s2,s3)), 0, t);
            parz += t[0]+t[1]+t[2]+t[3];
        }
        tot += parz;
        double dt = ora() - t0; if (dt < meglio) meglio = dt;
    }
    printf("lettura: %.2f GB/s con %d thread (controllo %g)\n", n / meglio / 1e9, omp_get_max_threads(), (double)tot);
    return 0;
}
