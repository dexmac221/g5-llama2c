/* 21 nov 2026 — banda di lettura della memoria sul G5: somma 256 MB, 5 passate, la migliore. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#ifdef __ALTIVEC__
#include <altivec.h>
#endif
static double ora(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec + t.tv_usec*1e-6; }
int main(void){
    size_t n = 256u<<20; char* g = malloc(n + 16); float* a = (float*)(((uintptr_t)g + 15) & ~(uintptr_t)15);
    size_t m = n / 4; for (size_t i = 0; i < m; i++) a[i] = (float)(i & 7);
    double meglio = 1e9; float tot = 0;
    for (int r = 0; r < 5; r++) {
        double t0 = ora();
#ifdef __ALTIVEC__
        vector float s0 = (vector float){0,0,0,0}, s1 = s0, s2 = s0, s3 = s0;
        for (size_t i = 0; i < m; i += 16) {
            s0 = vec_add(s0, vec_ld(0, a + i)); s1 = vec_add(s1, vec_ld(16, a + i));
            s2 = vec_add(s2, vec_ld(32, a + i)); s3 = vec_add(s3, vec_ld(48, a + i));
        }
        float t[4] __attribute__((aligned(16))); vec_st(vec_add(vec_add(s0,s1),vec_add(s2,s3)), 0, t); tot += t[0]+t[1]+t[2]+t[3];
#else
        float s0=0,s1=0,s2=0,s3=0;
        for (size_t i = 0; i < m; i += 4) { s0 += a[i]; s1 += a[i+1]; s2 += a[i+2]; s3 += a[i+3]; }
        tot += s0+s1+s2+s3;
#endif
        double dt = ora() - t0; if (dt < meglio) meglio = dt;
    }
    printf("lettura: %.2f GB/s (%s, controllo %g)\n", n / meglio / 1e9,
#ifdef __ALTIVEC__
        "AltiVec",
#else
        "scalare",
#endif
        (double)tot);
    return 0;
}
