/* 22 nov 2026 — quanto costa una regione parallela OpenMP sul G5 sotto Leopard (libgomp di Apple GCC 4.2):
   100.000 regioni da 768 righe quasi vuote, migliore di 3 giri. */
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
static double ora(void){ struct timeval t; gettimeofday(&t, NULL); return t.tv_sec + t.tv_usec * 1e-6; }
int main(void) {
    const int N = 100000; static volatile float sink[768]; double meglio = 1e9;
    for (int r = 0; r < 3; r++) {
        double t0 = ora();
        for (int k = 0; k < N; k++) {
            int i;
            #pragma omp parallel for private(i) schedule(static)
            for (i = 0; i < 768; i++) sink[i] += 1.0f;
        }
        double dt = ora() - t0; if (dt < meglio) meglio = dt;
    }
    printf("regione parallela: %.2f us con %d thread\n", meglio / N * 1e6, omp_get_max_threads());
    return 0;
}
