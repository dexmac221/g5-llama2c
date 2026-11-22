#!/bin/sh
# 22 nov 2026 (revisione di Claude web): costo delle regioni OpenMP, profilo 1/2 CPU, regioni fuse (97 -> 61 per token).
cd ~/llama2c-leo
F="-O3 -mcpu=970 -maltivec -fopenmp -std=gnu99 -DSENZA_MMAP -DVELOCE -DV3"
gcc-4.2 -O3 -fopenmp -std=gnu99 -o regioni regioni.c || exit 1
gcc-4.2 $F -DFONDI -o runq_fondi runq_leo2.c -lm || exit 1
gcc-4.2 $F -DPROFILO -o runq_prof runq_leo2.c -lm || exit 1
gcc-4.2 $F -DFONDI -DPROFILO -o runq_fondi_prof runq_leo2.c -lm || exit 1
sleep 20
echo "== $(date +%T) carico: $(uptime | sed 's/.*averages: //')"
for t in 1 2; do OMP_NUM_THREADS=$t ./regioni; done
A="stories110M_q80.bin -z tokenizer.bin -t 0 -n 128 -i Once"
for r in 1 2 3; do for t in 1 2; do for b in runq_leo runq_fondi; do
  OMP_NUM_THREADS=$t ./$b $A > out_$b.txt 2> err.txt
  echo "$b thread=$t giro=$r $(grep achieved err.txt) md5=$(md5 -q out_$b.txt)"
done; done; done
for r in 1 2; do for t in 1 2; do for b in runq_prof runq_fondi_prof; do
  OMP_NUM_THREADS=$t ./$b $A > /dev/null 2> err.txt
  echo "$b thread=$t giro=$r $(grep achieved err.txt)"; grep "ms per forward" err.txt; grep "profilo" err.txt
done; done; done
echo "== $(date +%T) FINE"
