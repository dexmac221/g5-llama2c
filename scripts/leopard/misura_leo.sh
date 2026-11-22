#!/bin/sh
# 22 nov 2026, G5 sotto Leopard: banda e llama2c int8 (V3+VELOCE, OpenMP) con 1 e 2 thread, 3 giri ciascuno.
cd ~/llama2c-leo
echo "== $(date +%T) carico prima: $(uptime | sed 's/.*averages: //')"
for t in 1 2; do for r in 1 2 3; do OMP_NUM_THREADS=$t ./banda_omp; done; done
for m in 110 15; do
  N=128; [ $m = 15 ] && N=256
  for t in 1 2; do for r in 1 2 3; do
    OMP_NUM_THREADS=$t ./runq_leo stories${m}M_q80.bin -z tokenizer.bin -t 0 -n $N -i Once > out_${m}_t$t.txt 2> err.txt
    echo "${m}M thread=$t giro=$r $(grep 'achieved' err.txt)"
  done; md5 -q out_${m}_t$t.txt; done
done
echo "== $(date +%T) FINE"
