# llama2.c on a 2004 Power Mac G5 (MorphOS and Mac OS X Leopard)

Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) ported to MorphOS 3.20 on a Power Mac G5, with AltiVec kernels for fp32 and for 8-bit weights. The int8 kernel is built on `vmsummbm`, AltiVec's signed × unsigned byte multiply-sum from 1999. The same code also runs on Mac OS X 10.5.8 Leopard, where OpenMP uses both processors.

The write-up, with the reasoning behind each step, is an article on Towards AI; the link will be added here once it is published.

## Results

Power Mac G5 (PowerMac7,3), two PowerPC 970 at 1.8 GHz (130 nm, PVR 0x0039 rev 2.2), 900 MHz bus, 2 GB DDR400. Greedy decoding, prompt "Once", 128 tokens for stories110M and 256 for stories15M. Each MorphOS figure is the median of three interleaved runs; the Leopard figures are three runs that differed by less than 0.5%.

| Tokens per second | stories15M | stories110M |
|---|---|---|
| fp32, plain C | 15.7 | 2.26 |
| fp32, AltiVec | 28.9 | 4.47 |
| int8, plain C | 11.0 | 1.91 |
| int8, `vmsummbm`, one group at a time | 29.8 | 7.77 |
| int8, four groups at a time | 48.9 | 11.4 |
| int8, faster RoPE, rounding and attention | 63.1 | 12.9 |
| **int8, weights stored unsigned (final, MorphOS, one CPU)** | **79.1** | **13.7** |
| int8 final, Leopard, one CPU (GCC 4.2) | 66.0 | 13.1 |
| **int8 final, Leopard, two CPUs (OpenMP)** | 76.0 | **17.2** |

Measured memory read bandwidth: 2.25 GB/s on one CPU (MorphOS), 2.21 GB/s on one CPU and 3.63 GB/s on two (Leopard). The 110M int8 model reads 116 MB per token, so one CPU tops out near 19 tokens/s.

The one-group-at-a-time version didn't reproduce across sessions: 9.14 and 37.2 tokens/s on 21 Nov 2026, 7.6-8.1 and 29.5-30.3 on 22 Nov. The others came within 6%. Raw output of the 22 Nov runs is in `results/`.

## What's here

| Path | What it is |
|---|---|
| `patches/patch_run_mos.py`, `patches/patch_runq_mos.py` | Turn llama2.c's `run.c` and `runq.c` (commit 350e04f) into the MorphOS versions |
| `src/run_mos.c`, `src/runq_mos.c` | The generated sources (fp32 and int8) |
| `src/runq_leo.c` | `runq_mos.c` plus an OpenMP pragma on the int8 matrix multiply, for Leopard |
| `tools/quantizza_q8.py` | NumPy quantizer: fp32 `.bin` (llama2.c v0 format) to the Q8_0 v2 format of `runq.c`, without PyTorch or pickle |
| `tools/converti_be.py` | Byte-swapped test files, to check the big-endian path on an x86 PC |
| `bench/banda.c`, `bench/banda_omp.c` | Memory read bandwidth: 256 MB, best of five; the second one splits the read between OpenMP threads |
| `scripts/morphos/build_all` | AmigaDOS script that builds every variant in the table |
| `scripts/morphos/rimisura` | AmigaDOS script for the three interleaved runs |
| `scripts/leopard/misura_leo.sh` | Leopard runs with one and two threads |
| `results/` | Raw output of the 22 Nov 2026 MorphOS runs |

Comments in the code are in Italian.

## Compile-time switches (`runq_mos.c`)

| Switch | Effect |
|---|---|
| `SENZA_MMAP` | Reads the model with `fread` (MorphOS has no `mmap`), byte-swaps the 32-bit fields on big-endian machines, uses `gettimeofday` |
| *(none)* | AltiVec `vmsummbm` kernel, one quantization group at a time |
| `QUATTRO` | Four groups at a time, with a transpose-sum and one float conversion |
| `VELOCE` | RoPE sines and cosines once per token, `round()` replaced by add-half-and-truncate in double, AltiVec attention, 16-byte aligned state |
| `V3` | Weights stored as `w + 128` at load time; the correction `128 × Σx` is computed once per group and call |
| `VERIFICA` | Compares the AltiVec matrix multiply with the plain C one on the first three calls |
| `PROFILO` | Timebase timers around each part of the forward pass |
| `FORZA_SWAP` | Forces the byte swapping on little-endian machines, for testing |

## How to reproduce

1. Get the models from Hugging Face (`karpathy/tinyllamas`: `stories15M.bin`, `stories110M.bin`) and llama2.c's `tokenizer.bin`.
2. Quantize on any machine with Python and NumPy:
   `python3 tools/quantizza_q8.py stories110M.bin stories110M_q80.bin`
3. **MorphOS** (SDK 2018, GCC 6.4.0): copy `src/`, the models and the tokenizer to `Work:llama2c/` and run `Execute scripts/morphos/build_all`. Over SSH the SDK environment isn't loaded: without `Execute SYS:Development/Startup` GCC exits with code 1 and no message. `-mcpu=970` doesn't assemble with the 2018 toolchain; `-mcpu=7450` works and enables AltiVec, so plain C builds need `-mno-altivec`.
4. **Leopard** (Xcode, Apple GCC 4.2.1):
   `gcc-4.2 -O3 -mcpu=970 -maltivec -fopenmp -std=gnu99 -DSENZA_MMAP -DVELOCE -DV3 -o runq_leo src/runq_leo.c -lm`
   then `OMP_NUM_THREADS=2 ./runq_leo stories110M_q80.bin -z tokenizer.bin -t 0 -n 128 -i Once`
5. To regenerate the sources from upstream: clone llama2.c, `git checkout 350e04f`, then
   `python3 patches/patch_run_mos.py run.c run_mos.c` and `python3 patches/patch_runq_mos.py runq.c runq_mos.c`.

## Prior work

- Andrew Rossignol ran llama2.c on a PowerBook G4 with AltiVec fp32 (March 2025).
- Scott (Scottcjn) used `vmsummbm` on a G5 under Leopard for ternary BitNet weights ([microsoft/BitNet#404](https://github.com/microsoft/BitNet/pull/404)) and maintains [llama-cpp-tigerleopard](https://github.com/Scottcjn/llama-cpp-tigerleopard).
- I found no earlier language model running locally on MorphOS.

## License

MIT. The files in `src/` are derived from llama2.c, © 2023 Andrej Karpathy, MIT: see `LICENSE-llama2c`. Everything else: see `LICENSE`.
