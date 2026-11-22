#!/usr/bin/env python3
# 21 nov 2026 — dal .bin fp32 di llama2.c (formato v0, quello di run.c) al formato v2 Q8_0 di runq.c,
# senza PyTorch (i .pt di tinyllamas sono pickle: non li usiamo). Stessa logica di export.py/version2_export:
# int8 simmetrico [-127,127] a gruppi di GS (64, dimezzato finché divide dim), norme in fp32.
# uso: quantizza_q8.py modello.bin uscita.bin [be]   ("be" = big-endian, solo per la prova su x86)
import struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
e = '>' if len(sys.argv) > 3 and sys.argv[3] == 'be' else '<'
b = open(src, 'rb').read()
dim, hidden, L, nh, nkv, vocab, seq = struct.unpack_from('<7i', b, 0)
condiviso = vocab > 0
vocab = abs(vocab)
hs = dim // nh
kv = nkv * hs
f = np.frombuffer(b, dtype='<f4', offset=28)
p = 0
def prendi(n, forma):
    global p
    a = f[p:p + n].reshape(forma); p += n
    return a
tok = prendi(vocab * dim, (vocab, dim))
rms_att = prendi(L * dim, (L, dim))
wq = prendi(L * dim * dim, (L, dim, dim))
wk = prendi(L * dim * kv, (L, kv, dim))
wv = prendi(L * dim * kv, (L, kv, dim))
wo = prendi(L * dim * dim, (L, dim, dim))
rms_ffn = prendi(L * dim, (L, dim))
w1 = prendi(L * hidden * dim, (L, hidden, dim))
w2 = prendi(L * dim * hidden, (L, dim, hidden))
w3 = prendi(L * hidden * dim, (L, hidden, dim))
rms_fin = prendi(dim, (dim,))
p += seq * hs // 2 * 2          # freq_cis_real/imag, non servono
wcls = None if condiviso else prendi(vocab * dim, (vocab, dim))
assert condiviso or p == len(f), (p, len(f))

gs = 64
while dim % gs:
    gs //= 2
pesi = [tok, *wq, *wk, *wv, *wo, *w1, *w2, *w3] + ([] if condiviso else [wcls])

def q80(w):
    w = w.astype(np.float32).reshape(-1, gs)
    scala = (np.abs(w).max(axis=1) / np.float32(127.0)).astype(np.float32)
    q = np.round(w / scala[:, None]).astype(np.int8)
    errore = np.abs(q.astype(np.float32) * scala[:, None] - w).max()
    return q, scala, errore

with open(dst, 'wb') as o:
    o.write(struct.pack(e + 'I', 0x616b3432))
    o.write(struct.pack(e + 'i', 2))
    o.write(struct.pack(e + '7i', dim, hidden, L, nh, nkv, vocab, seq))
    o.write(struct.pack('B', int(condiviso)))
    o.write(struct.pack(e + 'i', gs))
    o.write(b'\0' * (256 - o.tell()))
    for a in (rms_att, rms_ffn, rms_fin):
        o.write(a.astype(e + 'f4').tobytes())
    peggiore = 0.0
    for w in pesi:
        q, s, err = q80(w)
        o.write(q.tobytes())
        o.write(s.astype(e + 'f4').tobytes())
        peggiore = max(peggiore, float(err))
print(f"{dst}: dim={dim} hidden={hidden} strati={L} GS={gs} condiviso={condiviso} errore max={peggiore:.5f}")
