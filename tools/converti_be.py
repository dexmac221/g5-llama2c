#!/usr/bin/env python3
# 21 nov 2026 — solo per la PROVA su un PC x86: file di llama2.c girati in big-endian
# (modello: tutto a 32 bit; tokenizer: int max_len, poi per token float score, int len, byte)
import struct, sys
cosa, src, dst = sys.argv[1:4]
b = open(src, 'rb').read()
if cosa == 'modello':
    n = len(b) // 4
    out = struct.pack('>%dI' % n, *struct.unpack('<%dI' % n, b[:n*4])) + b[n*4:]
else:
    out = bytearray(); p = 0
    (m,) = struct.unpack_from('<i', b, p); out += struct.pack('>i', m); p += 4
    while p < len(b):
        sc, ln = struct.unpack_from('<fi', b, p); p += 8
        out += struct.pack('>fi', sc, ln) + b[p:p+ln]; p += ln
    out = bytes(out)
open(dst, 'wb').write(out); print(dst, len(out))
