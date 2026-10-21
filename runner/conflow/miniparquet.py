"""Built-in parquet reader used only when pyarrow is not installed (e.g. offline workspaces).

Supports the flat float64 SuperWing split tables: SNAPPY/UNCOMPRESSED, PLAIN and dictionary
encodings, data page v1/v2, no nulls. Checked against pandas+pyarrow in tests/test_t1_reader.py."""
import struct, sys
import numpy as np


class Compact:
    def __init__(self, b, p=0):
        self.b, self.p = b, p

    def varint(self):
        r = s = 0
        while True:
            c = self.b[self.p]; self.p += 1
            r |= (c & 0x7F) << s; s += 7
            if not c & 0x80:
                return r

    def zz(self):
        v = self.varint(); return (v >> 1) ^ -(v & 1)

    def val(self, t):
        if t in (1, 2): return t == 1
        if t == 3: v = self.b[self.p]; self.p += 1; return v
        if t in (4, 5, 6): return self.zz()
        if t == 7: v = struct.unpack_from('<d', self.b, self.p)[0]; self.p += 8; return v
        if t == 8: n = self.varint(); v = bytes(self.b[self.p:self.p + n]); self.p += n; return v
        if t in (9, 10):
            h = self.b[self.p]; self.p += 1
            n = h >> 4; et = h & 0x0F
            if n == 15: n = self.varint()
            return [self.val(et) for _ in range(n)]
        if t == 12: return self.struct()
        raise ValueError(t)

    def struct(self):
        d = {}; last = 0
        while True:
            h = self.b[self.p]; self.p += 1
            if h == 0: return d
            t = h & 0x0F; delta = h >> 4
            fid = last + delta if delta else self.zz()
            last = fid
            d[fid] = self.val(t)


def snappy(b):
    c = Compact(b); n = c.varint(); p = c.p
    out = bytearray()
    while p < len(b):
        tag = b[p]; p += 1; k = tag & 3
        if k == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(b[p:p + nb], 'little'); p += nb
            ln += 1; out += b[p:p + ln]; p += ln
        else:
            if k == 1:
                ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | b[p]; p += 1
            elif k == 2:
                ln = (tag >> 2) + 1; off = int.from_bytes(b[p:p + 2], 'little'); p += 2
            else:
                ln = (tag >> 2) + 1; off = int.from_bytes(b[p:p + 4], 'little'); p += 4
            for _ in range(ln):
                out.append(out[-off])
    assert len(out) == n
    return bytes(out)


def rle_hybrid(b, bw, count):
    c = Compact(b); out = []
    while len(out) < count and c.p < len(b):
        h = c.varint()
        if h & 1:
            ng = h >> 1; nbytes = ng * bw
            bits = int.from_bytes(b[c.p:c.p + nbytes], 'little'); c.p += nbytes
            for i in range(ng * 8):
                out.append((bits >> (i * bw)) & ((1 << bw) - 1))
        else:
            n = h >> 1; w = (bw + 7) // 8
            v = int.from_bytes(b[c.p:c.p + w], 'little'); c.p += w
            out += [v] * n
    return out[:count]


def read(path):
    b = open(path, 'rb').read()
    flen = struct.unpack_from('<i', b, len(b) - 8)[0]
    meta = Compact(b, len(b) - 8 - flen).struct()
    schema = meta[2]; names = [s[4].decode() for s in schema[1:]]
    types = [s.get(1) for s in schema[1:]]
    cols = {nm: [] for nm in names}
    for rg in meta[4]:
        for ci, cc in enumerate(rg[1]):
            md = cc[3]; codec = md[4]; nvals = md[5]
            start = md.get(11, md[9]); p = start
            dictionary = None; vals = []
            while len(vals) < nvals:
                c = Compact(b, p); ph = c.struct(); p = c.p
                ptype, usize, csize = ph[1], ph[2], ph[3]
                raw = b[p:p + csize]; p += csize
                if ptype == 3 and 8 in ph:  # data page v2
                    h = ph[8]; lvl = h.get(5, 0) + h.get(6, 0)  # audit-allow R1: thrift field default, not a metric
                    body = raw[lvl:]
                    data = snappy(body) if (codec == 1 and h.get(7, True)) else body
                    n = h[1]; enc = h[4]
                else:
                    data = snappy(raw) if codec == 1 else raw
                if ptype == 2:  # dictionary
                    n = ph[7][1]
                    dictionary = np.frombuffer(data[:8 * n], '<f8' if types[ci] == 5 else '<i8')
                    continue
                if ptype == 0:
                    h = ph[5]; n = h[1]; enc = h[2]
                    # required columns have no def levels; optional (rep 1) have RLE def levels with length prefix
                    if schema[ci + 1].get(3, 0) == 1:  # audit-allow R1: thrift field default, not a metric
                        ln = struct.unpack_from('<i', data, 0)[0]; data = data[4 + ln:]
                if enc == 0:
                    vals += list(np.frombuffer(data[:8 * n], '<f8' if types[ci] == 5 else '<i8'))
                else:
                    bw = data[0]; idx = rle_hybrid(data[1:], bw, n)
                    vals += list(dictionary[idx])
            cols[names[ci]] += vals
    return {k: np.asarray(v) for k, v in cols.items()}


