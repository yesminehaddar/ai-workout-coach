"""Lecture des archives QEVD-FIT-300k découpées (Part-X.z01, .z02, .zip) sans les recombiner.

Le dernier morceau (.zip) contient l'index de toute l'archive. Pour chaque fichier, l'index dit
dans quel morceau il commence (disk) et à quel octet. On peut donc lister l'archive avec le seul
.zip, et extraire n'importe quel fichier dont le morceau est présent sur le disque.

Usage :
    python scripts/qevd_split_zip.py list  <dossier des zips>            # résumé par morceau
    python scripts/qevd_split_zip.py get   <dossier des zips> <nom> <sortie>
"""
import struct
import sys
import zlib
from pathlib import Path


def _segment_paths(last_zip: Path, n_disks: int) -> list[Path]:
    stem = last_zip.with_suffix("")
    return [stem.with_suffix(f".z{i + 1:02d}") for i in range(n_disks - 1)] + [last_zip]


def read_index(last_zip: Path) -> tuple[list[dict], list[Path]]:
    data = last_zip.read_bytes()[-(1 << 20):]  # la fin du fichier suffit pour les en-têtes
    tail_start = last_zip.stat().st_size - len(data)
    i = data.rfind(b"PK\x06\x06")  # fin d'archive zip64
    if i < 0:
        raise ValueError("pas d'en-tête zip64")
    (_, _, _, _, this_disk, cd_disk, _, n_entries, cd_size, cd_offset) = struct.unpack(
        "<IQHHIIQQQQ", data[i:i + 56])
    if cd_disk != this_disk:
        raise ValueError("l'index n'est pas dans le dernier morceau")
    with last_zip.open("rb") as f:
        f.seek(cd_offset)
        cd = f.read(cd_size)
    entries, p = [], 0
    for _ in range(n_entries):
        (sig, _, _, _, method, _, _, crc, csize, usize, nlen, xlen, clen, disk, _, _,
         off) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        assert sig == 0x02014B50
        name = cd[p + 46:p + 46 + nlen].decode("utf-8")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):  # champs zip64 : tailles, offset, disque
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, r = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", vals[r:r + 8])[0]; r += 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<I", vals[r:r + 4])[0]
            q += 4 + hlen
        entries.append({"name": name, "method": method, "crc": crc, "csize": csize,
                        "usize": usize, "disk": disk, "offset": off})
        p += 46 + nlen + xlen + clen
    del tail_start
    return entries, _segment_paths(last_zip, this_disk + 1)


def extract(entry: dict, segments: list[Path]) -> bytes:
    """Lit un fichier ; il peut déborder sur le morceau suivant."""
    seg = entry["disk"]
    f = segments[seg].open("rb")
    f.seek(entry["offset"])
    hdr = f.read(30)
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    f.seek(entry["offset"] + 30 + nlen + xlen)
    raw, need = b"", entry["csize"]
    while need > 0:
        chunk = f.read(need)
        raw += chunk
        need -= len(chunk)
        if need > 0:
            f.close()
            seg += 1
            f = segments[seg].open("rb")
            if seg == 1 or raw == b"":  # le 1er morceau commence par une signature de découpage
                pass
    f.close()
    out = raw if entry["method"] == 0 else zlib.decompress(raw, -15)
    if zlib.crc32(out) & 0xFFFFFFFF != entry["crc"]:
        raise ValueError(f"CRC incorrect pour {entry['name']}")
    return out


def main():
    cmd, folder = sys.argv[1], Path(sys.argv[2])
    zips = sorted(folder.glob("QEVD-FIT-300k-Part-*.zip"))
    if cmd == "list":
        for z in zips:
            entries, segs = read_index(z)
            have = [s.exists() for s in segs]
            per = {}
            for e in entries:
                per.setdefault(e["disk"], 0)
                per[e["disk"]] += 1
            ok = sum(n for d, n in per.items() if have[d])
            print(f"{z.name}: {len(entries)} fichiers, morceaux présents {have}, "
                  f"fichiers par morceau {dict(sorted(per.items()))} -> extractibles {ok}")
    elif cmd == "get":
        name, out = sys.argv[3], Path(sys.argv[4])
        for z in zips:
            entries, segs = read_index(z)
            for e in entries:
                if e["name"].endswith(name):
                    out.write_bytes(extract(e, segs))
                    print(f"{e['name']} -> {out} ({e['usize']} octets)")
                    return
        raise SystemExit(f"{name} introuvable")


if __name__ == "__main__":
    main()
