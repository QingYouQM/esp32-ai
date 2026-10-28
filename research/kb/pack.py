"""Pack entries into the blob the device reads.

The device does no parsing. It holds a fixed-stride index and a UTF-8 blob, and
a button press turns into an index lookup, a bounds check and a memcpy of the
material straight into the prompt. Everything that needs judgement - field
order, punctuation, length - already happened here, at build time.

Layout, little-endian:

    header  12 B   4s magic "KBRD", u16 version, u16 count, u16 stride, u16 reserved
    index   N x 12  u32 off, u16 id_len, u16 title_len, u16 mat_len, u16 flags
    blob    ...    id, title, material, concatenated, UTF-8, no terminators

flags bit 0 is set when any field of the entry is inferred, so a reviewer can
find the guesses without opening the JSON.
"""

import argparse
import json
import struct
from pathlib import Path

from .render import material_text
from .spec import MATERIAL_MAX
from .validate import check

MAGIC = b"KBRD"
VERSION = 1
HEADER = struct.Struct("<4sHHHH")
RECORD = struct.Struct("<IHHHH")
FLAG_INFERRED = 0x0001


def pack(entries):
    blob = bytearray()
    records = []
    for entry in entries:
        material = material_text(entry).encode("utf-8")
        title = entry["title"].encode("utf-8")
        ident = entry["id"].encode("utf-8")
        inferred = any(bool(f.get("inferred")) for f in entry["fields"].values())
        records.append((
            len(blob),
            len(ident),
            len(title),
            len(material),
            FLAG_INFERRED if inferred else 0,
        ))
        blob += ident + title + material
        if len(material) > MATERIAL_MAX * 3 + 16:
            raise SystemExit(f"{entry['id']}: material does not fit the budget")

    head = HEADER.pack(MAGIC, VERSION, len(records), RECORD.size, 0)
    index = b"".join(RECORD.pack(*r) for r in records)
    return head + index + bytes(blob)


def read(data):
    """Parse the blob back into (id, title, material) tuples."""
    magic, version, count, stride, _ = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise SystemExit("not a kb blob")
    if version != VERSION:
        raise SystemExit(f"version {version} is not {VERSION}")
    if stride != RECORD.size:
        raise SystemExit(f"stride {stride} is not {RECORD.size}")
    base = HEADER.size + count * stride
    out = []
    for i in range(count):
        off, id_len, title_len, mat_len, flags = RECORD.unpack_from(
            data, HEADER.size + i * stride)
        if off + id_len + title_len + mat_len > len(data) - base:
            raise SystemExit(f"record {i} runs past the blob")
        p = base + off
        ident = data[p:p + id_len].decode("utf-8")
        title = data[p + id_len:p + id_len + title_len].decode("utf-8")
        material = data[p + id_len + title_len:p + id_len + title_len + mat_len]
        out.append((ident, title, material.decode("utf-8"), flags))
    return out


def main():
    ap = argparse.ArgumentParser(description="pack KB entries for the device")
    ap.add_argument("entries", type=Path)
    ap.add_argument("--out", type=Path, default=Path("research/kb/build/kb.bin"))
    ap.add_argument("--verify", action="store_true",
                    help="parse the result back and compare against the source")
    args = ap.parse_args()

    files = sorted(args.entries.glob("*.json"))
    if not files:
        raise SystemExit(f"no entries in {args.entries}")

    entries = []
    bad = 0
    for file in files:
        entry = json.loads(file.read_text(encoding="utf-8"))
        problems = [p for p in check(entry) if p.level == "error"]
        if problems:
            bad += len(problems)
            print(f"{file.name}: {len(problems)} error(s), skipped")
            for problem in problems:
                print(f"  {problem}")
            continue
        entries.append(entry)
    if bad:
        raise SystemExit(f"{bad} error(s); nothing packed")

    entries.sort(key=lambda e: e["id"])
    data = pack(entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)

    inferred = sum(1 for e in entries
                   if any(bool(f.get("inferred")) for f in e["fields"].values()))
    print(f"{len(entries)} entries, {len(data):,} bytes "
          f"({len(data) / 1024:.1f} KiB), {inferred} with inferred fields")

    if args.verify:
        got = read(data)
        if len(got) != len(entries):
            raise SystemExit("record count changed on the round trip")
        for entry, (ident, title, material, flags) in zip(entries, got):
            if ident != entry["id"] or title != entry["title"]:
                raise SystemExit(f"header mismatch for {entry['id']}")
            if material != material_text(entry):
                raise SystemExit(f"material mismatch for {entry['id']}")
            if flags != (FLAG_INFERRED if any(
                    bool(f.get("inferred")) for f in entry["fields"].values()) else 0):
                raise SystemExit(f"flags mismatch for {entry['id']}")
        print("round trip matches: every id, title and material byte for byte")


if __name__ == "__main__":
    main()
