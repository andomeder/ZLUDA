#!/usr/bin/env python3
"""Pack a PTX file or a cubin into CUDA fatbin containers for driver testing.

Produces two formats:
  fatbinc (magic 0x466243B1, used by CUDA >= 11.2 runtimes) and
  plain fatbin (magic 0xBA55ED50, older).

Both wrap the same single record. Layout mirrors what the ZLUDA fatbin parser
in zluda/src/impl/fatbin.rs expects.

    mkfatbin.py <in.ptx>  <out prefix>          one PTX record
    mkfatbin.py <in.cubin> <out prefix> --elf   one ELF/cubin record (no PTX),
                                                which is the shape a fatbin
                                                built with `code=sm_XX` has
"""
import struct
import sys

FATBINC_MAGIC = 0x466243B1
FATBIN_MAGIC = 0xBA55ED50
FATBINC_VERSION = 1
FATBIN_VERSION = 1
FILE_KIND_PTX = 1
FILE_KIND_ELF = 2
FILE_VERSION = 0x101
FLAG_64BIT = 0x1


def build_fatbin_file(payload: bytes, kind: int = FILE_KIND_PTX, sm_version: int = 0x80) -> bytes:
    header_size = 64
    payload_size = len(payload)
    header = struct.pack(
        "<HHIIIIIIIIIQQQ",  # u16x2 + u32x9 + u64x3 = 64 bytes, mirrors FatbinFileHeader
        kind,              # kind
        FILE_VERSION,      # version
        header_size,       # header_size
        payload_size,      # payload_size
        0,                 # unknown0
        0,                 # compressed_size
        0,                 # unknown1
        0,                 # unknown2
        sm_version,        # sm_version
        64,                # bit_width
        0,                 # unknown3
        FLAG_64BIT,        # flags
        0,                 # unknown5
        payload_size,      # uncompressed_payload
    )
    assert len(header) == header_size, f"header is {len(header)} bytes, expected {header_size}"
    return header + payload


def build_fatbin(payload: bytes, kind: int = FILE_KIND_PTX, sm_version: int = 0x80) -> bytes:
    """Plain FatbinHeader frame containing one record."""
    body = build_fatbin_file(payload, kind, sm_version)
    header_size = 16
    files_size = len(body)
    header = struct.pack(
        "<IHHQ",
        FATBIN_MAGIC,   # magic
        FATBIN_VERSION,  # version
        header_size,    # header_size
        files_size,     # files_size (excludes this header)
    )
    assert len(header) == header_size, f"header is {len(header)} bytes, expected {header_size}"
    return header + body


def build_fatbinc(payload: bytes, kind: int = FILE_KIND_PTX, sm_version: int = 0x80) -> bytes:
    """FatbincWrapper around a single FatbinHeader frame."""
    fatbin = build_fatbin(payload, kind, sm_version)
    # The wrapper layout: magic u32, version u32, data ptr, filename_or_fatbins ptr.
    # We can't embed real pointers in a file, so we use a 4-byte placeholder
    # offset for `data` and zero for filename_or_fatbins; the C test patches
    # the offset to point at the inline frame before calling.
    wrapper = struct.pack("<II", FATBINC_MAGIC, FATBINC_VERSION)
    return wrapper + fatbin


def main() -> None:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    flags = [arg for arg in sys.argv[1:] if arg.startswith("--")]
    if len(args) != 2:
        print(f"usage: {sys.argv[0]} <in.ptx|in.cubin> <out prefix> [--elf]")
        sys.exit(2)
    payload = open(args[0], "rb").read()
    kind = FILE_KIND_ELF if "--elf" in flags else FILE_KIND_PTX
    prefix = args[1]
    with open(prefix + ".fatbin", "wb") as f:
        f.write(build_fatbin(payload, kind))
    with open(prefix + ".fatbinc", "wb") as f:
        f.write(build_fatbinc(payload, kind))
    print(
        f"wrote {prefix}.fatbin ({len(build_fatbin(payload, kind))} bytes) and "
        f"{prefix}.fatbinc, kind={'ELF' if kind == FILE_KIND_ELF else 'PTX'}"
    )


if __name__ == "__main__":
    main()
