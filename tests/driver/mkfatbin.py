#!/usr/bin/env python3
"""Pack a PTX file into CUDA fatbin containers for driver-layer testing.

Produces two formats:
  fatbinc (magic 0x466243B1, used by CUDA >= 11.2 runtimes) and
  plain fatbin (magic 0xBA55ED50, older).

Both wrap the same single PTX file record. Layout mirrors what the ZLUDA
fatbin parser in zluda/src/impl/fatbin.rs expects.
"""
import struct
import sys

FATBINC_MAGIC = 0x466243B1
FATBIN_MAGIC = 0xBA55ED50
FATBINC_VERSION = 1
FATBIN_VERSION = 1
FILE_KIND_PTX = 1
FILE_VERSION = 0x101
FLAG_64BIT = 0x1


def build_fatbin_file(ptx: bytes) -> bytes:
    header_size = 64
    payload_size = len(ptx)
    header = struct.pack(
        "<HHIIIIIIIIIQQQ",  # u16x2 + u32x9 + u64x3 = 64 bytes, mirrors FatbinFileHeader
        FILE_KIND_PTX,     # kind
        FILE_VERSION,      # version
        header_size,       # header_size
        payload_size,      # payload_size
        0,                 # unknown0
        0,                 # compressed_size
        0,                 # unknown1
        0,                 # unknown2
        0x80,              # sm_version (sm_80)
        64,                # bit_width
        0,                 # unknown3
        FLAG_64BIT,        # flags
        0,                 # unknown5
        payload_size,      # uncompressed_payload
    )
    assert len(header) == header_size, f"header is {len(header)} bytes, expected {header_size}"
    return header + ptx


def build_fatbin(ptx: bytes) -> bytes:
    """Plain FatbinHeader frame containing one PTX file."""
    body = build_fatbin_file(ptx)
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


def build_fatbinc(ptx: bytes) -> bytes:
    """FatbincWrapper around a single FatbinHeader frame."""
    fatbin = build_fatbin(ptx)
    # The wrapper layout: magic u32, version u32, data ptr, filename_or_fatbins ptr.
    # We can't embed real pointers in a file, so we use a 4-byte placeholder
    # offset for `data` and zero for filename_or_fatbins; the C test patches
    # the offset to point at the inline frame before calling.
    wrapper = struct.pack("<II", FATBINC_MAGIC, FATBINC_VERSION)
    return wrapper + fatbin


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <in.ptx> <out prefix>")
        sys.exit(2)
    ptx = open(sys.argv[1], "rb").read()
    prefix = sys.argv[2]
    with open(prefix + ".fatbin", "wb") as f:
        f.write(build_fatbin(ptx))
    with open(prefix + ".fatbinc", "wb") as f:
        f.write(build_fatbinc(ptx))
    print(f"wrote {prefix}.fatbin ({len(build_fatbin(ptx))} bytes) and {prefix}.fatbinc")


if __name__ == "__main__":
    main()
