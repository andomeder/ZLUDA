//! Parses CUDA fatbinary containers (fatbin) and extracts PTX source.
//!
//! Modern CUDA apps embed kernels as fatbins produced by `fatbinary`. The
//! container can be either a `FatbincWrapper` (magic `FbC\xB1`, CUDA >= 11.2)
//! wrapping one or more `FatbinHeader` frames, or a bare `FatbinHeader`
//! (magic `P\xEDU\xBA`, older CUDA). Each frame holds a sequence of file
//! records; we only consume `PTX` records (kind 0x01), optionally
//! LZ4-compressed, and feed the resulting source to the SPIR-V translator.
//! CUBIN/ELF records (kind 0x02) are skipped.

use std::{
    cmp,
    ffi::{c_void, CStr},
    mem,
    os::raw::{c_ulonglong, c_uint, c_ushort},
    ptr, slice,
};

use super::CUresult;

const FATBINC_MAGIC: [u8; 4] = 0x466243B1u32.to_le_bytes();
const FATBIN_MAGIC: [u8; 4] = 0xBA55ED50u32.to_le_bytes();
const FATBINC_VERSION_V1: c_uint = 0x1;
const FATBINC_VERSION_V2: c_uint = 0x2;
const FATBIN_VERSION: c_ushort = 0x1;
const FATBIN_FILE_KIND_PTX: c_ushort = 0x1;
const FATBIN_FILE_KIND_ELF: c_ushort = 0x2;
const FATBIN_FILE_VERSION_CURRENT: c_ushort = 0x101;
const FATBIN_FILE_FLAG_64BIT: u64 = 0x1;
const FATBIN_FILE_FLAG_COMPRESSED_LZ4: u64 = 0x2000;
const FATBIN_FILE_FLAG_COMPRESSED_ZSTD: u64 = 0x8000;

const MAX_DECOMPRESSION_BOUND: usize = 64 * 1024 * 1024;

#[repr(C)]
#[derive(Copy, Clone)]
struct FatbincWrapper {
    magic: c_uint,
    version: c_uint,
    data: *const FatbinHeader,
    filename_or_fatbins: *const c_void,
}

#[repr(C, align(8))]
#[derive(Copy, Clone)]
struct FatbinHeader {
    magic: c_uint,
    version: c_ushort,
    header_size: c_ushort,
    files_size: c_ulonglong,
}

/// Per-file record header of a fatbin frame.
///
/// The header is variable-sized: `header_size` is 0x40 for plain records and
/// 0x48/0x60 for records that carry a kernel name inline. Only the first
/// 0x3c bytes have a fixed meaning:
///
/// ```text
/// 0x00 u16 kind          (1 = PTX, 2 = ELF/cubin, 3 = host registration)
/// 0x02 u16 version       (0x101)
/// 0x04 u32 header_size   (0x40, 0x48 or 0x60)
/// 0x08 u32 payload_size  (size of the record payload)
/// 0x0c u32 unknown0
/// 0x10 u32 compressed_size  (size of the compressed part of the payload)
/// 0x14 u32 unknown1
/// 0x18 u32 unknown2
/// 0x1c u32 sm_version
/// 0x20 u32 bit_width
/// 0x24 u32 unknown3
/// 0x28 u64 flags
/// 0x30 u64 unknown5
/// 0x38 u32 uncompressed_size   <-- u32, NOT u64
/// 0x3c u32 unknown6
/// 0x40 ... optional kernel name / options
/// ```
///
/// The uncompressed size is a u32: reading it as a u64 picks up the
/// following zero word and yields a nonsense size (0x24dd9e00000000),
/// which makes the LZ4 decompression fail with "invalid PTX".
///
/// The fields are decoded explicitly instead of casting a byte slice to a
/// `#[repr(C)]` struct, because the record header size varies and a struct
/// cast would read past the end of a 0x40-byte header.
#[derive(Copy, Clone)]
struct FatbinFileHeader {
    kind: c_ushort,
    version: c_ushort,
    header_size: c_uint,
    payload_size: c_uint,
    unknown0: c_uint,
    compressed_size: c_uint,
    unknown1: c_uint,
    unknown2: c_uint,
    sm_version: c_uint,
    bit_width: c_uint,
    unknown3: c_uint,
    flags: u64,
    unknown5: c_ulonglong,
    uncompressed_payload: c_uint,
    unknown6: c_uint,
}

/// Number of bytes with a fixed meaning at the start of a record header.
const FATBIN_FILE_HEADER_FIXED_SIZE: usize = 0x40;

impl FatbinFileHeader {
    unsafe fn parse(base: *const u8) -> Option<FatbinFileHeader> {
        if base.add(FATBIN_FILE_HEADER_FIXED_SIZE).is_null() {
            return None;
        }
        let u16_at = |off: usize| u16::from_le(ptr::read_unaligned(base.add(off) as *const u16));
        let u32_at = |off: usize| u32::from_le(ptr::read_unaligned(base.add(off) as *const u32));
        let u64_at = |off: usize| u64::from_le(ptr::read_unaligned(base.add(off) as *const u64));
        Some(FatbinFileHeader {
            kind: u16_at(0x00),
            version: u16_at(0x02),
            header_size: u32_at(0x04),
            payload_size: u32_at(0x08),
            unknown0: u32_at(0x0c),
            compressed_size: u32_at(0x10),
            unknown1: u32_at(0x14),
            unknown2: u32_at(0x18),
            sm_version: u32_at(0x1c),
            bit_width: u32_at(0x20),
            unknown3: u32_at(0x24),
            flags: u64_at(0x28),
            unknown5: u64_at(0x30),
            uncompressed_payload: u32_at(0x38),
            unknown6: u32_at(0x3c),
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum BinaryKind {
    Ptx,
    FatbincWrapper,
    FatbinHeader,
}

fn sniff(image: *const c_void) -> Result<BinaryKind, CUresult> {
    if image.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    match unsafe { *image.cast::<[u8; 4]>() } {
        FATBINC_MAGIC => Ok(BinaryKind::FatbincWrapper),
        FATBIN_MAGIC => Ok(BinaryKind::FatbinHeader),
        _ => Ok(BinaryKind::Ptx),
    }
}

struct FatbinFileIter<'a> {
    buffer: &'a [u8],
}

impl<'a> FatbinFileIter<'a> {
    unsafe fn new(header: &'a FatbinHeader) -> Self {
        let start = (header as *const FatbinHeader as *const u8)
            .add(header.header_size as usize);
        let buffer = unsafe { slice::from_raw_parts(start, header.files_size as usize) };
        FatbinFileIter { buffer }
    }
}

impl<'a> Iterator for FatbinFileIter<'a> {
    type Item = (FatbinFileHeader, *const u8);
    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.len() < FATBIN_FILE_HEADER_FIXED_SIZE {
            return None;
        }
        let base = self.buffer.as_ptr();
        let this = unsafe { FatbinFileHeader::parse(base)? };
        if (this.header_size as usize) < FATBIN_FILE_HEADER_FIXED_SIZE {
            return None;
        }
        let step = this.header_size as usize
            + cmp::max(this.payload_size, this.compressed_size) as usize;
        self.buffer = &self.buffer[cmp::min(step, self.buffer.len())..];
        Some((this, base))
    }
}

/// Visit every `FatbinHeader` frame reachable from `image`.
unsafe fn visit_submodules<'a>(
    image: *const c_void,
    mut visit: impl FnMut(&'a FatbinHeader) -> Result<(), CUresult>,
) -> Result<(), CUresult> {
    match unsafe { sniff(image) }? {
        BinaryKind::Ptx => Ok(()),
        BinaryKind::FatbinHeader => {
            let header = unsafe { &*image.cast::<FatbinHeader>() };
            visit(header)
        }
        BinaryKind::FatbincWrapper => {
            // The image can be a single wrapper or an array of them: a
            // program linked from many translation units has one wrapper per
            // unit in the .nvFatBinSegment section, and the runtime hands the
            // whole array to the driver. The kernels of the different units
            // then all live in one module.
            let mut offset = 0usize;
            loop {
                let wrapper = unsafe { &*image.cast::<FatbincWrapper>().add(offset) };
                if wrapper.magic != u32::from_le_bytes(FATBINC_MAGIC)
                    || (wrapper.version != FATBINC_VERSION_V1
                        && wrapper.version != FATBINC_VERSION_V2)
                {
                    if offset == 0 {
                        return Err(CUresult::CUDA_ERROR_INVALID_PTX);
                    }
                    break;
                }
                match wrapper.version {
                    FATBINC_VERSION_V1 => {
                        if wrapper.data.is_null() {
                            return Err(CUresult::CUDA_ERROR_INVALID_PTX);
                        }
                        visit(unsafe { &*wrapper.data })?;
                    }
                    _ => {
                        let mut p = wrapper.filename_or_fatbins as *const *const FatbinHeader;
                        while !p.is_null() && !unsafe { *p }.is_null() {
                            visit(unsafe { &**p })?;
                            p = unsafe { p.add(1) };
                        }
                    }
                }
                // Is the next slot another wrapper?
                let next = unsafe { &*image.cast::<FatbincWrapper>().add(offset + 1) };
                if next.magic != u32::from_le_bytes(FATBINC_MAGIC) {
                    break;
                }
                offset += 1;
                if let Ok(limit) = std::env::var("ZLUDA_FATBIN_LIMIT") {
                    if limit.parse::<usize>().map(|l| offset >= l).unwrap_or(false) {
                        break;
                    }
                }
            }
            Ok(())
        }
    }
}

fn file_score(file: &FatbinFileHeader) -> i32 {
    let mut score = 0;
    if file.flags & FATBIN_FILE_FLAG_64BIT != 0 {
        score += 4;
    }
    if file.flags & (FATBIN_FILE_FLAG_COMPRESSED_LZ4 | FATBIN_FILE_FLAG_COMPRESSED_ZSTD) == 0 {
        score += 2;
    }
    if file.flags & FATBIN_FILE_FLAG_COMPRESSED_LZ4 != 0 {
        score += 1;
    }
    score
}

fn decompress(file: &FatbinFileHeader, base: *const u8) -> Result<Vec<u8>, CUresult> {
    let start = unsafe { base.add(file.header_size as usize) };
    if file.flags & FATBIN_FILE_FLAG_COMPRESSED_LZ4 != 0 {
        let compressed = unsafe { slice::from_raw_parts(start, file.compressed_size as usize) };
        let mut out = vec![0u8; cmp::max(1024, file.uncompressed_payload as usize)];
        loop {
            let written = unsafe {
                lz4_sys::LZ4_decompress_safe(
                    compressed.as_ptr() as *const _,
                    out.as_mut_ptr() as *mut _,
                    compressed.len() as _,
                    out.len() as _,
                )
            };
            if written < 0 {
                let new_len = out.len().saturating_mul(2);
                if new_len > MAX_DECOMPRESSION_BOUND || new_len == out.len() {
                    return Err(CUresult::CUDA_ERROR_INVALID_PTX);
                }
                out.resize(new_len, 0);
            } else {
                out.truncate(written as usize);
                return Ok(out);
            }
        }
    } else {
        Ok(unsafe { slice::from_raw_parts(start, file.payload_size as usize) }.to_vec())
    }
}

/// Extract PTX source from a fatbin (or pass through plain PTX text).
pub unsafe fn extract_ptx(image: *const c_void) -> Result<Vec<u8>, CUresult> {
    match sniff(image)? {
        BinaryKind::Ptx => {
            // Plain PTX text: read it as a NUL-terminated string.
            let text = unsafe { CStr::from_ptr(image as *const i8) };
            Ok(text.to_bytes().to_vec())
        }
        _ => {
            let mut best: Option<(i32, Vec<u8>)> = None;
            // Track whether the fatbin carried any non-PTX image. A fatbin
            // with only CUBIN/ELF records holds NVIDIA SASS, which cannot be
            // translated to SPIR-V; report that distinctly so the user learns
            // the image has to be rebuilt with PTX instead of getting a
            // generic "invalid PTX".
            let mut saw_non_ptx = false;
            unsafe {
                visit_submodules(image, |header| {
                    if header.magic != u32::from_le_bytes(FATBIN_MAGIC)
                        || header.version != FATBIN_VERSION
                    {
                        if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
                            eprintln!(
                                "fatbin header invalid: magic={:#x} version={}",
                                header.magic, header.version
                            );
                        }
                        return Err(CUresult::CUDA_ERROR_INVALID_PTX);
                    }
                    for (file, base) in FatbinFileIter::new(header) {
                        if file.kind != FATBIN_FILE_KIND_PTX {
                            saw_non_ptx = true;
                            continue;
                        }
                        // Zstd compression is not supported yet.
                        if file.flags & FATBIN_FILE_FLAG_COMPRESSED_ZSTD != 0 {
                            continue;
                        }
                        let score = file_score(&file);
                        if best.as_ref().map(|(s, _)| score <= *s).unwrap_or(false) {
                            continue;
                        }
                        let mut payload = decompress(&file, base)?;
                        while payload.last() == Some(&0) {
                            payload.pop();
                        }
                        best = Some((score, payload));
                    }
                    Ok(())
                })?;
            }
            match best {
                Some((_, payload)) => Ok(payload),
                None if saw_non_ptx => {
                    // The fatbin only carried compiled NVIDIA code (cubin/ELF).
                    // ZLUDA translates PTX only, so the image has to be rebuilt
                    // with PTX embedded (for nvcc: -gencode arch=compute_XX,code=compute_XX).
                    Err(CUresult::CUDA_ERROR_NOT_SUPPORTED)
                }
                None => Err(CUresult::CUDA_ERROR_INVALID_PTX),
            }
        }
    }
}

/// Extract every PTX module the image carries.
///
/// A fatbin can hold many PTX records - one per translation unit, each
/// with its own kernels. The CUDA runtime JITs all of them into a single
/// module, so a driver that wants to be a drop-in replacement has to do
/// the same: every record is translated and the results are linked
/// together (see `SpirvModule::compile`).
pub unsafe fn extract_ptx_all(image: *const c_void) -> Result<Vec<Vec<u8>>, CUresult> {
    match unsafe { sniff(image)? } {
        BinaryKind::Ptx => {
            let text = unsafe { CStr::from_ptr(image as *const i8) };
            Ok(vec![text.to_bytes().to_vec()])
        }
        _ => {
            let mut modules: Vec<Vec<u8>> = Vec::new();
            let mut saw_non_ptx = false;
            unsafe {
                visit_submodules(image, |header| {
                    if header.magic != u32::from_le_bytes(FATBIN_MAGIC)
                        || header.version != FATBIN_VERSION
                    {
                        if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
                            eprintln!(
                                "fatbin header invalid: magic={:#x} version={}",
                                header.magic, header.version
                            );
                        }
                        return Err(CUresult::CUDA_ERROR_INVALID_PTX);
                    }
                    for (file, base) in FatbinFileIter::new(header) {
                        if file.kind != FATBIN_FILE_KIND_PTX {
                            saw_non_ptx = true;
                            continue;
                        }
                        // Zstd compression is not supported yet.
                        if file.flags & FATBIN_FILE_FLAG_COMPRESSED_ZSTD != 0 {
                            continue;
                        }
                        let mut payload = decompress(&file, base)?;
                        while payload.last() == Some(&0) {
                            payload.pop();
                        }
                        modules.push(payload);
                    }
                    Ok(())
                })?;
            }
            if modules.is_empty() {
                if saw_non_ptx {
                    // The fatbin only carried compiled NVIDIA code
                    // (cubin/ELF). ZLUDA translates PTX only, so the image
                    // has to be rebuilt with PTX embedded (for nvcc:
                    // -gencode arch=compute_XX,code=compute_XX).
                    return Err(CUresult::CUDA_ERROR_NOT_SUPPORTED);
                }
                return Err(CUresult::CUDA_ERROR_INVALID_PTX);
            }
            Ok(modules)
        }
    }
}
