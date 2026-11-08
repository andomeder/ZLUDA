//! Parses CUDA fatbinary containers (fatbin) and extracts what they carry.
//!
//! The image is either a `FatbincWrapper` (magic `FbC\xB1`) wrapping one or
//! more `FatbinHeader` frames, a bare `FatbinHeader` (magic `P\xEDU\xBA`), a
//! bare ELF cubin (magic `\x7fELF`), or plain PTX text. PTX records are
//! consumed as they are (optionally LZ4-compressed); CUBIN/ELF records are
//! NVIDIA SASS, which the translator cannot read. An image that carries no PTX
//! at all is refused unless the SASS lift is enabled (`super::sass`), in which
//! case its cubins are handed to the loader and lifted to PTX when one of
//! their kernels is first used.
//!
//! A bare cubin is the shape the CUDA runtime uses for a module it built at run
//! time (torch's jiterator ops, for one), and reading it as PTX text finds no
//! `.entry` at all, so it has to be sniffed rather than guessed.

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
/// Fields are decoded at their byte offsets rather than cast to this struct:
/// the header is variable-sized (`header_size` is 0x40 for plain records and
/// 0x48/0x60 for records that carry a kernel name inline), so a struct cast
/// would read past the end of a 0x40-byte header.
///
/// `uncompressed_payload` at 0x38 is a u32, not a u64: reading it as u64 picks
/// up the following zero word and yields a nonsense size, which makes the LZ4
/// decompression fail.
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
    Elf,
    FatbincWrapper,
    FatbinHeader,
}

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];

fn sniff(image: *const c_void) -> Result<BinaryKind, CUresult> {
    if image.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    match unsafe { *image.cast::<[u8; 4]>() } {
        FATBINC_MAGIC => Ok(BinaryKind::FatbincWrapper),
        FATBIN_MAGIC => Ok(BinaryKind::FatbinHeader),
        ELF_MAGIC => Ok(BinaryKind::Elf),
        _ => Ok(BinaryKind::Ptx),
    }
}

/// Size of a bare ELF image, from its own headers.
///
/// `cuModuleLoadData` takes a pointer with no length: PTX text is read as a C
/// string and a fatbin describes its own extent, but a cubin has neither, so
/// the size is the end of its section and program header tables. A cubin whose
/// headers do not describe it is not one, and is refused rather than read past
/// whatever the caller allocated.
fn elf_len(image: *const c_void) -> Option<usize> {
    unsafe {
        let head = slice::from_raw_parts(image as *const u8, 0x40);
        if head[0..4] != ELF_MAGIC {
            return None;
        }
        let u16_at = |off: usize| u16::from_le_bytes([head[off], head[off + 1]]) as usize;
        let u64_at = |off: usize| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&head[off..off + 8]);
            u64::from_le_bytes(bytes) as usize
        };
        let mut end = 0x40;
        let phoff = u64_at(0x20);
        let phentsize = u16_at(0x36);
        let phnum = u16_at(0x38);
        if phoff != 0 && phentsize != 0 {
            end = end.max(phoff.checked_add(phnum.checked_mul(phentsize)?)?);
        }
        let shoff = u64_at(0x28);
        let shentsize = u16_at(0x3a);
        let shnum = u16_at(0x3c);
        if shoff != 0 && shentsize != 0 {
            end = end.max(shoff.checked_add(shnum.checked_mul(shentsize)?)?);
        }
        Some(end)
    }
}

/// A bare cubin, as the one record of an image.
fn elf_module(image: *const c_void) -> Result<ModuleSource, CUresult> {
    let len = elf_len(image).ok_or(CUresult::CUDA_ERROR_INVALID_IMAGE)?;
    if !super::sass::enabled() {
        eprintln!(
            "zluda: the image is a cubin ({} bytes), which is NVIDIA SASS. Rebuild it with PTX \
             embedded (for nvcc: -gencode arch=compute_XX,code=compute_XX), or set \
             ZLUDA_SASS_LIFT=1 to translate it",
            len
        );
        return Err(CUresult::CUDA_ERROR_NOT_SUPPORTED);
    }
    let cubin = unsafe { slice::from_raw_parts(image as *const u8, len) }.to_vec();
    Ok(ModuleSource::Cubin(cubin))
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
        BinaryKind::Ptx | BinaryKind::Elf => Ok(()),
        BinaryKind::FatbinHeader => {
            let header = unsafe { &*image.cast::<FatbinHeader>() };
            visit(header)
        }
        BinaryKind::FatbincWrapper => {
            // The image can be a single wrapper or an array of them: a program
            // linked from many translation units carries one wrapper per unit
            // in the .nvFatBinSegment section, and the runtime hands the whole
            // array to the driver.
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

/// One module of an image: PTX text, or the compiled code a fatbin carries
/// when it was built for one architecture only.
pub enum ModuleSource {
    Ptx(Vec<u8>),
    /// An ELF/cubin image (NVIDIA SASS), lifted to PTX when a kernel of it is
    /// first used. Only produced when the SASS lift is enabled.
    Cubin(Vec<u8>),
}

/// Extract PTX source from a fatbin (or pass through plain PTX text).
pub unsafe fn extract_ptx(image: *const c_void) -> Result<Vec<u8>, CUresult> {
    match sniff(image)? {
        BinaryKind::Ptx => {
            let text = unsafe { CStr::from_ptr(image as *const i8) };
            Ok(text.to_bytes().to_vec())
        }
        // A bare cubin is SASS: there is no PTX to return.
        BinaryKind::Elf => Err(CUresult::CUDA_ERROR_NOT_SUPPORTED),
        _ => {
            let mut best: Option<(i32, Vec<u8>)> = None;
            // Whether the fatbin carried any non-PTX image: a SASS-only one is
            // reported distinctly from broken PTX.
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
                    // SASS only: the image has to be rebuilt with PTX
                    // embedded (nvcc: -gencode arch=compute_XX,code=compute_XX).
                    Err(CUresult::CUDA_ERROR_NOT_SUPPORTED)
                }
                None => Err(CUresult::CUDA_ERROR_INVALID_PTX),
            }
        }
    }
}

/// Extract every module the image carries, in image order.
///
/// The newest architecture the lifter implements.
const NEWEST_LIFTABLE_ARCH: u32 = 86;

/// An image carries one ELF record per architecture, oldest first (torch starts
/// at sm_50), so the records have to be narrowed to the one worth lifting.
fn newest_liftable(cubins: Vec<ModuleSource>) -> Vec<ModuleSource> {
    let arch_of = |source: &ModuleSource| match source {
        ModuleSource::Cubin(image) => super::sass::cubin_arch(image).unwrap_or(0),
        ModuleSource::Ptx(_) => 0,
    };
    let newest = cubins
        .iter()
        .map(arch_of)
        .filter(|arch| *arch != 0 && *arch <= NEWEST_LIFTABLE_ARCH)
        .max();
    match newest {
        Some(newest) => cubins
            .into_iter()
            .filter(|source| arch_of(source) == newest)
            .collect(),
        None => cubins,
    }
}

/// A fatbin holds one record per translation unit (and one per architecture),
/// each with its own kernels. A PTX record is returned as text; a compiled
/// (ELF/cubin) record is returned only when the image carries no PTX at all
/// and the SASS lift is enabled, so an image that can be served by its own PTX
/// always is.
pub unsafe fn extract_modules(image: *const c_void) -> Result<Vec<ModuleSource>, CUresult> {
    match sniff(image)? {
        BinaryKind::Ptx => {
            let text = unsafe { CStr::from_ptr(image as *const i8) };
            Ok(vec![ModuleSource::Ptx(text.to_bytes().to_vec())])
        }
        BinaryKind::Elf => Ok(vec![unsafe { elf_module(image) }?]),
        _ => {
            let mut modules: Vec<ModuleSource> = Vec::new();
            let mut cubins: Vec<ModuleSource> = Vec::new();
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
                        if file.kind == FATBIN_FILE_KIND_ELF {
                            saw_non_ptx = true;
                            if !super::sass::enabled() {
                                continue;
                            }
                            // Zstd compression is not supported yet.
                            if file.flags & FATBIN_FILE_FLAG_COMPRESSED_ZSTD != 0 {
                                continue;
                            }
                            let cubin = decompress(&file, base)?;
                            if cubin.len() < 0x40 || &cubin[0..4] != b"\x7fELF" {
                                eprintln!(
                                    "zluda: sass lift: an ELF record of the image is not an ELF \
                                     image ({} bytes, magic {:02x?})",
                                    cubin.len(),
                                    &cubin[..cubin.len().min(4)]
                                );
                                continue;
                            }
                            cubins.push(ModuleSource::Cubin(cubin));
                            continue;
                        }
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
                        modules.push(ModuleSource::Ptx(payload));
                    }
                    Ok(())
                })?;
            }
            if modules.is_empty() {
                if !cubins.is_empty() {
                    return Ok(newest_liftable(cubins));
                }
                if saw_non_ptx {
                    // SASS only: rebuild the image with PTX embedded, or
                    // enable the lift (ZLUDA_SASS_LIFT=1) to translate it.
                    return Err(CUresult::CUDA_ERROR_NOT_SUPPORTED);
                }
                return Err(CUresult::CUDA_ERROR_INVALID_PTX);
            }
            Ok(modules)
        }
    }
}
