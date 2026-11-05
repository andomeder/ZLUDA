//! On-the-fly SASS lift for fatbin records that carry only compiled code.
//!
//! A fatbin built with `-gencode arch=compute_XX,code=sm_XX` carries NVIDIA
//! SASS (kind=2 ELF records) and no PTX at all, and our translator reads PTX.
//! Such an image can still be loaded by lifting the cubin back to PTX first:
//!
//!     cubin -> CuLifter (nvvm backend) -> LLVM IR -> llc -> PTX -> translator
//!
//! The lifter (gthparch/CuLifter) ships no licence, so this path is off by
//! default and has to be enabled explicitly:
//!
//!     ZLUDA_SASS_LIFT      "1" enables the lift
//!     ZLUDA_SASS_LIFTER    CuLifter checkout (default /tmp/opencode/sasslifter/CuLifter)
//!     ZLUDA_SASS_CACHE     cache root (default ~/.cache/zluda/sass)
//!     ZLUDA_SASS_CUOBJDUMP cuobjdump to disassemble with (default: PATH)
//!     ZLUDA_SASS_PYTHON    python for the lifter (default: the venv next to it)
//!     ZLUDA_SASS_LLC       llc to use (default: PATH)
//!     ZLUDA_SASS_REQUIRE_FULL   "1" fails the load on an instruction the lifter
//!                          could only placeholder, instead of warning
//!
//! The result is cached per cubin (`<cache>/<sha256 of the cubin>/`), so a
//! second load of the same image is a file read.
//!
//! A kernel the lifter cannot handle fails with the stage that failed and the
//! kernel name, and the load fails with it: a module is never built with work
//! silently missing.

use super::CUresult;
use std::{
    convert::TryInto,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

/// Architectures the lifter implements. CuLifter models sm_70/75/80/86; the
/// sm_90 path (wgmma descriptors, mbarrier) is a different project.
const LIFTABLE_ARCHS: [u32; 4] = [70, 75, 80, 86];

const DEFAULT_LIFTER: &str = "/tmp/opencode/sasslifter/CuLifter";

/// Whether the SASS lift is enabled.
pub fn enabled() -> bool {
    match std::env::var("ZLUDA_SASS_LIFT") {
        Ok(value) => value == "1" || value.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

/// Instructions the lifter had to placeholder are emitted as a stub named
/// `sass_unsupported_<instruction>`, and the instruction's effect is absent from
/// the lifted text. Some of those are on unreachable paths (torch's `mean`
/// reductions carry one and compute correctly), and some are load bearing
/// (`ubmsk` feeds a lane mask in CUB's radix sort, which then computes garbage),
/// so the default is to say which instruction it was and let the load proceed:
/// refusing outright would break kernels that work. `ZLUDA_SASS_REQUIRE_FULL=1`
/// makes it fatal for a run that must not tolerate a partial translation.
const PLACEHOLDER: &str = "sass_unsupported_";

fn placeholders(ptx: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for line in ptx.lines() {
        if let Some(at) = line.find(PLACEHOLDER) {
            let rest = &line[at + PLACEHOLDER.len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() && !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

fn require_full() -> bool {
    match std::env::var("ZLUDA_SASS_REQUIRE_FULL") {
        Ok(value) => value == "1" || value.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

/// Reports, and optionally refuses, a lift that could not translate everything.
fn check_complete(ptx: &str, kernel: &str) -> Result<(), CUresult> {
    let names = placeholders(ptx);
    if names.is_empty() {
        return Ok(());
    }
    let list = names.join(", ");
    if require_full() {
        return Err(fail(
            "lift",
            kernel,
            &format!(
                "the lifter could not translate: {} (it substitutes a stub for an \
                 instruction it does not model), so the kernel would compute without \
                 that instruction",
                list
            ),
        ));
    }
    eprintln!(
        "zluda: sass lift: the lifter could not translate {} in {}; it substitutes a stub, \
         so the kernel runs without that instruction and may compute a wrong value \
         (set ZLUDA_SASS_REQUIRE_FULL=1 to turn this into a load failure)",
        list, kernel
    );
    Ok(())
}

/// The compute capability of a cubin, read from its ELF header.
///
/// CUDA 12 and earlier put the SM version in the low byte of `e_flags`
/// (sm_75 is 0x4b054b) and CUDA 13 moves it up one byte (the same sm_75 cubin
/// says 0x6004b04), so the candidate is searched for rather than assumed.
pub fn cubin_arch(cubin: &[u8]) -> Result<u32, CUresult> {
    let image = match elf_image(cubin) {
        Some(image) => image,
        None => {
            eprintln!("zluda: sass lift: the record is not a little-endian 64-bit ELF image");
            return Err(CUresult::CUDA_ERROR_INVALID_IMAGE);
        }
    };
    let flags = u32::from_le_bytes(image[0x30..0x34].try_into().unwrap());
    match arch_from_flags(flags) {
        Some(arch) => Ok(arch),
        None => {
            eprintln!(
                "zluda: sass lift: the cubin does not name a known architecture (e_flags={:#x})",
                flags
            );
            Err(CUresult::CUDA_ERROR_INVALID_IMAGE)
        }
    }
}

/// The architectures the CUDA toolchain has shipped, which is what makes a
/// byte of `e_flags` recognisable as the SM version.
const KNOWN_ARCHS: [u32; 15] = [
    50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 88, 89, 90,
];

/// Which byte of `e_flags` holds the SM version.
///
/// Measured on real cubins: CUDA 12 and earlier put it in the first byte and
/// repeat it in the third (`sm_75` is 0x4b054b, `sm_80` 0x500550, `sm_90`
/// 0x5a055a); CUDA 13 moved it to the second byte (`sm_75` built with nvcc
/// 13.4 is 0x06004b04). Only these two positions are considered: the third
/// byte is a copy of the first, so reading it would turn an image with an
/// unknown architecture into a plausible wrong answer.
fn arch_from_flags(flags: u32) -> Option<u32> {
    for shift in [0, 8] {
        let candidate = (flags >> shift) & 0xff;
        if KNOWN_ARCHS.contains(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// The kernels a cubin defines, from its ELF symbol table.
///
/// The loader registers these with the record that holds them, so a kernel is
/// found and lifted when it is first asked for, without disassembling or
/// lifting the whole record.
pub fn kernel_names(cubin: &[u8]) -> Result<Vec<String>, CUresult> {
    let image = match elf_image(cubin) {
        Some(image) => image,
        None => return Err(CUresult::CUDA_ERROR_INVALID_IMAGE),
    };
    let u16_at = |off: usize| u16::from_le_bytes(image[off..off + 2].try_into().unwrap()) as usize;
    let u64_at = |off: usize| u64::from_le_bytes(image[off..off + 8].try_into().unwrap()) as usize;
    let shoff = u64_at(0x28);
    let shentsize = u16_at(0x3a);
    let shnum = u16_at(0x3c);
    if shoff == 0 || shentsize < 0x40 {
        return Err(CUresult::CUDA_ERROR_INVALID_IMAGE);
    }
    let section = |index: usize| -> Option<&[u8]> {
        let start = shoff.checked_add(index * shentsize)?;
        image.get(start..start + shentsize)
    };
    let section_u32 = |section: &[u8], off: usize| {
        u32::from_le_bytes(section[off..off + 4].try_into().unwrap()) as usize
    };
    let section_u64 = |section: &[u8], off: usize| {
        u64::from_le_bytes(section[off..off + 8].try_into().unwrap()) as usize
    };
    let mut names = Vec::new();
    for index in 0..shnum {
        let symtab = match section(index) {
            Some(section) => section,
            None => continue,
        };
        // SHT_SYMTAB
        if section_u32(symtab, 0x04) != 2 {
            continue;
        }
        let offset = section_u64(symtab, 0x18);
        let size = section_u64(symtab, 0x20);
        let strtab_index = section_u32(symtab, 0x28);
        let strtab = match section(strtab_index) {
            Some(section) => section,
            None => continue,
        };
        let strtab_offset = section_u64(strtab, 0x18);
        let strtab_size = section_u64(strtab, 0x20);
        let strings = match image.get(strtab_offset..strtab_offset + strtab_size) {
            Some(strings) => strings,
            None => continue,
        };
        let table = match image.get(offset..offset + size) {
            Some(table) => table,
            None => continue,
        };
        for symbol in table.chunks_exact(0x18) {
            let name_offset = u32::from_le_bytes(symbol[0x00..0x04].try_into().unwrap()) as usize;
            let info = symbol[0x04];
            let section_index = u16::from_le_bytes(symbol[0x06..0x08].try_into().unwrap());
            // STT_FUNC (the low nibble of st_info is the type, the high one the
            // binding), and defined (SHN_UNDEF is a symbol the module imports)
            if info & 0xf != 2 || section_index == 0 || name_offset >= strings.len() {
                continue;
            }
            let name = &strings[name_offset..];
            let name = match name.iter().position(|b| *b == 0) {
                Some(end) => &name[..end],
                None => name,
            };
            if let Ok(name) = std::str::from_utf8(name) {
                names.push(name.to_string());
            }
        }
    }
    Ok(names)
}

/// The cubin, when it is a little-endian 64-bit ELF image.
fn elf_image(cubin: &[u8]) -> Option<&[u8]> {
    if cubin.len() < 0x40 || &cubin[0..4] != b"\x7fELF" {
        return None;
    }
    // EI_CLASS 2 = 64-bit, EI_DATA 1 = little endian
    if cubin[4] != 2 || cubin[5] != 1 {
        return None;
    }
    Some(cubin)
}

/// Lift one kernel of a cubin to PTX, from the on-disk cache when it is there.
pub fn lift_kernel(cubin: &[u8], kernel: &str) -> Result<String, CUresult> {
    let arch = cubin_arch(cubin)?;
    if !LIFTABLE_ARCHS.contains(&arch) {
        return Err(fail(
            "lift",
            kernel,
            &format!(
                "sm_{} is not a liftable architecture (the lifter implements sm_70/75/80/86)",
                arch
            ),
        ));
    }
    let hash = hex(&sha256(cubin));
    let path = cache_path(&hash, kernel)?;
    let started = Instant::now();
    if let Ok(recovered) = fs::read_to_string(&path) {
        let recovered = strip_cache_header(&recovered);
        // A cache entry that was interrupted while it was written, or written
        // by a different lifter, is not trusted: it has to define the kernel.
        if has_entry(&recovered, kernel) {
            check_complete(&recovered, kernel)?;
            eprintln!(
                "zluda: sass lift: {:.3} ms (cache hit) {}",
                started.elapsed().as_secs_f64() * 1e3,
                kernel
            );
            return Ok(recovered);
        }
    }
    let started = Instant::now();
    let ptx = lift_uncached(cubin, kernel, arch, &hash)?;
    check_complete(&ptx, kernel)?;
    let elapsed = started.elapsed().as_secs_f64() * 1e3;
    // The cache is an optimisation: a directory that cannot be written is not
    // a reason to fail the load.
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = fs::File::create(&path) {
        let _ = file.write_all(cache_header(kernel, &hash).as_bytes());
        let _ = file.write_all(ptx.as_bytes());
    }
    eprintln!(
        "zluda: sass lift: {:.0} ms (lifted, cached in {}) {}",
        elapsed,
        path.display(),
        kernel
    );
    Ok(ptx)
}

fn lift_uncached(cubin: &[u8], kernel: &str, arch: u32, hash: &str) -> Result<String, CUresult> {
    let lifter = lifter_dir();
    if !lifter.join("main.py").is_file() {
        return Err(fail(
            "lift",
            kernel,
            &format!(
                "the lifter checkout is not at {} (set ZLUDA_SASS_LIFTER)",
                lifter.display()
            ),
        ));
    }
    let work = work_dir(hash)?;
    let result = lift_in_dir(cubin, kernel, arch, &work, &lifter);
    let _ = fs::remove_dir_all(&work);
    result
}

fn lift_in_dir(
    cubin: &[u8],
    kernel: &str,
    arch: u32,
    work: &Path,
    lifter: &Path,
) -> Result<String, CUresult> {
    let cubin_path = work.join("record.cubin");
    let sass_path = work.join("kernel.sass");
    // The lifter appends ".ll" to the name it is given.
    let ir_name = work.join("kernel");
    let ir_path = work.join("kernel.ll");
    let ptx_path = work.join("kernel.ptx");
    fs::write(&cubin_path, cubin)
        .map_err(|e| fail("lift", kernel, &format!("cannot write the cubin out: {}", e)))?;

    // The disassembly is what the lifter parses; the cubin next to it is where
    // it reads the constant bank and the descriptor data from.
    let disassembly = run(
        Command::new(tool("ZLUDA_SASS_CUOBJDUMP", "cuobjdump")?)
            .arg("-sass")
            .arg("-fun")
            .arg(kernel)
            .arg(&cubin_path),
    )
    .map_err(|e| fail("lift", kernel, &format!("cuobjdump: {}", e)))?;
    fs::write(&sass_path, &disassembly)
        .map_err(|e| fail("lift", kernel, &format!("cannot write the disassembly: {}", e)))?;
    if !String::from_utf8_lossy(&disassembly).contains("Function :") {
        return Err(fail(
            "lift",
            kernel,
            "cuobjdump produced no disassembly for it (it is not a function of this image)",
        ));
    }

    let lifter_arch = arch.to_string();
    run(Command::new(python(lifter)?)
        .arg(lifter.join("main.py"))
        .arg("-i")
        .arg(&sass_path)
        .arg(&cubin_path)
        .arg("-o")
        .arg(&ir_name)
        .arg("-arch")
        .arg(&lifter_arch)
        .arg("--lifter")
        .arg("nvvm")
        .arg("-name")
        .arg(kernel))
    .map_err(|e| fail("lift", kernel, &format!("the lifter: {}", e)))?;
    if !ir_path.is_file() {
        return Err(fail("lift", kernel, "the lifter produced no IR"));
    }

    let ptx_version = if arch <= 75 { "ptx78" } else { "ptx80" };
    run(Command::new(tool("ZLUDA_SASS_LLC", "llc")?)
        .arg("-march=nvptx64")
        .arg(format!("-mcpu=sm_{}", arch))
        .arg(format!("-mattr=+{}", ptx_version))
        .arg("-O2")
        .arg("-o")
        .arg(&ptx_path)
        .arg(&ir_path))
    .map_err(|e| fail("llc", kernel, &format!("llc: {}", e)))?;
    let ptx = fs::read_to_string(&ptx_path)
        .map_err(|e| fail("llc", kernel, &format!("no PTX was written: {}", e)))?;
    if !has_entry(&ptx, kernel) {
        return Err(fail(
            "llc",
            kernel,
            "the PTX that came out does not define the kernel",
        ));
    }
    Ok(ptx)
}

/// Whether a PTX module defines a kernel.
pub fn has_entry(ptx: &str, kernel: &str) -> bool {
    let needle = [".entry ", ".visible .entry "];
    for prefix in needle {
        let mut rest = ptx;
        while let Some(pos) = rest.find(prefix) {
            let after = &rest[pos + prefix.len()..];
            let end = after
                .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$' || c == '.'))
                .unwrap_or(after.len());
            if &after[..end] == kernel {
                return true;
            }
            rest = &after[end..];
        }
    }
    false
}

fn tool(variable: &str, name: &str) -> Result<PathBuf, CUresult> {
    if let Ok(value) = std::env::var(variable) {
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
    if let Some(path) = search_path(name) {
        return Ok(path);
    }
    if name == "cuobjdump" {
        // The CUDA toolkit that ships inside the triton wheel is the only
        // cuobjdump an installation without a toolkit has.
        if let Some(path) = search_triton(name) {
            return Ok(path);
        }
    }
    Err(fail(
        "lift",
        "-",
        &format!(
            "{} was not found (set {}, or put it on PATH)",
            name, variable
        ),
    ))
}

fn search_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn search_triton(name: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    for root in ["lib", "lib64"] {
        let lib = Path::new(&home).join(root);
        let entries = fs::read_dir(&lib).ok()?;
        for entry in entries.flatten() {
            let candidate = entry
                .path()
                .join("site-packages/triton/backends/nvidia/bin")
                .join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn python(lifter: &Path) -> Result<PathBuf, CUresult> {
    if let Ok(value) = std::env::var("ZLUDA_SASS_PYTHON") {
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
    // The lifter needs llvmlite, which the system python does not have: prefer
    // the virtualenv kept next to the checkout.
    if let Some(parent) = lifter.parent() {
        for venv in ["culifter-venv", ".venv", "venv"] {
            let candidate = parent.join(venv).join("bin/python");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    search_path("python3")
        .ok_or_else(|| fail("lift", "-", "no python for the lifter (set ZLUDA_SASS_PYTHON)"))
}

fn lifter_dir() -> PathBuf {
    match std::env::var("ZLUDA_SASS_LIFTER") {
        Ok(value) if !value.is_empty() => PathBuf::from(value),
        _ => PathBuf::from(DEFAULT_LIFTER),
    }
}

fn cache_root() -> PathBuf {
    if let Ok(value) = std::env::var("ZLUDA_SASS_CACHE") {
        if !value.is_empty() {
            return PathBuf::from(value);
        }
    }
    match std::env::var_os("HOME") {
        Some(home) => Path::new(&home).join(".cache/zluda/sass"),
        None => std::env::temp_dir().join("zluda-sass"),
    }
}

fn cache_path(hash: &str, kernel: &str) -> Result<PathBuf, CUresult> {
    // A mangled kernel name is longer than a file name may be, and it can
    // contain characters a file name cannot: keep a readable prefix and make
    // the rest up from a hash of the name.
    let readable: String = kernel
        .chars()
        .take(64)
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let suffix = hex(&sha256(kernel.as_bytes()));
    let name = format!("{}-{}.ptx", readable, &suffix[..16]);
    Ok(cache_root().join(hash).join(name))
}

fn work_dir(hash: &str) -> Result<PathBuf, CUresult> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let unique = format!(
        ".work-{}-{}-{}",
        std::process::id(),
        sequence,
        &hash[..8]
    );
    let dir = cache_root().join(unique);
    let fallback = std::env::temp_dir().join(format!(
        "zluda-sass-{}-{}",
        std::process::id(),
        &hash[..8]
    ));
    let dir = if fs::create_dir_all(&dir).is_ok() {
        dir
    } else {
        fs::create_dir_all(&fallback)
            .map_err(|e| fail("lift", "-", &format!("no working directory: {}", e)))?;
        fallback
    };
    Ok(dir)
}

/// Run a command and return its stdout, or its error output as the message.
fn run(command: &mut Command) -> Result<Vec<u8>, String> {
    let output = command
        .output()
        .map_err(|e| format!("cannot run {:?}: {}", command.get_program(), e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail: String = stderr
            .lines()
            .chain(stdout.lines())
            .filter(|line| !line.trim().is_empty())
            .take(4)
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!(
            "exit {} - {}",
            output.status.code().unwrap_or(-1),
            detail
        ));
    }
    Ok(output.stdout)
}

fn cache_header(kernel: &str, hash: &str) -> String {
    format!("// zluda sass lift: {} from cubin {}\n", kernel, hash)
}

fn strip_cache_header(text: &str) -> String {
    match text.find('\n') {
        Some(end) if text.starts_with("// zluda sass lift: ") => text[end + 1..].to_string(),
        _ => text.to_string(),
    }
}

fn fail(stage: &str, kernel: &str, detail: &str) -> CUresult {
    if kernel == "-" {
        eprintln!("zluda: sass {} failed: {}", stage, detail);
    } else {
        eprintln!(
            "zluda: sass {} failed for kernel {:?}: {}",
            stage, kernel, detail
        );
    }
    CUresult::CUDA_ERROR_NOT_SUPPORTED
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

/// SHA-256, so the cache is keyed by the cubin itself and not by a path (the
/// same image is loaded from a file, from a fatbin array and from a mapped
/// object).
fn sha256(message: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = message.to_vec();
    let bit_len = (message.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = state;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let temp1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let temp2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(temp1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = temp1.wrapping_add(temp2);
        }
        for i in 0..8 {
            state[i] = state[i].wrapping_add(v[i]);
        }
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_of_a_cuda_12_cubin() {
        // as libtorch_cuda.so ships them
        assert_eq!(arch_from_flags(0x4b054b), Some(75));
        assert_eq!(arch_from_flags(0x500550), Some(80));
        assert_eq!(arch_from_flags(0x560556), Some(86));
        assert_eq!(arch_from_flags(0x5a055a), Some(90));
    }

    #[test]
    fn architecture_of_a_cuda_13_cubin() {
        // the same sm_75, compiled with nvcc 13.4: the SM byte moved up
        assert_eq!(arch_from_flags(0x6004b04), Some(75));
    }

    #[test]
    fn architecture_of_an_unknown_cubin() {
        assert_eq!(arch_from_flags(0), None);
        assert_eq!(arch_from_flags(0xffff_ffff), None);
        // the third byte is a copy of the first: an image that names no known
        // architecture must not be read as though it did
        assert_eq!(arch_from_flags(0x4b0590), None);
    }

    #[test]
    fn sha256_matches_the_published_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_kernel_is_found_in_ptx() {
        let text = ".version 7.8\n.target sm_75\n.visible .entry foo(\n)\n{\n ret;\n}\n";
        assert!(has_entry(text, "foo"));
        assert!(!has_entry(text, "fo"));
        assert!(!has_entry(text, "bar"));
        assert!(has_entry(".entry bar() { ret; }", "bar"));
    }

    #[test]
    fn a_kernel_is_found_in_a_cubin() {
        // A cubin whose symbol table has one defined function and one import.
        let cubin = build_cubin();
        assert_eq!(cubin_arch(&cubin), Ok(75));
        assert_eq!(kernel_names(&cubin).unwrap(), vec!["the_kernel".to_string()]);
    }

    /// A minimal little-endian 64-bit ELF: a section header table with one
    /// .symtab, its .strtab, and a .text for the defined symbol to point at.
    fn build_cubin() -> Vec<u8> {
        const TEXT: usize = 0x100;
        const STRTAB: usize = 0x200;
        const SYMTAB: usize = 0x300;
        const SECTIONS: usize = 0x400;
        let strings = b"\0the_kernel\0__assertfail\0";
        let mut image = vec![0u8; SECTIONS + 3 * 0x40];
        image[0..4].copy_from_slice(b"\x7fELF");
        image[4] = 2; // 64-bit
        image[5] = 1; // little endian
        image[0x28..0x30].copy_from_slice(&(SECTIONS as u64).to_le_bytes());
        image[0x30..0x34].copy_from_slice(&0x4b054bu32.to_le_bytes()); // sm_75
        image[0x3a..0x3c].copy_from_slice(&0x40u16.to_le_bytes());
        image[0x3c..0x3e].copy_from_slice(&3u16.to_le_bytes());
        image[STRTAB..STRTAB + strings.len()].copy_from_slice(strings);
        let mut symbols = [0u8; 3 * 0x18];
        // symbol 1: the kernel, defined in some section
        symbols[0x18..0x1c].copy_from_slice(&1u32.to_le_bytes());
        symbols[0x1c] = 0x12; // STB_GLOBAL | STT_FUNC
        symbols[0x1e..0x20].copy_from_slice(&3u16.to_le_bytes());
        // symbol 2: an import, which is not a kernel
        symbols[0x30..0x34].copy_from_slice(&12u32.to_le_bytes());
        symbols[0x34] = 0x12;
        symbols[0x36..0x38].copy_from_slice(&0u16.to_le_bytes());
        image[SYMTAB..SYMTAB + symbols.len()].copy_from_slice(&symbols);
        let mut symtab = [0u8; 0x40];
        symtab[0x04..0x08].copy_from_slice(&2u32.to_le_bytes()); // SHT_SYMTAB
        symtab[0x18..0x20].copy_from_slice(&(SYMTAB as u64).to_le_bytes());
        symtab[0x20..0x28].copy_from_slice(&(symbols.len() as u64).to_le_bytes());
        symtab[0x28..0x2c].copy_from_slice(&2u32.to_le_bytes()); // sh_link
        image[SECTIONS + 0x40..SECTIONS + 0x80].copy_from_slice(&symtab);
        let mut strtab = [0u8; 0x40];
        strtab[0x04..0x08].copy_from_slice(&3u32.to_le_bytes()); // SHT_STRTAB
        strtab[0x18..0x20].copy_from_slice(&(STRTAB as u64).to_le_bytes());
        strtab[0x20..0x28].copy_from_slice(&(strings.len() as u64).to_le_bytes());
        image[SECTIONS + 0x80..SECTIONS + 0xc0].copy_from_slice(&strtab);
        image[TEXT] = 0;
        image
    }
}
