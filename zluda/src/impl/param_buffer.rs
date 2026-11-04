//! Kernel arguments that do not fit in the parameter list.
//!
//! Intel's compiler caps the total size of a kernel's arguments at 2048 bytes:
//!
//!     error: Total size of kernel arguments exceeds limit!
//!            Total arguments size: 3440, limit: 2048
//!
//! A CUDA kernel is allowed far more than that. `CatArrayBatchedCopy`, which is
//! what `torch.cat` runs, takes a 3.4 KB block of per-input metadata by value;
//! PyTorch's kernels are written to a 4 KB limit and do not check for a smaller
//! one. So the kernel is rewritten here instead: it takes one eight byte value,
//! the address of a buffer, and every argument it used to receive in the
//! parameter list is read from that buffer instead. The driver fills the buffer
//! at launch time.
//!
//! This is what CUDA itself does with a kernel's parameter block; the code is
//! the same in spirit, only the layout is ours.

use super::CUresult;
use std::ffi::c_void;

/// The limit IGC enforces on the total size of a kernel's arguments.
pub const MAX_ARG_BYTES: usize = 2048;

/// The name of the rewritten entry point's only parameter.
const BUFFER_PARAM: &str = "__zluda_param_buffer";

/// Where each argument lives for a kernel that reads them from a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamBuffer {
    /// The size of each argument, in the order the application passes them.
    pub sizes: Vec<usize>,
    /// The byte offset of each argument inside the buffer.
    pub offsets: Vec<usize>,
    /// The size of the whole buffer.
    pub total: usize,
}

/// Fill a buffer for a launch and return its device address.
///
/// The arguments come from the application exactly as they would for the
/// parameter list: either a pointer per argument, or one packed block (the
/// `extra` form of `cuLaunchKernel`).
///
/// The buffer is device memory, and it has to stay alive until the kernel that
/// reads it has retired. Both calls here are blocking (the copy drains the
/// context's streams), so by the time this returns every earlier kernel has
/// finished and the buffers published before it can be released; this one is
/// released by the next publish, or at context teardown.
pub fn publish(
    layout: &ParamBuffer,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Result<*mut c_void, CUresult> {
    let mut block = vec![0u8; layout.total];
    if !kernel_params.is_null() {
        for (index, size) in layout.sizes.iter().enumerate() {
            let source = unsafe { *kernel_params.add(index) };
            if source.is_null() || *size == 0 {
                continue;
            }
            let bytes = unsafe { std::slice::from_raw_parts(source as *const u8, *size) };
            block[layout.offsets[index]..layout.offsets[index] + *size].copy_from_slice(bytes);
        }
    } else if !extra.is_null() {
        let (packed, packed_size) = unpacked_arguments(extra)?;
        // The application's own layout is the one the parameter list would have
        // had: each argument rounded up to its own size.
        let mut offset = 0usize;
        for (index, size) in layout.sizes.iter().enumerate() {
            let source_offset = round_up_to_multiple(offset, (*size).max(1));
            if source_offset + *size > packed_size {
                return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
            }
            let bytes =
                unsafe { std::slice::from_raw_parts(packed.add(source_offset), *size) };
            block[layout.offsets[index]..layout.offsets[index] + *size].copy_from_slice(bytes);
            offset = source_offset + *size;
        }
    } else {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let device = super::memory::alloc_v2_raw(layout.total)?;
    // The copy is blocking, so it also establishes that every earlier kernel
    // that read one of the buffers below has retired.
    super::memory::copy_v2(device, block.as_ptr() as *const c_void, layout.total)?;
    free_published();
    PUBLISHED
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .push(SendPtr(device));
    Ok(device)
}

/// The markers a packed launch uses, mirrored from `cuLaunchKernel`'s contract.
const LAUNCH_PARAM_END: *mut c_void = 0 as *mut _;
const LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = 1 as *mut _;
const LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = 2 as *mut _;

/// Read the packed argument block and its size out of `extra`.
fn unpacked_arguments(extra: *mut *mut c_void) -> Result<(*const u8, usize), CUresult> {
    let mut pointer = None;
    let mut size = None;
    let mut index = 0;
    loop {
        match unsafe { *extra.add(index) } {
            LAUNCH_PARAM_END => break,
            LAUNCH_PARAM_BUFFER_POINTER => {
                pointer = Some(unsafe { *extra.add(index + 1) as *const u8 })
            }
            LAUNCH_PARAM_BUFFER_SIZE => {
                size = Some(unsafe { *(*extra.add(index + 1) as *const usize) })
            }
            _ => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
        }
        index += 2;
    }
    match (pointer, size) {
        (Some(pointer), Some(size)) => Ok((pointer, size)),
        _ => Err(CUresult::CUDA_ERROR_INVALID_VALUE),
    }
}

fn round_up_to_multiple(x: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return x;
    }
    ((x + multiple - 1) / multiple) * multiple
}

/// A device address kept across threads, the same way the staging pool does.
struct SendPtr(*mut c_void);
unsafe impl Send for SendPtr {}

/// The buffers handed to kernels that have not been seen to finish with them.
static PUBLISHED: std::sync::OnceLock<std::sync::Mutex<Vec<SendPtr>>> =
    std::sync::OnceLock::new();

/// Release the buffers of the launches before the last drain.
fn free_published() {
    let Some(published) = PUBLISHED.get() else {
        return;
    };
    let mut published = published.lock().unwrap();
    for device in published.drain(..) {
        let _ = super::memory::free_v2(device.0);
    }
}

/// Release everything that is still published, for context teardown.
pub fn release_all() {
    free_published();
}

/// One declaration of an entry point's parameter list.
struct ParamDecl {
    name: String,
    size: usize,
}

/// A located entry point in a PTX text.
struct Entry {
    /// The byte range of the parameter list, without the parentheses.
    params_open: usize,
    params_close: usize,
    /// The byte range of the body, without the braces.
    body_open: usize,
    body_close: usize,
}

/// Rewrite `kernel`'s entry point so its arguments are read from a buffer.
///
/// Returns the rewritten text and the layout, or `None` when the arguments
/// already fit (in which case the text is used unchanged) and also when the
/// entry point cannot be rewritten safely, because then the compiler's own
/// error is a better answer than a wrong kernel.
pub fn rewrite(text: &str, kernel: &str) -> Option<(String, ParamBuffer)> {
    let entry = find_entry(text, kernel)?;
    let params = parse_params(&text[entry.params_open + 1..entry.params_close])?;
    let layout = layout(&params)?;
    let body = &text[entry.body_open + 1..entry.body_close];
    // Which parameters the body actually mentions, so an unused one costs
    // nothing.
    let mut used = vec![false; params.len()];
    let new_body = rewrite_body(body, &params, &mut used)?;
    let mut prelude = String::from("\n\t.reg .b64 %zb_base");
    for (index, is_used) in used.iter().enumerate() {
        if *is_used {
            prelude.push_str(&format!(", %zb_p{}", index));
        }
    }
    prelude.push_str(";\n");
    prelude.push_str(&format!(
        "\tld.param.u64 %zb_base, [{}];\n",
        BUFFER_PARAM
    ));
    for (index, is_used) in used.iter().enumerate() {
        if *is_used {
            prelude.push_str(&format!(
                "\tadd.s64 %zb_p{}, %zb_base, {};\n",
                index, layout.offsets[index]
            ));
        }
    }
    let mut result = String::with_capacity(text.len() + prelude.len() + 128);
    result.push_str(&text[..entry.params_open + 1]);
    result.push_str(&format!(
        "\n\t.param .align 8 .b8 {}[8]\n",
        BUFFER_PARAM
    ));
    result.push_str(&text[entry.params_close..entry.body_open + 1]);
    result.push_str(&prelude);
    result.push_str(&new_body);
    result.push_str(&text[entry.body_close..]);
    Some((result, layout))
}

/// Find the entry point named `kernel` and the byte ranges of its parameter
/// list and body.
fn find_entry(text: &str, kernel: &str) -> Option<Entry> {
    let mut search = 0;
    while let Some(found) = text[search..].find(".entry") {
        let after = search + found + ".entry".len();
        let rest = &text[after..];
        let name_start = after + rest.len() - rest.trim_start().len();
        let name: String = text[name_start..]
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '(')
            .collect();
        // An entry point name is mangled C++, so it is compared whole: a
        // substring search would rewrite the wrong kernel.
        if name == kernel {
            let params_open = name_start + text[name_start..].find('(')?;
            let params_close = matching(text, params_open, '(', ')')?;
            let body_open = params_close + text[params_close..].find('{')?;
            let body_close = matching(text, body_open, '{', '}')?;
            return Some(Entry {
                params_open,
                params_close,
                body_open,
                body_close,
            });
        }
        search = after;
    }
    None
}

/// The index of the delimiter that closes the one at `open`.
fn matching(text: &str, open: usize, opening: char, closing: char) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, c) in text[open..].char_indices() {
        if c == opening {
            depth += 1;
        } else if c == closing {
            depth -= 1;
            if depth == 0 {
                return Some(open + offset);
            }
        }
    }
    None
}

/// Read an entry point's parameter declarations.
fn parse_params(text: &str) -> Option<Vec<ParamDecl>> {
    let mut result = Vec::new();
    for declaration in split_top_level(text) {
        let declaration = declaration.trim();
        if declaration.is_empty() {
            continue;
        }
        if !declaration.starts_with(".param") && !declaration.contains(".param") {
            // A `.param` is the only thing an entry point's list may hold.
            return None;
        }
        result.push(parse_param(declaration)?);
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Split a parameter list on the commas that are not inside brackets.
fn split_top_level(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (offset, c) in text.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                result.push(&text[start..offset]);
                start = offset + 1;
            }
            _ => {}
        }
    }
    result.push(&text[start..]);
    result
}

/// Read one `.param` declaration: its name and the number of bytes it holds.
fn parse_param(declaration: &str) -> Option<ParamDecl> {
    // The name is the last word; the type and the alignment come before it.
    let mut name = String::new();
    let mut element_size = None;
    let mut vector_width = 1usize;
    for word in declaration.split_whitespace() {
        let word = word.trim_end_matches(',');
        if word.starts_with('.') {
            let type_word = word.trim_start_matches('.');
            if let Some(size) = type_size(type_word) {
                element_size = Some(size);
            } else if let Some(width) = vector_width_of(type_word) {
                vector_width = width;
            } else if let Some((width, scalar)) = split_vector(type_word) {
                vector_width = width;
                element_size = Some(type_size(scalar)?);
            } else if type_word.starts_with('{') && type_word.ends_with('}') {
                // An anonymous structure: its size is not written down.
                return None;
            }
        } else if !word.chars().all(|c| c.is_ascii_digit()) {
            name = word.to_string();
        }
    }
    if name.is_empty() {
        return None;
    }
    let (name, dims) = split_dims(&name);
    let count = match dims {
        Some(dims) => dims,
        None => 1,
    };
    Some(ParamDecl {
        name,
        size: element_size? * vector_width * count,
    })
}

/// `.v2` / `.v4`: the width of a vector type.
fn vector_width_of(word: &str) -> Option<usize> {
    match word {
        "v2" => Some(2),
        "v4" => Some(4),
        _ => None,
    }
}

/// `.v2.b32` written as one token.
fn split_vector(word: &str) -> Option<(usize, &str)> {
    let mut parts = word.split('.');
    let width = vector_width_of(parts.next()?)?;
    Some((width, parts.next()?))
}

/// The size of a scalar type token, e.g. `.b32` or `.f64`.
fn type_size(word: &str) -> Option<usize> {
    Some(match word {
        "b8" | "u8" | "s8" | "pred" => 1,
        "b16" | "u16" | "s16" | "f16" | "bf16" => 2,
        "b32" | "u32" | "s32" | "f32" => 4,
        "b64" | "u64" | "s64" | "f64" => 8,
        // A parameter of a type this parser does not model; the caller falls
        // back to the compiler's own limit error.
        _ => return None,
    })
}

/// Separate a name from its array dimensions: `blob[12][3]` -> (`blob`, 36).
fn split_dims(name: &str) -> (String, Option<usize>) {
    let Some(open) = name.find('[') else {
        return (name.to_string(), None);
    };
    let base = name[..open].to_string();
    let mut count = 1usize;
    let mut rest = &name[open..];
    while let Some(dims_open) = rest.find('[') {
        let Some(dims_close) = rest.find(']') else {
            return (name.to_string(), None);
        };
        match rest[dims_open + 1..dims_close].trim().parse::<usize>() {
            Ok(dimension) => count *= dimension,
            Err(_) => return (name.to_string(), None),
        }
        rest = &rest[dims_close + 1..];
    }
    (base, Some(count))
}

/// Arrange the arguments in a buffer, or `None` when they already fit in the
/// parameter list.
fn layout(params: &[ParamDecl]) -> Option<ParamBuffer> {
    // Every parameter is aligned to the width of its own element class, so the
    // accesses the kernel makes keep the alignment they had in the list.
    fn align_of(size: usize) -> usize {
        if size >= 16 {
            16
        } else if size >= 8 {
            8
        } else {
            4
        }
    }
    let mut offsets = Vec::with_capacity(params.len());
    let mut sizes = Vec::with_capacity(params.len());
    let mut offset = 0usize;
    for param in params {
        offset = (offset + align_of(param.size) - 1) / align_of(param.size) * align_of(param.size);
        offsets.push(offset);
        sizes.push(param.size);
        offset += param.size;
    }
    let total = (offset + 15) / 16 * 16;
    if total <= MAX_ARG_BYTES {
        return None;
    }
    Some(ParamBuffer {
        sizes,
        offsets,
        total,
    })
}

/// Replace every reference to a parameter in the body with the register that
/// holds its address in the buffer, and move the loads out of the parameter
/// space. Returns `None` when a reference this rewrite does not understand is
/// left behind: a kernel that reads its arguments from the wrong place would
/// compute silently wrong results.
fn rewrite_body(body: &str, params: &[ParamDecl], used: &mut [bool]) -> Option<String> {
    let mut result = String::with_capacity(body.len() + 64);
    let bytes = body.as_bytes();
    let mut index = 0;
    while index < body.len() {
        let c = bytes[index] as char;
        if is_word_start(c) {
            let start = index;
            while index < body.len() && is_word_char(bytes[index] as char) {
                index += 1;
            }
            let word = &body[start..index];
            // The access itself and its type are one dotted word in the text,
            // so the space is replaced on the prefix.
            if let Some(access) = word.strip_prefix("ld.param") {
                result.push_str("ld.global");
                result.push_str(access);
            } else if let Some(access) = word.strip_prefix("st.param") {
                result.push_str("st.global");
                result.push_str(access);
            } else if let Some(position) = word_position(word, params) {
                used[position] = true;
                result.push_str(&format!("%zb_p{}", position));
            } else {
                result.push_str(word);
            }
        } else {
            // A run of characters that cannot start a word is copied as it is.
            result.push(c);
            index += 1;
        }
    }
    // `ld.param.b32` and `st.param.b32` are dotted words, so the space is
    // replaced on the prefix; anything else that still names the parameter
    // space means a form this rewrite does not know.
    for line in result.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('.') {
            continue;
        }
        if let Some(space) = find_param_space(trimmed) {
            if space {
                return None;
            }
        }
    }
    Some(result)
}

/// Whether a word is one of the parameter names, and which one.
fn word_position(word: &str, params: &[ParamDecl]) -> Option<usize> {
    params.iter().position(|param| param.name == word)
}

/// Whether a line still uses the parameter space.
fn find_param_space(line: &str) -> Option<bool> {
    for word in line.split(|c: char| c.is_whitespace() || c == '[' || c == ']' || c == ',') {
        if word == ".param" || word.starts_with(".param.") {
            return Some(true);
        }
    }
    None
}

fn is_word_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || c == '$' || c == '%' || c == '.'
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '$' || c == '%' || c == '.'
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: &str = ".version 8.7\n.target sm_75\n.visible .entry add(\n\t.param .u64 out,\n\t.param .u32 n\n)\n{\n\tld.param.u32 %r1, [n];\n\tret;\n}\n";

    #[test]
    fn small_argument_lists_are_left_alone() {
        assert!(rewrite(SMALL, "add").is_none());
    }

    #[test]
    fn parameter_sizes_are_read_from_the_declaration() {
        let params = parse_params(
            "\n\t.param .u64 out,\n\t.param .align 4 .b8 blob[3000],\n\t.param .align 16 .b8 vec[16],\n\t.param .v2 .b32 pair,\n\t.param .f64 scale\n",
        )
        .unwrap();
        let sizes: Vec<usize> = params.iter().map(|p| p.size).collect();
        assert_eq!(sizes, vec![8, 3000, 16, 8, 8]);
        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["out", "blob", "vec", "pair", "scale"]);
    }

    #[test]
    fn a_large_argument_list_is_moved_into_a_buffer() {
        let text = ".version 8.7\n.visible .entry big(\n\t.param .u64 out,\n\t.param .align 4 .b8 blob[3000]\n)\n{\n\tld.param.u64 %rd1, [out];\n\tld.param.u32 %r2, [blob+8];\n\tst.global.u32 [%rd1], %r2;\n\tret;\n}\n";
        let (rewritten, layout) = rewrite(text, "big").expect("should be rewritten");
        assert_eq!(layout.sizes, vec![8, 3000]);
        // out at 0, blob aligned to 16.
        assert_eq!(layout.offsets, vec![0, 16]);
        assert_eq!(layout.total, 3024);
        assert!(rewritten.contains(".param .align 8 .b8 __zluda_param_buffer[8]"));
        assert!(rewritten.contains("ld.param.u64 %zb_base, [__zluda_param_buffer];"));
        assert!(rewritten.contains("add.s64 %zb_p0, %zb_base, 0;"));
        assert!(rewritten.contains("add.s64 %zb_p1, %zb_base, 16;"));
        // The body's parameter loads became global loads through the buffer.
        assert!(rewritten.contains("ld.global.u64 %rd1, [%zb_p0];"));
        assert!(rewritten.contains("ld.global.u32 %r2, [%zb_p1+8];"));
        assert!(!rewritten.contains("ld.param.u32 %r2"));
    }

    #[test]
    fn a_name_that_merely_starts_with_a_parameter_name_is_left_alone() {
        let text = ".version 8.7\n.visible .entry big(\n\t.param .align 4 .b8 blob[3000]\n)\n{\n\tld.param.u32 %r1, [blob];\n\tmov.u32 %r2, 7;\n\tret;\n}\n";
        let (rewritten, _) = rewrite(text, "big").expect("should be rewritten");
        assert!(rewritten.contains("ld.global.u32 %r1, [%zb_p0];"));
        assert!(!rewritten.contains("%zb_p0blob"));
    }

    #[test]
    fn a_kernel_that_is_not_the_entry_is_not_touched() {
        let text = ".version 8.7\n.visible .entry other(\n\t.param .align 4 .b8 blob[3000]\n)\n{\n\tld.param.u32 %r1, [blob];\n\tret;\n}\n";
        assert!(rewrite(text, "big").is_none());
    }
}
