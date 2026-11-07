//! How much shared memory a kernel can address.
//!
//! A launch asks the engine for `prefix + application_dynamic` bytes of shared
//! memory, and the prefix is the whole `.extern .shared` array the PTX
//! declares. That size is not the kernel's: a lifted kernel declares the
//! cubin's static shared size for its whole module (or 32768 when there is
//! none to read), so `torch.cat` asks for 32 KB to copy a few hundred bytes and
//! CUB's scan kernel asks for 33808 where it reaches about 7 KB. Shared memory
//! is what bounds how many workgroups a core holds, so an over-request costs
//! occupancy on every launch.
//!
//! So the bound is computed from the PTX instead: every shared address is
//! reduced to
//!
//!     symbol base + tid_coefficient * threads + warp_coefficient * warps + bytes
//!
//! and the largest one, with the launch's thread count supplied at launch time,
//! is what the kernel needs. A kernel that indexes shared memory by the thread
//! count - which is most of them - is bounded exactly where a fixed size has to
//! guess.
//!
//! Soundness is the whole point, so the rules are one-sided. An access whose
//! address cannot be reduced (it comes from memory, from a block or grid index,
//! from arithmetic this does not model, or from a shared pointer that arrived
//! as a function argument) makes the whole kernel unbounded, and the caller
//! keeps the declared size. So does a loop that carries an address, and so does
//! a register that is written by an instruction this does not model. `None`
//! therefore means "no bound is provable", never "no memory is used", and the
//! caller falls back to what it did before.

use std::collections::HashMap;

/// Why no bound was provable, for the diagnosis of a kernel that keeps its
/// declared size: set ZLUDA_TRACE_SHARED=1 and a load prints the first
/// instruction the analysis could not reduce.
fn trace(reason: &str, line: &str) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("ZLUDA_TRACE_SHARED").is_some()) {
        eprintln!("shared window: no bound: {}: {}", reason, line.trim());
    }
}

/// `bound(threads) = per_thread * threads + constant`, in bytes.
///
/// `threads` is the launch's total thread count, block_dim.x * y * z. A thread
/// index, the block's own thread count and the block's warp count are all
/// bounded by it, which is what makes the bound usable at launch time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedBound {
    pub per_thread: i64,
    pub constant: i64,
}

impl SharedBound {
    pub fn bytes(&self, threads: u64) -> u64 {
        let value = self
            .per_thread
            .saturating_mul(threads as i64)
            .saturating_add(self.constant);
        if value > 0 {
            value as u64
        } else {
            0
        }
    }
}

/// A value the analysis can reduce to part of an address.
///
/// `sym` marks a value derived from the base of an `.extern .shared` symbol,
/// which is what an address has to be for the analysis to say anything; `tid`
/// counts launched threads, `warp` counts the block's warps, and everything
/// else that is not a constant makes the value `unknown`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Val {
    sym: bool,
    tid: i64,
    warp: i64,
    constant: i64,
    nonneg: bool,
    unknown: bool,
}

impl Val {
    fn unbounded() -> Val {
        Val {
            sym: false,
            tid: 0,
            warp: 0,
            constant: 0,
            nonneg: false,
            unknown: true,
        }
    }

    fn constant(value: i64) -> Val {
        Val {
            sym: false,
            tid: 0,
            warp: 0,
            constant: value,
            nonneg: value >= 0,
            unknown: false,
        }
    }

    fn base() -> Val {
        Val {
            sym: true,
            tid: 0,
            warp: 0,
            constant: 0,
            nonneg: true,
            unknown: false,
        }
    }

    fn threads() -> Val {
        Val {
            sym: false,
            tid: 1,
            warp: 0,
            constant: 0,
            nonneg: true,
            unknown: false,
        }
    }

    fn warps() -> Val {
        Val {
            sym: false,
            tid: 0,
            warp: 1,
            constant: 0,
            nonneg: true,
            unknown: false,
        }
    }

    fn add(self, other: Val) -> Val {
        if self.unknown || other.unknown || !self.nonneg || !other.nonneg {
            return Val::unbounded();
        }
        Val {
            sym: self.sym || other.sym,
            tid: self.tid.saturating_add(other.tid),
            warp: self.warp.saturating_add(other.warp),
            constant: self.constant.saturating_add(other.constant),
            nonneg: true,
            unknown: false,
        }
    }

    /// `self - other`: an address that moves down does not raise the bound, so
    /// only `other`'s constant part is taken off and its other terms are
    /// dropped, which can only over-estimate.
    fn sub(self, other: Val) -> Val {
        if self.unknown || other.unknown || !self.nonneg {
            return Val::unbounded();
        }
        Val {
            sym: self.sym || other.sym,
            tid: self.tid,
            warp: self.warp,
            constant: self.constant.saturating_sub(other.constant.max(0)),
            nonneg: false,
            unknown: false,
        }
    }

    fn scale(self, factor: i64) -> Val {
        if self.unknown || factor < 0 {
            return Val::unbounded();
        }
        Val {
            sym: self.sym,
            tid: self.tid.saturating_mul(factor),
            warp: self.warp.saturating_mul(factor),
            constant: self.constant.saturating_mul(factor),
            nonneg: self.nonneg,
            unknown: false,
        }
    }

    /// The larger of two values, term by term: a predicate that selects
    /// between two addresses can reach either.
    fn join(self, other: Val) -> Val {
        if self.unknown || other.unknown {
            return Val::unbounded();
        }
        Val {
            sym: self.sym || other.sym,
            tid: self.tid.max(other.tid),
            warp: self.warp.max(other.warp),
            constant: self.constant.max(other.constant),
            nonneg: self.nonneg && other.nonneg,
            unknown: false,
        }
    }
}

/// Bytes one access of this type and vector width touches.
fn access_width(opcode: &str) -> i64 {
    let width = if opcode.contains(".v4") {
        4
    } else if opcode.contains(".v2") {
        2
    } else {
        1
    };
    let element = if opcode.contains("64") {
        8
    } else if opcode.contains("32") {
        4
    } else if opcode.contains("16") {
        2
    } else {
        1
    };
    element * width
}

fn is_shared_access(opcode: &str) -> bool {
    if !opcode.contains(".shared") {
        return false;
    }
    matches!(
        opcode.split('.').next().unwrap_or(""),
        "ld" | "st" | "atom" | "red" | "ldmatrix" | "stmatrix" | "cp" | "mbarrier" | "barrier"
    )
}

/// Split `"a, b, c;"` into its operands, keeping `{}` and `[]` groups whole.
fn split_operands(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    for ch in text.chars() {
        match ch {
            '{' | '[' => {
                depth += 1;
                current.push(ch);
            }
            '}' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                out.push(current.trim().to_string());
                current.clear();
            }
            ';' if depth == 0 => break,
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

/// Split an address operand `"%rd2+-256"` into its base and byte offset.
fn split_address(inner: &str) -> (String, i64) {
    let inner = inner.trim();
    let bytes = inner.as_bytes();
    for i in (1..bytes.len()).rev() {
        if bytes[i] == b'+' || bytes[i] == b'-' {
            let rest = &inner[i + 1..];
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                let value: i64 = rest.parse().unwrap_or(0);
                let offset = if bytes[i] == b'-' { -value } else { value };
                return (inner[..i].trim().to_string(), offset);
            }
        }
    }
    (inner.to_string(), 0)
}

fn address_of(operand: &str) -> Option<&str> {
    let start = operand.find('[')?;
    let end = operand[start..].find(']')? + start;
    Some(&operand[start + 1..end])
}

fn is_register(token: &str) -> bool {
    token.trim().starts_with('%')
}

fn operand_value(token: &str, registers: &HashMap<String, Val>) -> Val {
    let token = token.trim();
    match token {
        "%tid.x" | "%tid.y" | "%tid.z" | "%ntid.x" | "%ntid.y" | "%ntid.z" => Val::threads(),
        "%warpid" | "%nwarpid" => Val::warps(),
        "%laneid" => Val::constant(31),
        _ => {
            if let Some(text) = token.strip_prefix("0x") {
                i64::from_str_radix(text, 16)
                    .map(Val::constant)
                    .unwrap_or_else(|_| Val::unbounded())
            } else if let Ok(value) = token.parse::<i64>() {
                Val::constant(value)
            } else {
                registers
                    .get(token)
                    .copied()
                    .unwrap_or_else(Val::unbounded)
            }
        }
    }
}

/// The largest coefficients any single shared access in `text` needs.
///
/// A shared address is bounded by `tid * threads + warp * (threads / 32 + 1) +
/// fixed`, and the caller turns that into a `SharedBound`. Returns `None` as
/// soon as one access cannot be reduced, which is the safe answer.
pub fn bound(text: &str) -> Option<SharedBound> {
    let symbols = shared_symbols(text);
    if symbols.is_empty() {
        return Some(SharedBound {
            per_thread: 0,
            constant: 0,
        });
    }
    let bodies = function_bodies(text);
    if bodies.is_empty() {
        return None;
    }
    let mut tid = 0i64;
    let mut warp = 0i64;
    let mut fixed = 0i64;
    let mut saw_any = false;
    for body in bodies.iter() {
        let (body_tid, body_warp, body_fixed, body_saw) = analyse_body(body, &symbols)?;
        tid = tid.max(body_tid);
        warp = warp.max(body_warp);
        fixed = fixed.max(body_fixed);
        saw_any |= body_saw;
    }
    if !saw_any {
        // The symbol is declared but never addressed: nothing is needed.
        return Some(SharedBound {
            per_thread: 0,
            constant: 0,
        });
    }
    Some(SharedBound {
        per_thread: tid.saturating_add((warp + 31) / 32),
        constant: fixed.saturating_add(warp),
    })
}

fn shared_symbols(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with(".extern") || !line.contains("shared") {
            continue;
        }
        if let Some(open) = line.find('[') {
            let name = line[..open]
                .split_whitespace()
                .last()
                .unwrap_or("")
                .trim_end_matches('.');
            if !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    out
}

fn function_bodies(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let is_function = trimmed.starts_with(".entry")
            || trimmed.starts_with(".visible .entry")
            || trimmed.starts_with(".func")
            || trimmed.starts_with(".visible .func")
            || trimmed.starts_with(".weak .func")
            || trimmed.starts_with(".extern .func");
        if !is_function {
            continue;
        }
        // The header runs to a `{`, and a declaration of a function that is
        // never defined here runs to a `;` and has no body. The `{` can be on
        // the declaration's own line or on one of the lines after it.
        let mut opens = trimmed.contains('{');
        if !opens && !trimmed.ends_with(';') {
            for header in lines.by_ref() {
                let header = header.trim();
                if header.ends_with(';') {
                    break;
                }
                if header.contains('{') {
                    opens = true;
                    break;
                }
            }
        }
        if !opens {
            continue;
        }
        let mut body = String::new();
        let mut depth = 1i32;
        for line in lines.by_ref() {
            let trimmed = line.trim();
            depth += trimmed.matches('{').count() as i32;
            depth -= trimmed.matches('}').count() as i32;
            if depth <= 0 {
                break;
            }
            body.push_str(line);
            body.push('\n');
        }
        out.push(body);
    }
    out
}

/// One function body's bound, `(tid, warp, fixed, saw_access)`.
///
/// The body is split at labels into blocks and the register state is propagated
/// over the block graph until it stops growing, because a loop that moves an
/// address has to be seen moving: a linear walk of the text would read the
/// value the first iteration produces and bound the kernel by it. A body that
/// does not settle is not bounded, which is the safe answer.
fn analyse_body(body: &str, symbols: &[String]) -> Option<(i64, i64, i64, bool)> {
    let blocks = build_blocks(body);
    if blocks.is_empty() {
        return Some((0, 0, 0, false));
    }
    let mut entry: Vec<Option<HashMap<String, Val>>> = vec![None; blocks.len()];
    let mut best = (0i64, 0i64, 0i64, false);
    let mut visits = vec![0u32; blocks.len()];
    let mut worklist = vec![0usize];
    entry[0] = Some(HashMap::new());
    const MAX_VISITS: u32 = 8;
    while let Some(index) = worklist.pop() {
        visits[index] += 1;
        if visits[index] > MAX_VISITS {
            // The state is still growing here, so a loop carries it.
            trace("a loop carries the address", "");
            return None;
        }
        let mut state = entry[index].clone().unwrap_or_default();
        let (tid, warp, fixed, saw) = run_block(&blocks[index], symbols, &mut state)?;
        best = (
            best.0.max(tid),
            best.1.max(warp),
            best.2.max(fixed),
            best.3 || saw,
        );
        for successor in blocks[index].successors.iter() {
            let had = entry[*successor].clone();
            let mut merged = match had.clone() {
                Some(existing) => existing,
                None => HashMap::new(),
            };
            for (name, value) in state.iter() {
                merged
                    .entry(name.clone())
                    .and_modify(|previous| *previous = previous.join(*value))
                    .or_insert(*value);
            }
            let changed = had != Some(merged.clone());
            entry[*successor] = Some(merged);
            if changed {
                worklist.push(*successor);
            }
        }
    }
    Some(best)
}

/// A straight run of instructions between two labels.
struct Block {
    instructions: Vec<String>,
    successors: Vec<usize>,
}

fn build_blocks(body: &str) -> Vec<Block> {
    let mut names: HashMap<String, usize> = HashMap::new();
    let mut order: Vec<(Option<String>, Vec<String>)> = Vec::new();
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('$') {
            continue;
        }
        if line.ends_with(':') {
            let name = line.trim_end_matches(':').trim().to_string();
            names.insert(name, order.len());
            order.push((Some(String::new()), Vec::new()));
            continue;
        }
        if order.is_empty() {
            order.push((Some(String::new()), Vec::new()));
        }
        order.last_mut().unwrap().1.push(line.to_string());
    }
    let mut blocks: Vec<Block> = order
        .into_iter()
        .map(|(_, instructions)| Block {
            instructions,
            successors: Vec::new(),
        })
        .collect();
    for index in 0..blocks.len() {
        let instructions = blocks[index].instructions.clone();
        let mut successors = Vec::new();
        let mut ends_here = false;
        for line in instructions.iter() {
            let (opcode, operands) = match line.find(char::is_whitespace) {
                Some(position) => (&line[..position], &line[position..]),
                None => (line.as_str(), ""),
            };
            let head = opcode.trim_start_matches('@').split('.').next().unwrap_or("");
            let unconditional = !opcode.starts_with('@');
            match head {
                "bra" | "brx" => {
                    let target = operands
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_end_matches(';')
                        .to_string();
                    if let Some(target_index) = names.get(&target) {
                        successors.push(*target_index);
                    }
                    if unconditional {
                        ends_here = true;
                    }
                }
                "ret" | "exit" => {
                    if unconditional {
                        ends_here = true;
                    }
                }
                _ => {}
            }
        }
        if !ends_here && index + 1 < blocks.len() {
            successors.push(index + 1);
        }
        successors.sort_unstable();
        successors.dedup();
        blocks[index].successors = successors;
    }
    blocks
}

fn run_block(
    block: &Block,
    symbols: &[String],
    registers: &mut HashMap<String, Val>,
) -> Option<(i64, i64, i64, bool)> {
    let mut best_tid = 0i64;
    let mut best_warp = 0i64;
    let mut best_fixed = 0i64;
    let mut saw_access = false;
    for raw in block.instructions.iter() {
        let mut line = raw.as_str();
        if line.starts_with('{') || line.starts_with('}') || line.starts_with('.') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('@') {
            line = match rest.split_once(char::is_whitespace) {
                Some((_, rest)) => rest.trim_start(),
                None => continue,
            };
        }
        let (opcode, operands) = match line.find(char::is_whitespace) {
            Some(index) => (&line[..index], split_operands(&line[index..])),
            None => continue,
        };
        let Some(destination) = operands.first() else {
            continue;
        };
        // Every instruction that names a register first overwrites it. Anything
        // this does not model therefore forgets what it knew: keeping a stale
        // value would bound an address by a register the code has since reused.
        let new_value = if is_shared_access(opcode) {
            let address = match operands.iter().find(|operand| operand.contains('[')) {
                Some(operand) => match address_of(operand) {
                    Some(address) => address,
                    None => {
                        trace("no address operand", line);
                        return None;
                    }
                },
                None => {
                    trace("no address operand", line);
                    return None;
                }
            };
            let (base, offset) = split_address(address);
            let value = if symbols.iter().any(|symbol| *symbol == base) {
                Val::base()
            } else {
                operand_value(&base, registers)
            };
            if value.unknown || !value.sym {
                trace(if value.unknown { "unbounded value" } else { "not the shared base" }, line);
                return None;
            }
            saw_access = true;
            best_tid = best_tid.max(value.tid);
            best_warp = best_warp.max(value.warp);
            best_fixed = best_fixed.max(
                value
                    .constant
                    .saturating_add(offset)
                    .saturating_add(access_width(opcode)),
            );
            if opcode.starts_with("ld") || opcode.starts_with("atom") {
                Val::unbounded()
            } else {
                continue;
            }
        } else if opcode.starts_with("call") {
            Val::unbounded()
        } else {
            evaluate(opcode, &operands[1..], symbols, registers)
        };
        if is_register(destination) {
            registers.insert(destination.clone(), new_value);
        }
    }
    Some((best_tid, best_warp, best_fixed, saw_access))
}

fn evaluate(
    opcode: &str,
    sources: &[String],
    symbols: &[String],
    registers: &HashMap<String, Val>,
) -> Val {
    let head = opcode.split('.').next().unwrap_or("");
    let argument = |index: usize| -> Val {
        match sources.get(index) {
            Some(token) => {
                let token = token.trim();
                if symbols.iter().any(|symbol| symbol == token) {
                    Val::base()
                } else {
                    operand_value(token, registers)
                }
            }
            None => Val::unbounded(),
        }
    };
    match head {
        "mov" => {
            if sources.len() == 1 && !sources[0].starts_with('{') {
                argument(0)
            } else {
                Val::unbounded()
            }
        }
        // A widening of a value that cannot be negative keeps its bound; a
        // zero-extended negative value would be huge, so anything else is not
        // bounded.
        "cvt" => match argument(0) {
            value if !value.unknown && value.nonneg => value,
            _ => Val::unbounded(),
        },
        "add" => argument(0).add(argument(1)),
        "sub" => argument(0).sub(argument(1)),
        "mul" | "mad" | "madc" => {
            let a = argument(0);
            let b = argument(1);
            let product = if a.unknown || b.unknown {
                return Val::unbounded();
            } else if a.tid == 0 && a.warp == 0 {
                b.scale(a.constant)
            } else if b.tid == 0 && b.warp == 0 {
                a.scale(b.constant)
            } else {
                // A product of two values that both scale with the launch is
                // quadratic, which no linear bound covers.
                return Val::unbounded();
            };
            if matches!(head, "mad" | "madc") && sources.len() >= 3 {
                product.add(argument(2))
            } else {
                product
            }
        }
        "shl" => {
            let a = argument(0);
            let b = argument(1);
            if b.unknown || b.tid != 0 || b.warp != 0 || b.constant < 0 || b.constant > 31 {
                return Val::unbounded();
            }
            a.scale(1i64 << b.constant)
        }
        // A shift down cannot raise an address above where it started.
        "shr" => argument(0),
        "and" => {
            let a = argument(0);
            let b = argument(1);
            if a.sym || b.sym {
                return Val::unbounded();
            }
            if a.unknown || b.unknown {
                return Val::unbounded();
            }
            if a.tid == 0 && a.warp == 0 {
                // `x & mask` is at most the mask.
                Val::constant(a.constant.max(0))
            } else if b.tid == 0 && b.warp == 0 {
                Val::constant(b.constant.max(0))
            } else {
                Val::unbounded()
            }
        }
        // Both keep the sum of the operands as an upper bound.
        "or" | "xor" => argument(0).add(argument(1)),
        "selp" => argument(0).join(argument(1)),
        "min" => {
            let a = argument(0);
            let b = argument(1);
            if a.unknown || b.unknown {
                Val::unbounded()
            } else if a.tid == 0 && a.warp == 0 && (b.tid != 0 || b.warp != 0) {
                a
            } else if b.tid == 0 && b.warp == 0 && (a.tid != 0 || a.warp != 0) {
                b
            } else {
                a.join(b)
            }
        }
        "max" => argument(0).join(argument(1)),
        // Everything else is not modelled, so its result is not an address
        // this can bound.
        _ => Val::unbounded(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero() -> Option<SharedBound> {
        Some(SharedBound {
            per_thread: 0,
            constant: 0,
        })
    }

    #[test]
    fn no_shared_symbol_is_zero() {
        assert_eq!(bound(".version 8.0\n.visible .entry k() { ret; }"), zero());
    }

    #[test]
    fn a_declared_array_that_is_never_touched_is_free() {
        let text = "\
.extern .shared .align 16 .b8 shared_mem[32768];
.visible .entry k() {
\tmov.u32 %r1, %ctaid.x;
\tret;
}";
        assert_eq!(bound(text), zero());
    }

    #[test]
    fn a_constant_access_is_its_own_extent() {
        let text = "\
.extern .shared .align 16 .b8 shared_mem[32768];
.visible .entry k() {
\tmov.b64 %rd1, shared_mem;
\tst.volatile.shared.b64 [%rd1+24], %rd2;
\tret;
}";
        assert_eq!(
            bound(text),
            Some(SharedBound {
                per_thread: 0,
                constant: 32
            })
        );
    }

    /// The shape of tests/driver/shared_prefix_test: the static part is 1024
    /// bytes and every thread writes four more after it.
    #[test]
    fn a_thread_indexed_access_scales_with_the_launch() {
        let text = "\
.extern .shared .align 16 .b8 shared_mem[1024];
.visible .entry k(.param .u64 p) {
\tmov.u32 %r1, %tid.x;
\tshl.b32 %r2, %r1, 2;
\tadd.s32 %r3, %r2, 1024;
\tcvt.u64.u32 %rd1, %r3;
\tmov.b64 %rd2, shared_mem;
\tadd.s64 %rd3, %rd2, %rd1;
\tst.volatile.shared.b32 [%rd3], %r4;
\tret;
}";
        let found = bound(text).expect("bounded");
        assert_eq!(found.per_thread, 4);
        assert_eq!(found.bytes(1024), 1024 + 4 * 1024 + 4);
        assert_eq!(found.bytes(128), 1024 + 4 * 128 + 4);
    }

    #[test]
    fn a_value_loaded_from_memory_is_not_bounded() {
        let text = "\
.extern .shared .align 16 .b8 shared_mem[32768];
.visible .entry k(.param .u64 p) {
\tld.param.u64 %rd1, [p];
\tmov.b64 %rd2, shared_mem;
\tadd.s64 %rd3, %rd2, %rd1;
\tld.volatile.shared.b32 %r1, [%rd3];
\tret;
}";
        assert_eq!(bound(text), None);
    }

    #[test]
    fn a_block_index_in_an_address_is_not_bounded() {
        let text = "\
.extern .shared .align 16 .b8 shared_mem[32768];
.visible .entry k() {
\tmov.u32 %r1, %ctaid.x;
\tshl.b32 %r2, %r1, 2;
\tcvt.u64.u32 %rd1, %r2;
\tmov.b64 %rd2, shared_mem;
\tadd.s64 %rd3, %rd2, %rd1;
\tst.volatile.shared.b32 [%rd3], %r4;
\tret;
}";
        assert_eq!(bound(text), None);
    }

    #[test]
    fn a_loop_carried_address_is_not_bounded() {
        let text = "\
.extern .shared .align 16 .b8 shared_mem[32768];
.visible .entry k() {
\tmov.b64 %rd1, shared_mem;
L_loop:
\tst.volatile.shared.b32 [%rd1], %r1;
\tadd.s64 %rd1, %rd1, 8;
\tbra L_loop;
}";
        assert_eq!(bound(text), None);
    }

    #[test]
    fn a_warp_index_is_counted_as_warps_not_threads() {
        let text = "\
.extern .shared .align 16 .b8 shared_mem[33808];
.visible .entry k() {
\tmov.u32 %r1, %warpid;
\tmul.lo.s32 %r2, %r1, 1792;
\tcvt.u64.u32 %rd1, %r2;
\tmov.b64 %rd2, shared_mem;
\tadd.s64 %rd3, %rd2, %rd1;
\tst.volatile.shared.b64 [%rd3], %rd4;
\tret;
}";
        let found = bound(text).expect("bounded");
        assert_eq!(found.per_thread, 56);
        assert_eq!(found.bytes(128), 1792 * (128 / 32 + 1) + 8);
    }

    #[test]
    fn a_register_reused_by_an_unmodelled_instruction_is_not_bounded() {
        let text = "\
.extern .shared .align 16 .b8 shared_mem[32768];
.visible .entry k(.param .u64 p) {
\tmov.b64 %rd1, shared_mem;
\tld.global.u64 %rd1, [%rd9];
\tst.volatile.shared.b32 [%rd1], %r4;
\tret;
}";
        assert_eq!(bound(text), None);
    }
}

#[cfg(test)]
mod corpus {
    /// Distribution over a directory of real PTX files:
    ///
    ///     SHAREDDIAG_CORPUS=<dir> cargo test -p zluda --lib corpus -- --nocapture
    ///
    /// Prints, per kernel, the declared size, the bound the analysis proves and
    /// whether that bound would narrow the launch.
    #[test]
    fn report_corpus() {
        let Some(root) = std::env::var_os("SHAREDDIAG_CORPUS") else {
            return;
        };
        let mut files = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(root)];
        while let Some(directory) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().map(|e| e == "ptx").unwrap_or(false) {
                    files.push(path);
                }
            }
        }
        files.sort();
        let mut bounded = 0;
        let mut narrowed = 0;
        let mut unbounded = 0;
        for path in files {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let declared = super::shared_symbols(&text)
                .into_iter()
                .filter_map(|name| {
                    text.lines().find_map(|line| {
                        if line.trim().starts_with(".extern") && line.contains(&name) {
                            let open = line.find('[')?;
                            let close = line.find(']')?;
                            line[open + 1..close].parse::<i64>().ok()
                        } else {
                            None
                        }
                    })
                })
                .max()
                .unwrap_or(0);
            let entry = text
                .lines()
                .find(|line| line.contains(".entry"))
                .and_then(|line| line.split(".entry").nth(1))
                .map(|rest| rest.split('(').next().unwrap_or("").trim().to_string())
                .unwrap_or_default();
            match super::bound(&text) {
                Some(found) => {
                    bounded += 1;
                    let need = found.bytes(1024);
                    if (need as i64) < declared {
                        narrowed += 1;
                    }
                    println!(
                        "BOUND    declared={:6} bound1024={:6} {}{}",
                        declared,
                        need,
                        if (need as i64) < declared { " NARROWS " } else { "         " },
                        entry.chars().take(60).collect::<String>()
                    );
                }
                None => {
                    unbounded += 1;
                    println!(
                        "UNBOUND  declared={:6}                       {}",
                        declared,
                        entry.chars().take(60).collect::<String>()
                    );
                }
            }
        }
        println!(
            "SUMMARY kernels={} bounded={} narrowed_at_1024_threads={} unbounded={}",
            bounded + unbounded,
            bounded,
            narrowed,
            unbounded
        );
    }
}
