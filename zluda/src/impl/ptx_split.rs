//! Split a PTX translation unit into the part that one kernel needs.
//!
//! A fatbin record is a whole translation unit, and one record of llama.cpp is
//! sixteen million SPIR-V words once translated. A run uses a handful of its
//! kernels, so the record is kept as text and translated per kernel: the entry
//! point asked for plus the functions it reaches through `call`. Everything
//! outside the chosen blocks is dropped, including the other entry points.

/// The name of an entry point or function and the text of its body.
struct Block<'a> {
    name: &'a str,
    is_entry: bool,
    /// Byte range of the block in the source text.
    start: usize,
    end: usize,
}

/// Header text (directives and global declarations) and the bodies.
struct Parsed<'a> {
    /// Everything that is not a function body: `.version`, `.target`,
    /// `.address_size`, declarations of globals, shared arrays, parameters of
    /// extern functions and so on.
    header: String,
    blocks: Vec<Block<'a>>,
    text: &'a str,
}

/// Strip a `//` comment so that braces inside one are not counted.
fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(index) => &line[..index],
        None => line,
    }
}

/// The name of the function a line opens, if it opens one. The forms include
/// `.entry name(`, `.func name(` (with `.visible`/`.weak`/`.extern`/`.hidden`
/// prefixes) and `.func (.param .b8 retval[2]) name(`.
fn function_name(line: &str) -> Option<(&str, bool)> {
    let mut rest = line.trim_start();
    for prefix in [".visible ", ".weak ", ".extern ", ".hidden "] {
        if let Some(tail) = rest.strip_prefix(prefix) {
            rest = tail;
        }
    }
    let (mut rest, is_entry) = if let Some(tail) = rest.strip_prefix(".entry ") {
        (tail, true)
    } else if let Some(tail) = rest.strip_prefix(".func ") {
        (tail, false)
    } else {
        return None;
    };
    // The name may be preceded by a return value list, and the prefix may be
    // followed by more than one space (".func  (.param ...) name(").
    rest = rest.trim_start();
    if rest.starts_with('(') {
        let close = rest.find(')')?;
        rest = rest[close + 1..].trim_start();
    }
    let name = rest
        .split(|c: char| c == '(' || c.is_whitespace())
        .next()?;
    if name.is_empty() || name.starts_with('.') {
        return None;
    }
    Some((name, is_entry))
}

fn parse(text: &str) -> Parsed<'_> {
    let mut header = String::with_capacity(text.len() / 8);
    let mut blocks = Vec::new();
    // The block being read: where it starts, its name, whether it is an entry
    // point, whether its opening brace has been seen and the brace depth.
    let mut current: Option<(usize, &str, bool, bool, i32)> = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let line_start = offset;
        offset += line.len();
        let code = strip_comment(line);
        let opens = code.matches('{').count() as i32;
        let closes = code.matches('}').count() as i32;
        if current.is_none() {
            match function_name(code) {
                Some((name, is_entry)) => {
                    current = Some((line_start, name, is_entry, false, 0));
                }
                None => {
                    header.push_str(line);
                    continue;
                }
            }
        }
        if let Some((start, name, is_entry, seen_open, depth)) = current {
            let seen_open = seen_open || opens > 0;
            let depth = depth + opens - closes;
            let finished = if seen_open {
                depth == 0
            } else {
                // A declaration ends with a semicolon instead of a brace.
                code.trim_end().ends_with(';')
            };
            if finished {
                if seen_open {
                    blocks.push(Block {
                        name,
                        is_entry,
                        start,
                        end: offset,
                    });
                } else {
                    header.push_str(&text[start..offset]);
                }
                current = None;
            } else {
                current = Some((start, name, is_entry, seen_open, depth));
            }
        }
    }
    if let Some((start, ..)) = current {
        // Unterminated: keep it, the parser will complain.
        header.push_str(&text[start..]);
    }
    Parsed {
        header,
        blocks,
        text,
    }
}

/// The names of the entry points of a unit, in the order they appear.
pub fn entry_names(text: &str) -> Vec<String> {
    parse(text)
        .blocks
        .iter()
        .filter(|block| block.is_entry)
        .map(|block| block.name.to_string())
        .collect()
}

/// The names a block calls with `call`.
fn callees(body: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut rest = body;
    while let Some(index) = rest.find("call") {
        rest = &rest[index + 4..];
        // The forms are "call NAME" and "call.uni (retvals), NAME, (params)".
        let after = rest.trim_start();
        let after = if let Some(after) = after.strip_prefix(".uni") {
            after.trim_start()
        } else if after.starts_with('(') || after.starts_with(';') || after.starts_with('.') {
            // Another `call` prefix or an unrelated word containing "call".
            continue;
        } else {
            after
        };
        // A return value list may come first, then the name.
        let after = if after.starts_with('(') {
            match after.find(')') {
                Some(close) => {
                    let tail = &after[close + 1..];
                    tail.trim_start().strip_prefix(',').map(|s| s.trim_start()).unwrap_or(tail)
                }
                None => continue,
            }
        } else {
            after
        };
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$' || *c == '.')
            .collect();
        if !name.is_empty() {
            result.push(name);
        }
    }
    result
}

/// The text of one kernel and everything it reaches.
///
/// Returns `None` when the unit does not have that entry point.
pub fn split_for_kernel(text: &str, kernel: &str) -> Option<String> {
    let parsed = parse(text);
    let by_name = |name: &str| -> Vec<usize> {
        parsed
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.name == name)
            .map(|(index, _)| index)
            .collect()
    };
    let entry = by_name(kernel)
        .into_iter()
        .find(|index| parsed.blocks[*index].is_entry)?;
    let mut keep = vec![entry];
    let mut seen = std::collections::HashSet::new();
    seen.insert(kernel.to_string());
    let mut queue = vec![entry];
    while let Some(index) = queue.pop() {
        let block = &parsed.blocks[index];
        let body = &text[block.start..block.end];
        for callee in callees(body) {
            if !seen.insert(callee.clone()) {
                continue;
            }
            for found in by_name(&callee) {
                if !parsed.blocks[found].is_entry {
                    keep.push(found);
                    queue.push(found);
                }
            }
        }
    }
    keep.sort_unstable();
    // Every function the kept code calls has to be in the result, either as a
    // kept block or as a declaration in the header. A callee the closure missed
    // would be replaced by a stub that computes the wrong thing, so an
    // incomplete split is refused rather than handed over.
    let defined = |name: &str| -> bool {
        parsed.blocks.iter().any(|block| block.name == name)
    };
    // The name has to appear as a whole identifier: a substring test also
    // matches a longer symbol, and a split that drops a definition leaves the
    // caller with the stub body that returns zero.
    let declared = |name: &str| -> bool {
        parsed
            .header
            .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$' || c == '.'))
            .any(|token| token == name)
    };
    let mut kept_names = std::collections::HashSet::new();
    for index in keep.iter() {
        kept_names.insert(parsed.blocks[*index].name);
    }
    for index in keep.iter() {
        let block = &parsed.blocks[*index];
        for callee in callees(&text[block.start..block.end]) {
            let kept = kept_names.contains(callee.as_str());
            if !kept && defined(&callee) && !declared(&callee) {
                return None;
            }
        }
    }
    let mut result = String::with_capacity(parsed.header.len() + 64 * 1024);
    result.push_str(&parsed.header);
    for index in keep {
        let block = &parsed.blocks[index];
        result.push_str(&text[block.start..block.end]);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: &str = ".version 6.5\n\
.target sm_80\n\
.address_size 64\n\
.global .align 1 .b8 table[16];\n\
.visible .func helper(\n\
\t.param .u64 a\n\
)\n\
{\n\
\tret;\n\
}\n\
.visible .entry one(\n\
\t.param .u64 p\n\
)\n\
{\n\
\t{ // a brace in a comment }\n\
\tcall.uni (%%r1), helper, (%%rd1);\n\
\t}\n\
\tret;\n\
}\n\
.visible .entry two(\n\
\t.param .u64 p\n\
)\n\
{\n\
\tret;\n\
}\n";

    #[test]
    fn finds_the_entry_points() {
        assert_eq!(entry_names(UNIT), vec!["one".to_string(), "two".to_string()]);
    }

    #[test]
    fn keeps_the_header_and_the_callees() {
        let split = split_for_kernel(UNIT, "one").unwrap();
        assert!(split.contains(".address_size"));
        assert!(split.contains("table[16]"));
        assert!(split.contains(".entry one"));
        assert!(split.contains(".func helper"));
        assert!(!split.contains(".entry two"));
    }

    const NESTED: &str = ".version 6.5\n\
.address_size 64\n\
.weak .func _Z1av\n\
(\n\
)\n\
;\n\
.visible .entry k(\n\
\t.param .u64 p\n\
)\n\
{\n\
\tcall.uni (retval0), _Z1av, ();\n\
\tret;\n\
}\n\
.weak .func _Z1av(\n\
)\n\
{\n\
\tcall.uni (retval0), _Z1bv, ();\n\
\tret;\n\
}\n\
.weak .func _Z1bv(\n\
)\n\
{\n\
\tret;\n\
}\n";

    #[test]
    fn follows_calls_two_levels_deep() {
        let split = split_for_kernel(NESTED, "k").unwrap();
        assert!(split.contains(".entry k"), "kernel missing");
        assert!(split.contains("_Z1av(\n)"), "first callee missing");
        assert!(split.contains("_Z1bv(\n)"), "second callee missing");
    }

    #[test]
    fn a_kernel_without_callees() {
        let split = split_for_kernel(UNIT, "two").unwrap();
        assert!(split.contains(".entry two"));
        assert!(!split.contains(".func helper"));
        assert!(!split.contains(".entry one"));
    }

    // Probe: split a real record and write it out for comparison.
    #[test]
    fn dump_real_record() {
        let path = "/home/cloudsurfer/projects/intel-zluda/tmp/rasterizer_impl_app.ptx";
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let parsed = parse(&text);
        for block in &parsed.blocks {
            eprintln!(
                "BLOCK {} entry={} bytes={}..{}",
                block.name, block.is_entry, block.start, block.end
            );
        }
        for kernel in entry_names(&text) {
            let split = split_for_kernel(&text, &kernel).unwrap();
            std::fs::write(format!("/tmp/opencode/split_{}.ptx", kernel), split).unwrap();
        }
    }

    #[test]
    fn an_unknown_kernel() {
        assert!(split_for_kernel(UNIT, "three").is_none());
    }
}
