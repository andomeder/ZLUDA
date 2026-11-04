//! Translate a PTX file and report the result.
//!
//! Usage: ptx_translate <file.ptx>
//! Prints "TRANSLATE OK: N entrypoints, M kernels" or the parse/translate error.

fn main() {
    let path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("usage: ptx_translate <file.ptx>");
            std::process::exit(2);
        }
    };
    let source = std::fs::read_to_string(&path).expect("cannot read PTX file");
    let mut errors = Vec::new();
    let ast = match ptx::ModuleParser::new().parse(&mut errors, &source) {
        Ok(ast) => {
            if !errors.is_empty() {
                eprintln!("PARSE WARNINGS: {} errors", errors.len());
                for e in errors.iter().take(3) {
                    eprintln!("  {:?}", e);
                }
            }
            ast
        }
        Err(err) => {
            eprintln!("PARSE ERROR: {:?}", err);
            for e in &errors {
                eprintln!("  {:?}", e);
            }
            std::process::exit(1);
        }
    };
    match ptx::to_spirv_module(ast) {
        Ok(module) => {
            let entrypoints = module.spirv.entry_points.len();
            let kernels = module.kernel_info.len();
            let words = module.assemble().len();
            println!(
                "TRANSLATE OK: {} entrypoints, {} kernels, {} spirv words",
                entrypoints, kernels, words
            );
            if std::env::var_os("PTX_LIST_KERNELS").is_some() {
                for name in module.kernel_info.keys() {
                    println!("kernel: {}", name);
                }
            }
        }
        Err(err) => {
            eprintln!("TRANSLATE ERROR: {:?}", err);
            std::process::exit(1);
        }
    }
}
