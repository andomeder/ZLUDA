//! Translate a PTX file to SPIR-V and write it out, for external validation.
//
// Usage: ptx_to_spirv <file.ptx> <out.spv>

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: ptx_to_spirv <file.ptx> <out.spv>");
        std::process::exit(2);
    }
    let source = std::fs::read_to_string(&args[1]).expect("cannot read PTX file");
    let mut errors = Vec::new();
    let ast = match ptx::ModuleParser::new().parse(&mut errors, &source) {
        Ok(ast) => ast,
        Err(err) => {
            eprintln!("PARSE ERROR: {:?}", err);
            for e in &errors {
                eprintln!("  {:?}", e);
            }
            std::process::exit(1);
        }
    };
    // Recovery errors do not stop the parse, but the loader refuses a module
    // that has any, so a translator debug run needs to see all of them.
    if !errors.is_empty() {
        eprintln!("{} recovery error(s):", errors.len());
        for error in &errors {
            eprintln!("  {:?}", error);
        }
    }
    match ptx::to_spirv_module(ast) {
        Ok(module) => {
            let words = module.assemble();
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(&args[2], &bytes).expect("cannot write SPIR-V");
            println!(
                "wrote {} ({} words, {} entrypoints, {} kernels)",
                args[2],
                words.len(),
                module.spirv.entry_points.len(),
                module.kernel_info.len()
            );
        }
        Err(err) => {
            eprintln!("TRANSLATE ERROR: {:?}", err);
            std::process::exit(1);
        }
    }
}
