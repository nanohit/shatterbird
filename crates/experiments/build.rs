//! Source hash for provenance (the tree is not under version control):
//! FNV-1a 64 over every .rs/.toml file of the workspace, in sorted path order.

use std::fs;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().map(|n| n != "target").unwrap_or(true) {
                    walk(&p, out);
                }
            } else if matches!(p.extension().and_then(|x| x.to_str()), Some("rs") | Some("toml")) {
                out.push(p);
            }
        }
    }
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();
    walk(&root.join("crates"), &mut files);
    files.push(root.join("Cargo.toml"));
    files.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for f in &files {
        let rel = f.strip_prefix(&root).unwrap_or(f).to_string_lossy().to_string();
        for b in rel.bytes().chain(fs::read(f).unwrap_or_default()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    println!("cargo:rustc-env=SB_SOURCE_HASH={h:016x}");
    println!("cargo:rerun-if-changed=../../crates");
    println!("cargo:rerun-if-changed=../../Cargo.toml");
}
