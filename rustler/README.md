# rustler - context sampler for RT, written in Rust

`pre.rs` contains code to preprocess datasets.

`fly.rs` contains code for on-the-fly context sampling.

`common.rs` contains data structures exchanged between `pre.rs` and `fly.rs`
(using `rkyv`: https://github.com/rkyv/rkyv).

`lib.rs` is the entry point to rustler as a library.

`main.rs` is the entry point to rustler as a standalone executable.
