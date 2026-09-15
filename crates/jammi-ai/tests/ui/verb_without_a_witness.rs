//! On a worker thread there is no witness to pass: the verb cannot be
//! called at all.
use jammi_ai::fine_tune::collective::{Collective, Noop};

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let noop = Noop::new();
        noop.barrier().unwrap();
    });
}
