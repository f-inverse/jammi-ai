//! A `BlockingCall` minted on a blocking thread cannot be carried into a
//! `tokio::spawn`ed future: the witness is `!Send`, so the verb cannot run
//! on a runtime worker thread.
use jammi_ai::fine_tune::collective::{BlockingCall, Collective, Noop};

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let noop = Noop::new();
        let handle = BlockingCall::spawn_blocking(move |call| {
            tokio::spawn(async move { noop.barrier(&call) })
        });
        let _ = handle;
    });
}
