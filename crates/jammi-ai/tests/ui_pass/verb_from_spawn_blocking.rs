//! The control: the same verb from a `spawn_blocking` closure compiles.
use jammi_ai::fine_tune::collective::{BlockingCall, Collective, Noop};

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let noop = Noop::new();
        BlockingCall::spawn_blocking(move |call| noop.barrier(&call))
            .await
            .unwrap()
            .unwrap();
    });
}
