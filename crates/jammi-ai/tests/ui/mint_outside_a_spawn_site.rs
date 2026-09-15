//! The witness has no constructor outside its three minting sites.
use jammi_ai::fine_tune::collective::{BlockingCall, Collective, Noop};

fn main() {
    let call = BlockingCall::mint();
    Noop::new().barrier(&call).unwrap();
}
