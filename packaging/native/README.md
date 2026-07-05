# jammi-ai-native

The compiled, in-process Jammi engine — the `jammi_native` PyO3 extension, packaged as a standalone wheel.

```python
import jammi_native

db = jammi_native.open_local(artifact_dir="/data")
```

`jammi-ai-native` is the engine `.so` and nothing else. Most users install
[`jammi-ai`](https://pypi.org/project/jammi-ai/) instead — the pure-Python
convenience front-end that bundles this wheel plus the remote
[`jammi-client`](https://pypi.org/project/jammi-client/) and exposes a single
`connect(target)` that dispatches `file://` targets to this engine and remote
targets to the client. This dist exists so that lean deployments (for example a
client that opts into an embedded engine) can depend on the compiled engine
directly, without pulling the whole front-end.

The dist ships in lockstep at the shared workspace version with `jammi-ai`,
`jammi-client`, and `jammi-server`.
