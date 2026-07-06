# jammi-ai-native

The compiled, in-process Jammi engine — the `jammi_native` PyO3 extension, packaged as a standalone wheel.

```python
import jammi_native

db = jammi_native.open_local(artifact_dir="/data")
```

`jammi-ai-native` is the engine `.so` and nothing else. Most users install
[`jammi-ai`](https://pypi.org/project/jammi-ai/) instead — the base Python
client (import `jammi`) whose single `connect(target)` dispatches `file://`
targets to this engine (pulled in via its `[embedded]` extra) and remote targets
to the gRPC transport. This dist exists so that lean deployments (for example a
client that opts into an embedded engine) can depend on the compiled engine
directly.

The dist ships in lockstep at the shared workspace version with `jammi-ai` and
`jammi-server`.
