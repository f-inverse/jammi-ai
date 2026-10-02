# One program, two transports

Run the same program against the engine in your process and against a
`jammi-server`, and get the same answer.

**When to use this pattern.** You prototype locally (`file://`) and deploy
against a server (`grpc://`), or you serve several clients from one
engine. `jammi.connect(target)` picks the transport from the target alone.

## Run it

Needs a `jammi-server` on PATH:

```bash
pip install jammi-server
python cookbook/recipes/remote_session/example.py
```

It prints the five nearest rows from each transport, which agree, and the
journal of every session opened and closed.
