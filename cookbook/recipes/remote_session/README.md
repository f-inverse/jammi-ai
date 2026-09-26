# One program, two transports

Run the same program against the engine in your process and against a
`jammi-server`, and get the same answer.

**When to use this pattern.** You prototype locally (`file://`) and deploy
against a server (`grpc://`), or you serve several clients from one
engine. `jammi.connect(target)` picks the transport from the target alone.

## What `example.py` does

1. `jammi.parse_target` — how a target string picks its transport
2. Registers, embeds and searches on an embedded session
3. Starts a server with `jammi.testing.LiveServer` and runs the same
   function over `grpc://`; the results are identical
4. `supports(Capability.…)` on both, and the remote-only `session_id`
5. The session journal — `observe`, `open_sessions`, `open_session_labels`,
   `describe_sessions` — shows every session opened and closed

## Run it

Needs a `jammi-server` on PATH:

```bash
pip install jammi-server
python cookbook/recipes/remote_session/example.py
```
