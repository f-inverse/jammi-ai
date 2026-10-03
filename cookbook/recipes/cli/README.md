# Operate a server from the command line

`jammi` is the command-line client of a `jammi-server`: everything an operator
does to a running server — check it, register sources, embed and search, read
models and jobs, manage mutable tables, topics and provenance channels, and
reconcile storage — without writing a program.

**When to use this pattern.** You run Jammi as a server and want to inspect or
change it from a shell, a cron job or a deployment script. Every command takes
`--target`, the server's address, and `--tenant` to act inside one tenant's
scope; `search` prints JSON lines, one row per line, for a pipeline to read.

## Run it

Needs a `jammi-server` on PATH, and the `jammi` CLI (fetched for your platform
if it is not):

```bash
pip install jammi-server
python cookbook/recipes/cli/example.py
```

It walks one server through each command group and prints what each command
printed.
