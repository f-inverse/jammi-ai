# jammi-server

CPU build of the Jammi engine server, packaged as a pip-installable wheel that
ships the compiled `jammi-server` binary behind a `jammi-server` console script.

```console
pip install jammi-server
jammi-server --help
```

The wheel bundles a prebuilt Linux `x86_64` binary (manylinux_2_28 glibc floor)
compiled with the broker and cloud-storage backends. For GPU-accelerated
inference and fine-tuning, install `jammi-server-cu12` instead — it provides the
same `jammi-server` command built against the CUDA backend.
