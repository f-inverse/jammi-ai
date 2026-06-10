"""CUDA (cu12) build of the Jammi engine server.

The package is a thin launcher around a bundled, CUDA-enabled `jammi-server`
binary compiled from the Rust workspace. The CUDA runtime libraries arrive as
`nvidia-*-cu12` wheel dependencies; the launcher (`_entry.main`) puts them on the
loader path before exec'ing the binary. There is no Python API surface here.
"""
