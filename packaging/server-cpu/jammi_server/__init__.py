"""CPU build of the Jammi engine server.

The package is a thin launcher around a bundled, statically-discoverable
`jammi-server` binary compiled from the Rust workspace. There is no Python API
surface here — `import jammi_server` exists only so the `jammi-server` console
script can resolve `jammi_server._entry:main`.
"""
