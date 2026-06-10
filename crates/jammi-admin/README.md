# jammi-admin

The control-plane client for [Jammi AI](https://github.com/f-inverse/jammi-ai).

`jammi-admin` exposes `CatalogClient`, a candle-free `CatalogService` client over the `jammi.v1` gRPC wire. It carries every control verb the single server-side `CatalogService` holds: the source/model registry, the channel declarations, the mutable-table lifecycle, the topic-admin verbs, the server-info handshake, and the tenant trio.

Every failure decodes the structured error detail the server attaches, so a control verb returns the exact `jammi_db::error::JammiError` variant the in-process path would — never a lossy gRPC-code-category guess. Tenant scope rides on the session header the transport stamps, never in a request body.
