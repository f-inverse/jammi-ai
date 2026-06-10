# jammi-client

The data-plane client for [Jammi AI](https://github.com/f-inverse/jammi-ai).

`jammi-client` exposes `DataClient`, the candle-free network peer of the embedded session for the data verbs: SQL (over Flight SQL), embeddings / encode / search, inference, fine-tune submit + status, the eval verbs, the trigger publish / subscribe surface, and audit. It composes a `jammi_admin::CatalogClient` over the same `SessionTransport` for the control verbs and the tenant trio, so a tenant bound through `bind_tenant` is observed by every data verb on the same session id.

It speaks the gRPC + Flight SQL wire only and pulls no embedded engine. Errors decode to the exact `jammi_db::error::JammiError` variant the in-process path returns, and request encode / response decode reuse the `jammi-wire` conversions the server's receive side uses.
