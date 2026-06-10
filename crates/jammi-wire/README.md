# jammi-wire

The gRPC wire substrate for [Jammi AI](https://github.com/f-inverse/jammi-ai).

`jammi-wire` is the candle-free home for everything both sides of the `jammi.v1` wire share: the generated tonic stubs (client + server), the proto↔domain conversions, the request / eval / fine-tune vocabulary the conversions map, the Arrow-IPC framing helpers, and the `SessionTransport` the typed clients build their per-service stubs over. It pulls no embedded ML stack, so a consumer that speaks the wire links no candle / tokenizers / symphonia / hf-hub.

`jammi-server` consumes the server stubs and these conversions; `jammi-admin` (control plane) and `jammi-client` (data plane) consume the client stubs and the same conversions — neither side reimplements a mapping.
