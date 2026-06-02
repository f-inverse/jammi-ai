# @jammi/client

The official TypeScript gRPC-web client for the Jammi engine.

It is **generated from the canonical proto** at `crates/jammi-ai/proto/jammi/v1`
(the single codegen source for every language binding — Rust, Python, and this
one) using [buf] + [protobuf-es] (`protoc-gen-es`) + [Connect-ES]. The transport
is gRPC-web over `fetch`, so the client runs unchanged in a browser, a
**Cloudflare Worker / V8 isolate** (no native code), or Node — interoperating
with the server's `tonic-web` surface.

## Use

```ts
import { connect, Modality } from "@jammi/client";

// One transport, one session-scoped tenant binding, a client per service.
const jammi = connect("https://engine.example.com");

await jammi.session.setTenant({ tenant: { id: tenantUuid } });

const hits = await jammi.embedding.search({
  sourceId: "corpus",
  query: { case: "rowKey", value: "track-42" },
  k: 10,
});

// Server-streaming subscribe is an async iterable.
for await (const batch of jammi.trigger.subscribe({ topic: { name: "events" } })) {
  // ...
}
```

`connect(endpoint, opts?)` builds the gRPC-web transport and returns a
`JammiClient` with one client per service (`session`, `embedding`, `inference`,
`eval`, `fineTune`, `mutableTable`, `channel`, `trigger`, `audit`). Each
connection mints an opaque session id (overridable via `opts.sessionId`) and
injects it as the `jammi-session-id` header on every request — the key the
server binds tenant state against. Pass extra interceptors via
`opts.interceptors`.

## Regenerate the client surface

The generated code lives in `src/gen/` and is **never committed** — it is
emitted at build time from the repo proto, so it cannot drift. One command:

```sh
npm run generate   # == buf generate ../../crates/jammi-ai/proto
```

`npm run build` (and `typecheck` / `test`) regenerate first, so you rarely call
`generate` directly. After changing a `.proto`, just rebuild.

## Develop

```sh
npm install
npm run typecheck   # tsc --noEmit (strict), regenerates first
npm run test        # vitest (hermetic — no network), regenerates first
npm run build       # clean + generate + tsc → dist/
```

Versioned in lockstep with the engine (`workspace.package.version`); the npm
publish runs only on the engine's release tag.

[buf]: https://buf.build
[protobuf-es]: https://github.com/bufbuild/protobuf-es
[Connect-ES]: https://github.com/connectrpc/connect-es
