// @jammi/client — the official TypeScript gRPC-web client for the Jammi engine.
//
// The wire surface is GENERATED from the canonical jammi.v1 proto by
// protoc-gen-es (see buf.gen.yaml + the `generate` script); this module is the
// thin ergonomic seam over it. It does three things and nothing more:
//
//   1. Re-export the generated message types + service descriptors, so a
//      consumer imports everything from one entry point.
//   2. Provide `connect(endpoint, opts?)`, which builds ONE gRPC-web transport
//      (fetch-based — runs in a Cloudflare Worker / V8 isolate, interoperating
//      with the server's tonic-web surface) and returns a client for every
//      service over that shared transport.
//   3. Centralize tenant scoping: like the Rust `RemoteSession::connect`, it
//      mints one opaque session id per connection and injects it on every
//      outbound request as the `jammi-session-id` header — the key the server's
//      tenant interceptor binds tenant state against (set via the SessionService
//      tenant trio). No per-call header plumbing leaks to consumers.
//
// Encoding/decoding is entirely the generated code's job; this file holds no
// hand-rolled protobuf logic.

import { createClient, type Client, type Interceptor } from "@connectrpc/connect";
import { createGrpcWebTransport } from "@connectrpc/connect-web";

import { SessionService } from "./gen/jammi/v1/session_pb.js";
import { EmbeddingService } from "./gen/jammi/v1/embedding_pb.js";
import { InferenceService } from "./gen/jammi/v1/inference_pb.js";
import { EvalService } from "./gen/jammi/v1/eval_pb.js";
import { FineTuneService } from "./gen/jammi/v1/fine_tune_pb.js";
import { MutableTableService } from "./gen/jammi/v1/mutable_table_pb.js";
import { ChannelService } from "./gen/jammi/v1/channel_pb.js";
import { TriggerService } from "./gen/jammi/v1/trigger_pb.js";
import { AuditService } from "./gen/jammi/v1/audit_pb.js";

// Re-export the full generated surface (messages, enums, service descriptors).
export * from "./gen/jammi/v1/session_pb.js";
export * from "./gen/jammi/v1/embedding_pb.js";
export * from "./gen/jammi/v1/inference_pb.js";
export * from "./gen/jammi/v1/eval_pb.js";
export * from "./gen/jammi/v1/fine_tune_pb.js";
export * from "./gen/jammi/v1/mutable_table_pb.js";
export * from "./gen/jammi/v1/channel_pb.js";
export * from "./gen/jammi/v1/trigger_pb.js";
export * from "./gen/jammi/v1/audit_pb.js";
export * from "./gen/jammi/v1/error_pb.js";

/**
 * The request header carrying a connection's opaque session id. The server's
 * Tonic TenantInterceptor reads it on every request and resolves the bound
 * tenant; it is the same header the Rust SDK and Flight SQL lane use. Kept in
 * one place so the value never drifts across the seam.
 */
export const SESSION_HEADER = "jammi-session-id";

/** Options for {@link connect}. */
export interface ConnectOptions {
  /**
   * The opaque session id to scope this connection's tenant binding by. The
   * server keys tenant state against it, so two connections that must stay
   * tenant-isolated MUST pass distinct ids. Defaults to a fresh UUID, matching
   * the Rust `RemoteSession::connect` behavior (each connection isolated).
   */
  sessionId?: string;
  /**
   * Extra Connect interceptors, applied outside the session-header interceptor
   * (e.g. auth, logging). Optional.
   */
  interceptors?: Interceptor[];
}

/**
 * A client per Jammi service, all sharing one gRPC-web transport and one
 * session-scoped tenant binding. This is the value {@link connect} returns.
 */
export interface JammiClient {
  /** The session id this connection scopes its tenant binding by. */
  readonly sessionId: string;
  readonly session: Client<typeof SessionService>;
  readonly embedding: Client<typeof EmbeddingService>;
  readonly inference: Client<typeof InferenceService>;
  readonly eval: Client<typeof EvalService>;
  readonly fineTune: Client<typeof FineTuneService>;
  readonly mutableTable: Client<typeof MutableTableService>;
  readonly channel: Client<typeof ChannelService>;
  readonly trigger: Client<typeof TriggerService>;
  readonly audit: Client<typeof AuditService>;
}

/** Mint an opaque per-connection session id (a v4 UUID). */
function mintSessionId(): string {
  // `crypto.randomUUID` is available in every target runtime: browsers,
  // Cloudflare Workers / V8 isolates, and Node >= 18.
  return crypto.randomUUID();
}

/**
 * Connect to a Jammi engine's `jammi.v1` gRPC-web endpoint and return a client
 * for every service.
 *
 * @param endpoint The engine's base URL (e.g. `"https://engine.example.com"`).
 *   Requests go to `<endpoint>/<package>.<Service>/<Method>`.
 */
export function connect(endpoint: string, opts: ConnectOptions = {}): JammiClient {
  const sessionId = opts.sessionId ?? mintSessionId();

  // Inject the session header on every outbound request (unary and streaming),
  // mirroring the Rust SDK's SessionHeader interceptor. Applied innermost so a
  // consumer-supplied interceptor cannot accidentally drop it.
  const sessionHeader: Interceptor = (next) => (req) => {
    req.header.set(SESSION_HEADER, sessionId);
    return next(req);
  };

  const transport = createGrpcWebTransport({
    baseUrl: endpoint,
    interceptors: [...(opts.interceptors ?? []), sessionHeader],
  });

  return {
    sessionId,
    session: createClient(SessionService, transport),
    embedding: createClient(EmbeddingService, transport),
    inference: createClient(InferenceService, transport),
    eval: createClient(EvalService, transport),
    fineTune: createClient(FineTuneService, transport),
    mutableTable: createClient(MutableTableService, transport),
    channel: createClient(ChannelService, transport),
    trigger: createClient(TriggerService, transport),
    audit: createClient(AuditService, transport),
  };
}
