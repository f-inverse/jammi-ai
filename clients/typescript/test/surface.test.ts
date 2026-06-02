// Hermetic smoke test for @jammi/client.
//
// It does NOT hit the network. It proves two things the generated surface must
// guarantee, entirely at construction + type level:
//
//   1. `connect()` builds a gRPC-web transport and returns a client for every
//      service, scoped by an injected `jammi-session-id`.
//   2. The FULL verb surface is generated and type-correct — every service's
//      every RPC is referenced with a well-typed request, so a missing verb or
//      a drifted field shape fails `tsc` (the test compiles under the package's
//      strict config). The references sit behind `if (false)` so no call is
//      ever dispatched: this is a type-level completeness proof, not a live RPC.

import { describe, expect, it, expectTypeOf } from "vitest";

import {
  connect,
  SESSION_HEADER,
  type JammiClient,
  // A representative message + enum from each service, proving the message
  // surface (not just the service descriptors) is generated and importable.
  type SearchResponse,
  type EncodeQueryResponse,
  Modality,
  type InferResponse,
  type EmbeddingEvalReport,
  type StartFineTuneResponse,
  type CreateMutableTableResponse,
  type RegisterTopicResponse,
  type SubscribedBatch,
  type AuditFetchRecentResponse,
} from "../src/index.js";

describe("connect()", () => {
  it("returns a client for every jammi.v1 service over one transport", () => {
    const client = connect("https://engine.invalid");
    // Every service arm is present.
    expect(client.session).toBeDefined();
    expect(client.embedding).toBeDefined();
    expect(client.inference).toBeDefined();
    expect(client.eval).toBeDefined();
    expect(client.fineTune).toBeDefined();
    expect(client.mutableTable).toBeDefined();
    expect(client.channel).toBeDefined();
    expect(client.trigger).toBeDefined();
    expect(client.audit).toBeDefined();
  });

  it("mints a fresh session id per connection (tenant isolation)", () => {
    const a = connect("https://engine.invalid");
    const b = connect("https://engine.invalid");
    expect(a.sessionId).not.toBe(b.sessionId);
    expect(a.sessionId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });

  it("honors an explicit session id", () => {
    const id = "11111111-2222-4333-8444-555555555555";
    expect(connect("https://engine.invalid", { sessionId: id }).sessionId).toBe(id);
  });

  it("exposes the tenant header name it injects", () => {
    expect(SESSION_HEADER).toBe("jammi-session-id");
  });
});

// Full verb surface, compile-checked. `if (false)` guards every call: tsc still
// type-checks the bodies, but nothing runs — keeping the test hermetic while
// proving each RPC exists with the right request/response shape.
async function verbSurface(c: JammiClient): Promise<void> {
  if (Math.random() < 0) {
    // ── SessionService: the tenant trio ───────────────────────────────────
    await c.session.setTenant({ tenant: { id: "t-1" } });
    await c.session.getTenant({});
    await c.session.clearTenant({});

    // ── EmbeddingService: add/remove source, generate, encode, search ─────
    await c.embedding.addSource({ sourceId: "s1" });
    await c.embedding.removeSource({ sourceId: "s1" });
    await c.embedding.generateEmbeddings({
      sourceId: "s1",
      modelId: "local:/m",
      columns: ["text"],
      keyColumn: "id",
      modality: Modality.TEXT,
    });
    const enc: EncodeQueryResponse = await c.embedding.encodeQuery({
      modelId: "local:/m",
      modality: Modality.TEXT,
      input: { case: "text", value: "hello" },
    });
    expectTypeOf(enc.embedding).toEqualTypeOf<number[]>();
    const search: SearchResponse = await c.embedding.search({
      sourceId: "s1",
      query: { case: "rowKey", value: "k1" },
      k: 5,
      select: ["title"],
    });
    expectTypeOf(search.hits).toBeArray();

    // ── InferenceService: infer ───────────────────────────────────────────
    const inf: InferResponse = await c.inference.infer({
      sourceId: "s1",
      modelId: "local:/clf",
      keyColumn: "id",
    });
    expectTypeOf(inf).toMatchTypeOf<InferResponse>();

    // ── EvalService: the four eval verbs ──────────────────────────────────
    const embEval: EmbeddingEvalReport = await c.eval.evalEmbeddings({
      sourceId: "s1",
      embeddingTable: "emb_s1",
      goldenSource: "gold",
    });
    expectTypeOf(embEval.evalRunId).toBeString();
    await c.eval.evalPerQuery({ evalRunId: "run-1" });
    await c.eval.evalInference({ modelId: "m", sourceId: "s1", goldenSource: "gold" });
    await c.eval.evalCompare({ sourceId: "s1", goldenSource: "gold" });

    // ── FineTuneService: start + status ───────────────────────────────────
    const ft: StartFineTuneResponse = await c.fineTune.startFineTune({
      sourceId: "s1",
      baseModel: "local:/m",
    });
    await c.fineTune.fineTuneStatus({ jobId: ft.jobId });

    // ── MutableTableService: create + drop ────────────────────────────────
    const mt: CreateMutableTableResponse = await c.mutableTable.createMutableTable({});
    await c.mutableTable.dropMutableTable({ mutableTableId: mt.mutableTableId });

    // ── ChannelService: register + add columns ────────────────────────────
    await c.channel.registerChannel({ channelId: "ch1" });
    await c.channel.addChannelColumns({ channelId: "ch1" });

    // ── TriggerService: register/drop/publish/list + server-streaming ─────
    const topic: RegisterTopicResponse = await c.trigger.registerTopic({
      name: "default.events",
      schema: new Uint8Array(),
    });
    await c.trigger.publish({ topic: { name: "default.events" } });
    await c.trigger.listTopics({ pageSize: 10 });
    // Subscribe is server-streaming: the client method yields an async iterable.
    for await (const batch of c.trigger.subscribe({
      topic: { name: "default.events" },
      predicate: "",
    })) {
      const b: SubscribedBatch = batch;
      expectTypeOf(b.offset).toEqualTypeOf<bigint>();
    }
    await c.trigger.dropTopic({ topicId: topic.topicId, ifExists: true });

    // ── AuditService: log + fetch ─────────────────────────────────────────
    await c.audit.auditLog({});
    await c.audit.auditFetchByQueryId({ queryId: "q1" });
    const recent: AuditFetchRecentResponse = await c.audit.auditFetchRecent({});
    expectTypeOf(recent).toMatchTypeOf<AuditFetchRecentResponse>();
  }
}

describe("verb surface (type-level completeness)", () => {
  it("references every service's every RPC without dispatching one", async () => {
    // The function never makes a call (guarded by an always-false branch); this
    // assertion only proves it is callable and the surface compiles.
    await expect(verbSurface(connect("https://engine.invalid"))).resolves.toBeUndefined();
  });
});
