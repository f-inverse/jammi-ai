// The cookbook's one program through the TypeScript client: register a source,
// embed it, and search it over gRPC-web against a running server.
//
// Usage: `npx tsx search.ts <endpoint> <corpus-url> <model> <row-key>`. Prints
// the three rows nearest `<row-key>` as JSON lines, in rank order.

import { connect, FileFormat, Modality, SourceKind } from "@f-inverse/jammi-client";
import { tableFromIPC } from "apache-arrow";

const [endpoint, corpus, model, rowKey] = process.argv.slice(2);
const jammi = connect(endpoint);
const sourceId = "corpus_ts";

// 1. Register the source.
await jammi.catalog.addSource({
  sourceId,
  sourceKind: SourceKind.FILE,
  connection: { url: corpus, format: FileFormat.PARQUET },
});

// 2. Embed its `content` column, one vector per `id`.
await jammi.embedding.generateEmbeddings({
  sourceId,
  modelId: model,
  columns: ["content"],
  keyColumn: "id",
  modality: Modality.TEXT,
});

// 3. Search: the rows nearest the vector stored for `rowKey`. The result is an
// Arrow IPC stream (header + body), read with Apache Arrow's JS library.
const { result } = await jammi.embedding.search({
  sourceId,
  query: { case: "rowKey", value: rowKey },
  k: 3,
  select: ["id", "similarity"],
});
if (!result) throw new Error("search returned no result batch");
const ipc = new Uint8Array(result.dataHeader.length + result.dataBody.length);
ipc.set(result.dataHeader, 0);
ipc.set(result.dataBody, result.dataHeader.length);
for (const row of tableFromIPC(ipc)) {
  console.log(JSON.stringify({ id: Number(row.id), similarity: row.similarity }));
}
