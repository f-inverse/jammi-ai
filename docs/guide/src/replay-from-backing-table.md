# Replay Events from the Backing Table

Every topic's event log is a Phase-2 mutable companion table named
`__topic_<topic_id>`. The double-underscore prefix is reserved for
engine-controlled tables; consumers do not register tables under that
namespace. Flight SQL queries against the backing table compose with
the same federation surface the rest of Jammi exposes — joins with
result tables, predicate pushdown, aggregates over event history.

Reach for direct replay when a tenant needs ad-hoc analytics on the
event log that would be awkward through the subscribe surface —
counting events per key, computing per-day rollups, joining the event
stream against a Parquet result table.

## Goal

Run an ad-hoc query that returns the count of `op = 'd'` events per
hour over the durable log for `cdc.orders`.

## Backing table naming

Every registered topic has a backing table whose name is
`__topic_<topic_id>` where `<topic_id>` is the hyphenated lowercase
`TopicId::Display`. To find the name, query `topics`:

```sql
SELECT topic_id, name, backing_table FROM topics WHERE name = 'cdc.orders';
```

## Schema

The backing table's columns are the topic's user schema with three
engine-controlled columns prepended:

| Column          | Type                                      | Purpose                                                |
|-----------------|-------------------------------------------|--------------------------------------------------------|
| `_offset`       | `BIGINT NOT NULL`                         | Monotonic offset; stable across rows of one publish.   |
| `_row_idx`      | `BIGINT NOT NULL`                         | Position within a publish, for the composite PK.       |
| `_produced_at`  | `BIGINT NOT NULL` (UTC microseconds)      | Publisher-side timestamp, single value per offset.     |
| *…user cols…*   | per `TopicDefinition.schema`              | Payload columns.                                       |
| `tenant_id`     | `TEXT` (nullable, added by Phase 2)       | Tenant scope per ADR-00.                               |

The primary key is `(_offset, _row_idx)`; `_offset` is the order column
so `scan_after` and `ORDER BY _offset` agree.

## Query

```sql
SELECT
    DATE_TRUNC('hour', TIMESTAMP_MICROS(_produced_at))      AS hour,
    COUNT(*)                                                 AS deletes
FROM    mutable.public.__topic_019088da_1234_7890_abcd_ef1234567890
WHERE   op = 'd'
GROUP BY hour
ORDER BY hour;
```

Substitute your topic's `backing_table` (looked up from the `topics`
catalog row) for the literal name in the example. The query runs
through Flight SQL like any other federated query — predicate
pushdown applies, joins compose, aggregates run.

## Tenant scoping

The backing table carries the `tenant_id` column added by the Phase-2
mutable backend. Sessions bound to a tenant see only rows whose
`tenant_id` matches or is `NULL`, per Phase 3's predicate-injection
analyzer rule — the same guard that scopes the rest of the catalog.

## See also

- [Publish Events to a Topic](./publish-events.md) — the publisher side.
- [Subscribe with a SQL Predicate Filter](./subscribe-with-filter.md) —
  the live-tail surface for these same events.
- [Register a Mutable Companion Table](./register-mutable-table.md) —
  the substrate the backing table reuses; same schema validation,
  same federation properties.
