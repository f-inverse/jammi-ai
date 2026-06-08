.PHONY: generate test clean

# The canonical jammi.v1 proto tree. `jammi-client` is generated from the
# SAME protos the Rust engine, npm client, and embed wheel speak — one source,
# no parallel schema.
PROTO_ROOT := ../../crates/jammi-ai/proto
OUT_DIR    := jammi_client/_generated

# Every jammi.v1 service proto. Compiling the whole tree (not just the
# embedding + session verbs `connect` exposes today) keeps the generated
# surface a faithful mirror of the npm client and lets the client grow new
# verbs without a Makefile edit — the proto stays the single source.
PROTO_FILES := \
	$(PROTO_ROOT)/jammi/v1/session.proto \
	$(PROTO_ROOT)/jammi/v1/embedding.proto \
	$(PROTO_ROOT)/jammi/v1/inference.proto \
	$(PROTO_ROOT)/jammi/v1/eval.proto \
	$(PROTO_ROOT)/jammi/v1/training.proto \
	$(PROTO_ROOT)/jammi/v1/mutable_table.proto \
	$(PROTO_ROOT)/jammi/v1/channel.proto \
	$(PROTO_ROOT)/jammi/v1/trigger.proto \
	$(PROTO_ROOT)/jammi/v1/audit.proto \
	$(PROTO_ROOT)/jammi/v1/error.proto

# `generate` rebuilds the proto stubs under _generated/. Run after any
# *.proto change. CI runs `make generate` and asserts no diff; if this
# target produces changes that aren't in git, CI fails.
generate:
	mkdir -p $(OUT_DIR)
	python -m grpc_tools.protoc \
		-I$(PROTO_ROOT) \
		--python_out=$(OUT_DIR) \
		--grpc_python_out=$(OUT_DIR) \
		--pyi_out=$(OUT_DIR) \
		$(PROTO_FILES)
	# `grpc_tools.protoc` emits modules with package-style imports
	# (`from jammi.v1 import embedding_pb2`). We rewrite those to relative
	# imports (`from . import embedding_pb2`) so the stubs resolve correctly
	# when imported as `jammi_client._generated.jammi.v1.…`.
	python scripts/fix_proto_imports.py $(OUT_DIR)
	# Drop empty __init__.py files at every package level.
	find $(OUT_DIR) -type d -exec touch {}/__init__.py \;

test:
	pytest tests/ -v

clean:
	rm -rf $(OUT_DIR)
