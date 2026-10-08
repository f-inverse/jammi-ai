#!/usr/bin/env bash
# Tests for `merge_image_index.sh` over a stubbed `docker` on PATH: the
# verify-then-promote order, each refusal arm, and the restore guidance.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/merge_image_index.sh"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bin"
export STUB_LOG="$work/calls" STUB_DIR="$work"
# The stub answers from files in $STUB_DIR: `index.json` for a dry-run or a
# json inspect, `digest.<tag>` for a digest inspect (missing: "not found").
cat > "$work/bin/docker" <<'STUB'
#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG"
case "$*" in
  "buildx imagetools create --dry-run"*) cat "$STUB_DIR/index.json" ;;
  "buildx imagetools create"*) [ -f "$STUB_DIR/create-fails" ] && exit 1; [ -f "$STUB_DIR/index-after.json" ] && cp "$STUB_DIR/index-after.json" "$STUB_DIR/index.json"; exit 0 ;;
  "buildx imagetools inspect "*"--format {{.Manifest.Digest}}")
    tag="$4"; f="$STUB_DIR/digest.${tag//[:\/]/_}"
    if [ -f "$STUB_DIR/inspect-broken" ]; then echo "connection refused" >&2; exit 1; fi
    if [ -f "$f" ]; then cat "$f"; else echo "$tag: not found" >&2; exit 1; fi ;;
  "buildx imagetools inspect "*"--format {{json .Manifest}}") cat "$STUB_DIR/index.json" ;;
  *) echo "unexpected docker call: $*" >&2; exit 99 ;;
esac
STUB
chmod +x "$work/bin/docker"
export PATH="$work/bin:$PATH"
unknown='{"platform":{"os":"unknown","architecture":"unknown"}}'
good='{"manifests":[{"platform":{"os":"linux","architecture":"amd64"}},{"platform":{"os":"linux","architecture":"arm64"}},'"$unknown,$unknown"']}'
one_arch='{"manifests":[{"platform":{"os":"linux","architecture":"amd64"}},'"$unknown"']}'
D1="sha256:$(printf '1%.0s' {1..64})"; D2="sha256:$(printf '2%.0s' {1..64})"
fails=0
check() { if [ "$1" = "$2" ]; then echo "ok   $3"; else echo "FAIL $3: expected [$1] got [$2]"; fails=$((fails+1)); fi; }
reset() { rm -f "$STUB_DIR"/digest.* "$STUB_DIR"/inspect-broken "$STUB_DIR"/create-fails "$STUB_DIR"/index-after.json "$STUB_LOG"; echo "$good" > "$STUB_DIR/index.json"; }
run() { set +e; out="$(bash "$SCRIPT" "$@" 2>&1)"; rc=$?; set -e; }

reset; echo "$D1" > "$STUB_DIR/digest.img_sha"; echo "$D2" > "$STUB_DIR/digest.img_latest"
run --platforms linux/amd64,linux/arm64 --tag img:sha --tag img:latest -- img:sha-amd64 img:sha-arm64
check 0 "$rc" "a good merge succeeds"
check "digest=$D1" "$(tail -n1 <<<"$out")" "prints the immutable tag's digest"
check "$(printf '%s\n' dry create)" "$(grep -o 'create --dry-run\|create -t' "$STUB_LOG" | sed 's/create --dry-run/dry/; s/create -t/create/' | uniq)" "dry-runs before it creates"

reset; echo "$one_arch" > "$STUB_DIR/index.json"
run --platforms linux/amd64,linux/arm64 --tag img:sha -- img:sha-amd64
check 1 "$rc" "a dry-run missing a platform refuses"
check 0 "$(grep -c 'imagetools create -t' "$STUB_LOG" || true)" "nothing is pushed on a bad dry run"

reset; touch "$STUB_DIR/inspect-broken"
run --platforms linux/amd64,linux/arm64 --tag img:sha -- a b
check 1 "$rc" "an unreadable previous digest refuses"
check 0 "$(grep -c 'imagetools create -t' "$STUB_LOG" || true)" "and pushes nothing"

reset; echo "$D1" > "$STUB_DIR/digest.img_sha"; echo "$D2" > "$STUB_DIR/digest.img_latest"; echo "$one_arch" > "$STUB_DIR/index-after.json"
run --platforms linux/amd64,linux/arm64 --tag img:sha --tag img:latest -- a b
check 1 "$rc" "a wrong post-push index fails"
check 1 "$(grep -c "create -t img:latest img@$D2" <<<"$out")" "names latest's own previous digest to restore"
check 1 "$(grep -c "create -t img:sha img@$D1" <<<"$out")" "and sha's own"

reset; echo "$D1" > "$STUB_DIR/digest.img_sha"
run --platforms linux/amd64,linux/arm64 --tag img:sha --tag img:new -- a b
check 0 "$rc" "a first publish of one tag merges"
check 1 "$(grep -c 'img:new: first publish' <<<"$out")" "and says which tag is new"

[ "$fails" -eq 0 ] || { echo "$fails check(s) failed"; exit 1; }
echo "merge_image_index: all checks passed"
