#!/usr/bin/env bash
# Sourced by fa2_ab.sh (never executed on its own; has no shebang-exec
# path of its own worth taking): one repeat of one (shape, leg) combination
# in fa2_ab.sh's flash-vs-block A/B sweep -- run the binary, capture
# step_rc, parse the emitted report, and fold both into the CALLER's
# overall_rc. Factored out of the inline loop body into its own file so a
# test can `source` this EXACT code (never a hand-written stand-in of it)
# against a stub binary and observe overall_rc move for real.
#
#   fa2_ab_run_leg BIN OUT_DIR DISABLE_KEY LEG BATCH SEQ REPEAT CONFIG_ARGS...
#
# `BIN` is the binary path fa2_ab.sh already resolved and provenance-checked
# before the sweep loop started (`check_producer_provenance_gates.py`'s
# producer-parity gate reads that assignment, and the `provenance`/
# `build_sha` cross-check it pairs with, off fa2_ab.sh's own text -- both
# stay there, never here). This file takes the binary as a PARAMETER and
# assigns no jammi-bench BINARY path of its own.
#
# Must be `source`d, not executed in a subshell: `step_rc`, `parse_rc`, and
# (only on a refusal) `overall_rc` are written into the CALLING shell, the
# same as fa2_ab.sh's own inline loop body always has -- a subshell's
# variable writes would never reach the caller and a refused leg would stop
# moving fa2_ab.sh's own exit status.
fa2_ab_run_leg() {
  local bin="$1" out_dir="$2" disable_key="$3" leg="$4" batch="$5" seq="$6" r="$7"
  shift 7
  local c=("$@")
  local json_path="$out_dir/b${batch}_s${seq}_${leg}.${r}.json"
  local err_path="$out_dir/b${batch}_s${seq}_${leg}.${r}.err"
  # `--expect-kernels-disabled` is ALWAYS passed (finetune_ab.sh:582's
  # convention): the block leg names the SAME op key it puts in
  # JAMMI_KERNELS_DISABLE, and the flash leg passes the empty string -- an
  # exact-set-equality guard against an ambient JAMMI_KERNELS_DISABLE
  # leaking into the "flash" leg from the calling shell, which would
  # otherwise silently turn it back into the block leg wearing a flash
  # label. This makes the binary itself refuse (nonzero exit), before any
  # step runs, if the expectation and the real env var disagree.
  if [ "$leg" = block ]; then
    JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE="$disable_key" "$bin" finetune-step "${c[@]}" --batch "$batch" --seq "$seq" --expect-kernels-disabled "$disable_key" > "$json_path" 2> "$err_path"
  else
    JAMMI_KERNELS_STRICT=1 "$bin" finetune-step "${c[@]}" --batch "$batch" --seq "$seq" --expect-kernels-disabled "" > "$json_path" 2> "$err_path"
  fi
  step_rc=$?
  python3 -c "
import json,sys
try:
  t=json.load(open(sys.argv[1]))['tiers']['finetune_step']; c={k:v for k,v in t.items() if 'flash' in k or k.startswith('attention_block')}
  print('FA2AB',sys.argv[2],sys.argv[3],'p50',round(t['s_per_step_p50']['value'],4),c,'req',t.get('kernels_disabled_requested'),'fired',t.get('kernels_disabled_fired'))
except Exception as e:
  print('FA2AB',sys.argv[2],sys.argv[3],'FAILED',e, open(sys.argv[4]).read()[-300:].replace(chr(10),' | '))
  sys.exit(1)" "$json_path" "b${batch}s${seq}" "${leg}-${r}" "$err_path"
  parse_rc=$?
  # shellcheck disable=SC2034 # written for the CALLER's overall_rc (this
  # file is always sourced, never run standalone -- see the module doc).
  if [ "$step_rc" -ne 0 ] || [ "$parse_rc" -ne 0 ]; then overall_rc=1; fi
}
