#!/usr/bin/env bash
# Time the multi_tile pipeline at any N, over several core counts and cards.
#
# Usage:  ./multi_tile/run_bench.sh N "CORES..." [ITERS] ["CARDS..."]
#   e.g.  ./multi_tile/run_bench.sh 1048576 "1 8 56" 5 "4 5 7"
#
# The input comes from the baseline's generator (turboquant_host --mode input).
# Each card gets a contiguous, tile-aligned slice of the N vectors and runs the
# six stages on it in build/bench/card<k>/ with TT_VISIBLE_DEVICES=<k> (default
# card: $TT_VISIBLE_DEVICES, else 0). All cards run at once, and Stage 2a takes
# multi_tile Stage 1's output. For each core count it prints each card's summed
# stage time (best of ITERS warm runs) and checks that each card's output is
# byte-identical to its output at the first core count. Needs about 3 GB of
# disk per million vectors; rm -rf build/bench afterwards.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${TT_METAL_HOME:?Set TT_METAL_HOME first}"
: "${TT_METAL_RUNTIME_ROOT:=$TT_METAL_HOME}"
export TT_METAL_RUNTIME_ROOT

[[ $# -ge 2 ]] || { sed -n '4,5p' "$0"; exit 1; }
N="$1"; read -r -a CORES <<< "$2"; ITERS="${3:-5}"; read -r -a CARDS <<< "${4:-${TT_VISIBLE_DEVICES:-0}}"
(( ITERS > 0 )) || { echo "ITERS must be at least 1"; exit 1; }
STAGES=(multi_tile multi_tile_pq "multi_tile_ir --input dump_centroids_tile.bin"
        multi_tile_resid multi_tile_proj multi_tile_rnorm)
OUTS=(rotated centroids xhat r b bits rnorm)
BENCH="$REPO/build/bench"
PER=$(( (N / 32 + ${#CARDS[@]} - 1) / ${#CARDS[@]} * 32 ))   # vectors per card, whole tiles

rm -rf "$BENCH"; mkdir -p "$BENCH"
(cd "$BENCH" && "$REPO/build/turboquant_host" --num-vectors "$N" --mode input > /dev/null)
for i in "${!CARDS[@]}"; do
    dir="$BENCH/card${CARDS[$i]}"; lo=$(( i * PER )); COUNT[$i]=$(( N - lo < PER ? N - lo : PER ))
    (( COUNT[i] > 0 )) || { echo "card ${CARDS[$i]} gets no vectors"; exit 1; }
    mkdir -p "$dir/ref"; ln -s "$REPO/multi_tile" "$dir/multi_tile"   # kernel sources
    dd if="$BENCH/dump_input.bin" of="$dir/dump_input.bin" bs=256 skip="$lo" count="${COUNT[$i]}" status=none
    echo "card ${CARDS[$i]}: vectors $lo..$(( lo + COUNT[i] - 1 ))"
done

ok=1
for ci in "${!CORES[@]}"; do
    C=${CORES[$ci]}; pids=(); failed=0
    for i in "${!CARDS[@]}"; do
        k=${CARDS[$i]}
        (cd "$BENCH/card$k"
         export TT_VISIBLE_DEVICES=$k TT_METAL_CACHE="$REPO/build/tt_cache/card$k" TT_METAL_LOGS_PATH="$PWD"
         for s in "${STAGES[@]}"; do
             "$REPO"/build/turboquant_$s --num-vectors "${COUNT[$i]}" --cores "$C" --iters "$ITERS"
         done > log.txt 2>&1) &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "card ${CARDS[$i]} FAILED at cores=$C: build/bench/card${CARDS[$i]}/log.txt"; failed=1; }
    done
    (( failed == 0 )) || exit 1

    echo "cores=$C:"
    times=""
    for k in "${CARDS[@]}"; do
        dir="$BENCH/card$k"
        ms=$(awk '/ warm: / { s += $(NF - 1) } END { printf "%.1f", s }' "$dir/log.txt")
        times+=" $ms"
        if (( ci == 0 )); then
            for f in "${OUTS[@]}"; do mv "$dir/dump_${f}_tile.bin" "$dir/ref/"; done
            same="reference output"
        else
            same="output same as cores=${CORES[0]}"
            for f in "${OUTS[@]}"; do
                cmp -s "$dir/dump_${f}_tile.bin" "$dir/ref/dump_${f}_tile.bin" \
                    || { same="output DIFFERS from cores=${CORES[0]}, first in dump_${f}_tile.bin"; ok=0; break; }
            done
        fi
        echo "  card $k: $ms ms, $same"
    done
    echo "$times" | awk -v n="$N" '{ m = 0; for (i = 1; i <= NF; ++i) if ($i > m) m = $i
        printf "  slowest card %.1f ms: %.2f M vectors/s\n", m, n / m / 1000 }'
done
(( ok == 1 )) && echo "All outputs match." || { echo "OUTPUT MISMATCH"; exit 1; }
