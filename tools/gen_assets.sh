#!/usr/bin/env bash
# Generate the frontend sprites with Retro Diffusion (paid: RD Pro 32x32, ~$0.18 each, 7 sprites
# ~= $1.26 per run). Candidates land in golddiggers/build/rd/, the first of each is copied to
# src/golddiggers/static/assets/<name>.png, which the frontend prefers over its built-in art.
# Run from the repo root: bash golddiggers/tools/gen_assets.sh
set -euo pipefail

RD="python3 general/skills/retro-diffusion-api/tools/retro_diffusion.py"
OUT=golddiggers/build/rd
ASSETS=golddiggers/src/golddiggers/static/assets
STYLE=rd_pro__default
mkdir -p "$OUT" "$ASSETS"

$RD balance

gen() {   # name, prompt, extra flags
  local name=$1 prompt=$2; shift 2
  $RD generate "$prompt" --style "$STYLE" -W 32 -H 32 -n 1 --seed 11 --out "$OUT" \
      --name "$name" --max-cost 0.40 "$@"
  cp "$OUT/${name}_0.png" "$ASSETS/$name.png"
}

gen grass  "seamless top-down forest floor grass tile, dark green, subtle texture"
gen tree   "single small round pine tree seen from above at an angle, forest obstacle" --remove-bg
gen gold   "small pile of shiny gold nuggets" --remove-bg
gen depot  "wooden treasure chest trading post, gold depot" --remove-bg
gen mark   "small red flag on a wooden stick" --remove-bg
gen miner0 "gold miner with red shirt, yellow helmet and pickaxe, full body, facing viewer" --remove-bg
gen miner1 "gold miner with blue shirt, yellow helmet and pickaxe, full body, facing viewer" --remove-bg

$RD balance
