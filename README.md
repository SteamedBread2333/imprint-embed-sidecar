# imprint-embed-sidecar

**English** · [中文](./README.zh-CN.md)

Local HTTP sidecar for **imprint**: **embeddings** and **NLI pair scores**. Imprint (Go) applies thresholds and decides duplicate vs advisory; this repo only serves numbers.

## Quick start

Config lives in **`~/.imprint/embed.yaml`** (`package` = absolute path to this repo). `imprint embed prepare` and `imprint plugin start embed` work from any directory.

```bash
# Requires Python ≥ 3.11 (macOS /usr/bin/python3 is often too old)
PYTHON=/opt/homebrew/bin/python3.13 ./scripts/setup.sh   # if needed

./scripts/setup.sh
./run.sh prepare                    # download + warm models; or: imprint embed prepare

# ~/.imprint/embed.yaml → package: /abs/path/to/imprint-embed-sidecar
imprint plugin start embed

curl --noproxy '*' http://127.0.0.1:4174/health   # expect "ready": true
```

Imprint Console can clone this repo and write `embed.yaml` for you.

**Optional:** skip model warm at start: `IMPRINT_EMBED_NO_WARM=1 ./run.sh`.  
**China / slow HF:** `export IMPRINT_HF_ENDPOINT=https://hf-mirror.com` before prepare.

## Config sketch

```yaml
enabled: true
package: /path/to/imprint-embed-sidecar
config:
  port: 4174
  duplicate_threshold: 0.65
  relate_contradiction_min: 0.40
  relate_entailment_min: 0.35
  relate_bidirectional: true
  duplicate_high_cosine_block: 0.80   # 0 = off; blocks high-cos paraphrases when NLI is neutral
  timeout_seconds: 5
  # relate_model: off                 # omit for default; off disables POST /relate
```

Default model IDs are in `imprint.plugin.json` (`configSchema`). Override at runtime with `IMPRINT_EMBED_MODEL` or `server.py --model`.

**On disk:** weights `~/.cache/imprint/models/` · venv `~/.imprint/plugins/embed/venv` · logs `~/.imprint/state/plugins/embed/sidecar.log` · vectors stored in the vault (`rule_vectors`).

## HTTP API

| Method | Body | Limits |
|--------|------|--------|
| `GET /health` | — | `ready`, embed/relate capability |
| `POST /embed` | `{"texts":["…"]}` | ≤64 strings, 4000 chars each → `vectors`, `dim`, `model` |
| `POST /relate` | `{"pairs":[["a","b"],…]}` | ≤64 pairs → `results[]` with `contradiction`, `neutral`, `entailment` |

Imprint gate order (when cos ≥ threshold): **contradiction** → **entailment** → **high cosine block**. Bidirectional relate merges max(contradiction/entailment) in Go.

Sidecar not ready → imprint falls back to lexical (Jaccard); `add` still works.

## Tune & verify

| Script | Use |
|--------|-----|
| `calibrate.py` | Grid over T / C / E on 60 labeled pairs; prints suggested yaml |
| `eval_gate.py` | Same gate as Go against a **running** sidecar (`--bidirectional`, `--high-cos-block 0.80`) |

After changing embed or relate models: `calibrate.py` → `eval_gate.py` → imprint’s `embed-eval-batch.py` on a test vault.

## Repo map

| Path | Role |
|------|------|
| `server.py` | `/health`, `/embed`, `/relate` |
| `relate.py` | CrossEncoder NLI |
| `run.sh` | Start sidecar; `prepare` subcommand |
| `scripts/setup.sh` | Create venv + install deps |
