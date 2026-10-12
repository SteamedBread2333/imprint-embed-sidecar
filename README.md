# imprint-embed-sidecar

语义查重插件：给 imprint 的 `add` 加向量 + NLI 分数，拦住词面 Jaccard 抓不到的语义重复，并把对立政策放进 `similar` advisory。

Python 常驻 sidecar + Go 薄客户端；**sidecar 只出分，duplicate/advisory 决策在 imprint（Go + `imprint.yaml`）**。

## 快速开始

在 **imprint 项目根**（有 `imprint.yaml` 的仓库）操作，不要只在 sidecar 目录里 `plugin start`：

```bash
# 1) 本仓库
./scripts/setup.sh
./run.sh prepare          # 前台下载/加载（HF 进度在终端）；或 imprint embed prepare

# 2) imprint 仓库 imprint.yaml → plugins.embed.package: ../imprint-embed-sidecar
cd ../imprint
imprint embed prepare     # 可选，与 ./run.sh prepare 等价
imprint plugin start embed

curl --noproxy '*' http://127.0.0.1:4174/health   # 需 "ready": true
```

`run.sh` 默认 `--warm`；跳过 warm：`IMPRINT_EMBED_NO_WARM=1 ./run.sh`。

国内网络：Go 启动 sidecar 时会设 `HF_HUB_DISABLE_XET=1`，并在 huggingface.co 不可达时使用 `HF_ENDPOINT=https://hf-mirror.com`。手动下载可 `export IMPRINT_HF_ENDPOINT=https://hf-mirror.com`。

## imprint.yaml 示例

```yaml
plugins:
  embed:
    enabled: true
    package: ../imprint-embed-sidecar
    config:
      port: 4174
      model: BAAI/bge-small-zh-v1.5
      relate_model: cross-encoder/nli-deberta-v3-base
      duplicate_threshold: 0.65
      relate_contradiction_min: 0.40
      relate_entailment_min: 0.35
      relate_bidirectional: true
      duplicate_high_cosine_block: 0.80   # 0 = 关闭；NLI neutral 时高 cos 仍拒写
      timeout_seconds: 5
```

归属：向量在 vault `rule_vectors` · 权重 `~/.cache/imprint/models/` · venv `~/.imprint/plugins/embed/venv`。

## HTTP 接口

```jsonc
// GET /health
{ "ok": true, "ready": true, "embed": {...}, "relate": {"enabled": true, "model": "..."} }

// POST /embed   {"texts": ["..."]}   // ≤64 条，≤4000 字符/条
{ "vectors": [[...]], "dim": 512, "model": "..." }

// POST /relate   {"pairs": [["claim a", "claim b"], ...]}   // ≤64 对
{ "model": "...", "labels": ["contradiction","entailment","neutral"], "results": [{...}] }
```

Go 判定顺序（阈值见 yaml）：**contradiction → entailment → high cosine block**；`relate_bidirectional` 时在 Go 侧打两次 `/relate` 并合并 max 分数。

## 标定与回归

| 脚本 | 作用 |
| --- | --- |
| `calibrate.py` | 60 对样本上扫 **T / C / E** 网格，打印推荐 yaml 片段 |
| `eval_gate.py` | 对 **live sidecar** 复现 Go 语义闸（含 `--bidirectional`、`--high-cos-block`） |
| imprint `scripts/embed-eval-batch.py` | 临时 vault 上真实 `add` 回归（在 imprint 仓库） |

```bash
# 需已加载模型（或 sidecar ready）
python3 calibrate.py

python3 eval_gate.py --bidirectional --high-cos-block 0.80
# 样本集上常见：duplicate 100%，conflict 误拦 0%
```

换 **embed 或 relate 模型** 后：先 `calibrate.py`，再 `eval_gate.py` 扫 `duplicate_high_cosine_block`，最后在 imprint 仓库跑 batch。

### 为何需要 relate + 高余弦兜底

余弦衡量话题相近，**分不清同义改述与反义冲突**（冲突组 cos 中位数 ~0.75，验收对 cos ~0.71）。因此：

| 信号 | Go 行为 |
| --- | --- |
| 高 cos + 高 entail + 低 contradiction | **duplicate 拒写** |
| 高 cos + 高 contradiction | **写入** + `similar` advisory |
| 高 cos + NLI neutral（如 `%w` 改述） | **`duplicate_high_cosine_block`** 拒写（默认 0.80） |

维度以 `/health`、`/embed` 返回的 `dim` 为准（bge-small-zh-v1.5 为 **512**）。

## 验收（imprint CLI）

```
add "Go exported identifiers must use PascalCase"     → 成功
add "Exported things use Pascal Case naming"          → duplicate
add "Wrap errors with fmt.Errorf and %w" 再 add 同义改述 → duplicate（高 cos 兜底）
tabs vs spaces 对立对                                  → 两次写入，第二次带 similar
sidecar 未 ready                                      → 降级 Jaccard，add 仍可用
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `server.py` | HTTP：`/health`、`/embed`、`/relate`；`--prepare` / `--warm` |
| `relate.py` | CrossEncoder NLI（softmax 概率） |
| `prefetch.py` | 下载 relate 权重（跳过 ONNX；mirror-safe） |
| `run.sh` | 启动；`prepare` 子命令 |
| `calibrate.py` | T/C/E 网格 + yaml 推荐 |
| `eval_gate.py` | 60 对 HTTP 闸回归 |
| `samples.py` | 标定样本 |
| `scripts/setup.sh` | venv + `requirements.txt` |
| `imprint.plugin.json` | 插件 manifest + configSchema 默认值 |
