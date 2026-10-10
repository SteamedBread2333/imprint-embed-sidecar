# imprint-embed-sidecar

语义查重插件：给 imprint 的 `add` 加一条向量通路，拦住词面 Jaccard 抓不到的语义重复。

形态是 Python 常驻 sidecar + Go 侧薄客户端，复用 imprint 已有的插件机制（`imprint.plugin.json`、端口分配、`up/down` 生命周期）。**默认关闭，缺失即降级**，插件挂掉时 `add` 照常工作。

## 快速开始

```bash
./scripts/setup.sh          # 建 venv、pip install 本包（embed + relate 依赖）
./run.sh                    # 常驻监听，端口取 IMPRINT_PLUGIN_PORT（默认 4174）
imprint-embed-sidecar --print-plugin-dir   # 填 imprint.yaml 的 plugins.embed.package
curl --noproxy '*' http://127.0.0.1:4174/health
```

在项目的 `imprint.yaml` 启用：

```yaml
plugins:
  embed:
    enabled: false          # 默认关；置 true 启用
    package: ../imprint-embed-sidecar
    config:
      port: 4174
      model: bge-small-zh-v1.5
      timeout_seconds: 2
      duplicate_threshold: 0.70
      relate_model: cross-encoder/nli-deberta-v3-small
      relate_contradiction_min: 0.50   # imprint.yaml；由 calibrate.py 推荐
      relate_entailment_min: 0.50
      timeout_seconds: 4
```

归属划分（已确认，别挪）：向量数据跟项目（`.imprint/vault.db` 侧表 `rule_vectors`）· 模型权重 `~/.cache/imprint/models/` · venv `~/.imprint/plugins/embed/venv`。

## 接口

```jsonc
// GET /health
{ "ok": true, "model": "...", "dim": 512, "embed": {...}, "relate": {"enabled": true, "model": "..."} }

// POST /embed   {"texts": ["..."]}
{ "vectors": [[...]], "dim": 512, "model": "..." }

// POST /relate   {"pairs": [["claim a", "claim b"], ...]}
{ "model": "...", "labels": ["contradiction","neutral","entailment"], "results": [{...}] }
```

Sidecar **只返回分数**；duplicate / advisory 阈值在 **imprint.yaml**（Go 读）。单次 ≤ 64 条/对，单条 ≤ 4000 字符；失败时 Go 降级到 Jaccard。

### 多 MCP / 多窗口并发

HTTP 使用 `ThreadingHTTPServer`，多个 Cursor 窗口会同时连 **同一个** sidecar。ONNX 推理在进程内 **串行**（`_infer_lock`）：请求会在 sidecar 里 **排队**，不会并行抢同一份 model handle（避免不稳定或更慢的超时）。

若排队导致 Go 端 2s 超时，会 **降级到 Jaccard**，写入/召回不失败。缓解方式：

- 在 `imprint.yaml` 把 `plugins.embed.config.timeout_seconds` 略调高（例如 4–5s）；
- 减少同时 `add`/`find`+query 的风暴（同一 claim 在 Go client 内有 LRU 缓存，重复文本不再打 HTTP）；
- 后续可做 sidecar 侧 **微批次合并**（把几毫秒内的多条 texts 合成一次 `embed()`）。

## 阈值标定（最重要，改模型必重跑）

```bash
./run.sh --warm &          # 或让 calibrate.py 自己加载模型
python3 calibrate.py
```

`calibrate.py` 用三组各 20 对样本（真重复 / 反义冲突 / 无关）跑余弦分布，输出阈值扫描矩阵并给出推荐值。

### 实测结果（bge-small-zh-v1.5，2026-09-23）

| 组 | min | p25 | median | p75 | max | mean |
| --- | --- | --- | --- | --- | --- | --- |
| 真重复 | 0.652 | 0.778 | 0.823 | 0.893 | 0.932 | 0.819 |
| 反义冲突 | 0.492 | 0.690 | 0.753 | 0.786 | **1.000** | 0.749 |
| 无关 | 0.432 | 0.484 | 0.528 | 0.549 | 0.644 | 0.520 |

**关键结论：余弦分不清「语义重复」与「反义冲突」。** 反义组最高 1.000（"Prefer tabs over spaces" ↔ "Prefer spaces over tabs" 词序互换即满分）——余弦测的是话题相关度，而反义恰恰话题最相关。这与早先字面 Jaccard 的失败（真重复 0.383 < 反义 0.700）同源。

因此单靠余弦做硬拒不成立。imprint（Go）采用 **余弦 + `/relate` NLI 分数**（sidecar 本地 cross-encoder，无词典硬编码）：

| 组合 | 行为 |
| --- | --- |
| 高余弦 + 高 entail + 低 contradiction | **硬拒** duplicate |
| 高余弦 + 高 contradiction | **放行**，`similar[]` advisory |
| relate 不可用 | 仅余弦/Jaccard（可能误拒反义对，与 embed 降级相同） |

**阈值取 0.70**：这是能拦住验收对的最高值。验收对
`Go exported identifiers must use PascalCase` ↔ `Exported things use Pascal Case naming`
余弦 **0.711**（Jaccard 仅 0.38，词面完全抓不到）。再高就漏拦，再低则冲突误杀上升。

> 维度实测为 **512**，不是早期笔记里的 384 —— 永远以 `/health` 与 `/embed` 返回的 `dim` 为准。

## 验收

```
add "Go exported identifiers must use PascalCase"    → 成功
add "Exported things use Pascal Case naming"         → 被拒，candidates 指向上一条
add "Never use tabs …"（库里有 "Use tabs …"）         → 放行（advisory，余弦 0.927）
插件进程 kill 后 add                                   → 成功（降级）
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `server.py` | HTTP（`/health`、`/embed`、`/relate`） |
| `relate.py` | 本地 cross-encoder 打分 |
| `run.sh` | 启动入口 |
| `calibrate.py` | 余弦 + relate 分布，输出推荐 yaml 片段 |
| `samples.py` | 标定样本 |
| `scripts/setup.sh` | venv + `pip install` |
| `pyproject.toml` | 包与 `imprint-embed-sidecar` CLI |
