# imprint-embed-plugin

语义查重插件：给 imprint 的 `add` 加一条向量通路，拦住词面 Jaccard 抓不到的语义重复。

形态是 Python 常驻 sidecar + Go 侧薄客户端，复用 imprint 已有的插件机制（`imprint.plugin.json`、端口分配、`up/down` 生命周期）。**默认关闭，缺失即降级**，插件挂掉时 `add` 照常工作。

## 快速开始

```bash
./scripts/setup.sh          # 建 venv、装 fastembed、下载模型权重（约 130MB）
./run.sh                    # 常驻监听，端口取 IMPRINT_PLUGIN_PORT（默认 4174）
curl --noproxy '*' http://127.0.0.1:4174/health
```

在项目的 `imprint.yaml` 启用：

```yaml
plugins:
  embed:
    enabled: false          # 默认关；置 true 启用
    package: ../imprint-embed-plugin
    config:
      port: 4174
      model: bge-small-zh-v1.5
      timeout_seconds: 2
      duplicate_threshold: 0.70   # 由 calibrate.py 标定，勿拍脑袋改
```

归属划分（已确认，别挪）：向量数据跟项目（`.imprint/vault.db` 侧表 `rule_vectors`）· 模型权重 `~/.cache/imprint/models/` · venv `~/.imprint/plugins/embed/venv`。

## 接口

```jsonc
// GET /health
{ "ok": true, "model": "BAAI/bge-small-zh-v1.5", "dim": 512, "cache_dir": "..." }

// POST /embed   {"texts": ["..."], "model": "..."}
{ "vectors": [[0.01, -0.02, ...]], "dim": 512, "model": "..." }
```

约束：单次 ≤ 64 条文本、单条 ≤ 4000 字符；Go 侧超时 2s，失败静默降级。

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

因此单靠余弦做硬拒不成立。本插件采用**双信号**：

1. 余弦 ≥ 阈值
2. `PolarityConflict(claim, existingClaim)` 为假 —— 极性守卫：否定词不对称（never / do not / without / instead of / 不要 / 禁止…）或命中反义词对（tabs↔spaces、wrap↔bare、snake_case↔camelCase…），或同词集合词序互换

| 组合 | 行为 |
| --- | --- |
| 高余弦 + 无极性冲突 | **硬拒**，返回 candidates + hint（reinforce / supersede / reword） |
| 高余弦 + 有极性冲突 | **放行**，写入成功但结果带 `similar[]` advisory，裁决权归 LLM |
| 低余弦 | 放行 |

极性守卫实测：命中 20 组反义样本中的 **18 对**，把冲突误杀率从 **70% 降到 10%**（阈值 0.65–0.75 区间），无关误杀率 **0%**。

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
| `server.py` | HTTP sidecar（`/health`、`/embed`），懒加载模型、线程安全 |
| `run.sh` | 启动入口，解析 venv，`IMPRINT_PLUGIN_PORT` 优先 |
| `calibrate.py` | 阈值标定，输出分布与扫描矩阵 |
| `samples.py` | 三组各 20 对标定样本 |
| `polarity.py` | 极性守卫（与 `pkg/imprint/polarity.go` 保持同步） |
| `scripts/setup.sh` | venv 与依赖安装 + 模型加载自检 |
