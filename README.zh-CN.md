# imprint-embed-sidecar

[English](./README.md) · **中文**

为 **imprint** 提供本地 HTTP：**向量 embed** 与 **NLI 句对分数**。是否 duplicate / similar advisory 由 imprint（Go + `~/.imprint/embed.yaml`）判定；本仓库**只出分**。

## 快速开始

配置在 **`~/.imprint/embed.yaml`**（`package` 填本仓库绝对路径）。`imprint embed prepare`、`imprint plugin start embed` **任意目录**可执行。

```bash
# 需要 Python ≥ 3.11（macOS 自带 python3 常为 3.9）
PYTHON=/opt/homebrew/bin/python3.13 ./scripts/setup.sh   # 按需

./scripts/setup.sh
./run.sh prepare                    # 下载并预热模型；或 imprint embed prepare

# ~/.imprint/embed.yaml → package: /abs/path/to/imprint-embed-sidecar
imprint plugin start embed

curl --noproxy '*' http://127.0.0.1:4174/health   # 应看到 "ready": true
```

Imprint Console 可自动 clone 本仓库并写入 `embed.yaml`。

**可选：** 启动不预热模型：`IMPRINT_EMBED_NO_WARM=1 ./run.sh`。  
**国内 / HF 慢：** prepare 前 `export IMPRINT_HF_ENDPOINT=https://hf-mirror.com`。

## 配置示例

```yaml
enabled: true
package: /path/to/imprint-embed-sidecar
config:
  port: 4174
  duplicate_threshold: 0.65
  relate_contradiction_min: 0.40
  relate_entailment_min: 0.35
  relate_bidirectional: true
  duplicate_high_cosine_block: 0.80   # 0 关闭；NLI neutral 时仍用高 cos 拒写改述重复
  timeout_seconds: 5
  # relate_model: off                 # 省略用默认；off 则关闭 POST /relate
```

默认模型 ID 见 `imprint.plugin.json`。运行时可用 `IMPRINT_EMBED_MODEL` 或 `server.py --model` 覆盖。

**路径：** 权重 `~/.cache/imprint/models/` · venv `~/.imprint/plugins/embed/venv` · 日志 `~/.imprint/state/plugins/embed/sidecar.log` · 向量在 vault（`rule_vectors`）。

## HTTP 接口

| 方法 | 请求体 | 限制 |
|------|--------|------|
| `GET /health` | — | `ready`、embed/relate 能力 |
| `POST /embed` | `{"texts":["…"]}` | ≤64 条，每条 ≤4000 字 → `vectors`、`dim`、`model` |
| `POST /relate` | `{"pairs":[["a","b"],…]}` | ≤64 对 → `results[]`（contradiction / neutral / entailment） |

imprint 语义闸顺序（cos ≥ 阈值后）：**contradiction** → **entailment** → **高 cos 兜底**。双向 relate 在 Go 侧合并 max 分数。

sidecar 未 ready → imprint 降级 Jaccard，`add` 仍可用。

## 标定与回归

| 脚本 | 用途 |
|------|------|
| `calibrate.py` | 60 对样本扫 T/C/E，输出建议 yaml |
| `eval_gate.py` | 对已运行 sidecar 复现 Go 闸（`--bidirectional`、`--high-cos-block 0.80`） |

更换 embed 或 relate 模型后：`calibrate.py` → `eval_gate.py` → imprint 仓库 `embed-eval-batch.py`。

## 文件

| 路径 | 作用 |
|------|------|
| `server.py` | `/health`、`/embed`、`/relate` |
| `relate.py` | CrossEncoder NLI |
| `run.sh` | 启动 sidecar；`prepare` 子命令 |
| `scripts/setup.sh` | 创建 venv、安装依赖 |
