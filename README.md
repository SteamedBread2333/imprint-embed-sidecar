# imprint-embed-sidecar

语义查重插件：给 imprint 的 `add` 加向量 + NLI 分数，拦住词面 Jaccard 抓不到的语义重复，并把对立政策放进 `similar` advisory。

Python 常驻 sidecar + Go 薄客户端；**sidecar 只出分，duplicate/advisory 决策在 imprint（Go + `~/.imprint/embed.yaml`）**。

## 快速开始

配置写在 **`~/.imprint/embed.yaml`**（全局一份，`package` 指向本仓库绝对路径）。`imprint embed prepare` / `imprint plugin start embed` **可在任意 cwd** 运行。

**Imprint Console** 用户可在 App 内自动 clone 本仓库到应用数据目录并写入 `embed.yaml`，无需手改项目 `imprint.yaml`。

```bash
# 1) 本仓库
./scripts/setup.sh
./run.sh prepare          # 或 imprint embed prepare

# 2) ~/.imprint/embed.yaml → package: /abs/path/to/imprint-embed-sidecar
imprint embed prepare
imprint plugin start embed

curl --noproxy '*' http://127.0.0.1:4174/health   # 需 "ready": true
```

`run.sh` 默认 `--warm`；跳过 warm：`IMPRINT_EMBED_NO_WARM=1 ./run.sh`。

国内网络：Go 启动 sidecar 时会设 `HF_HUB_DISABLE_XET=1`，并在 huggingface.co 不可达时使用 `HF_ENDPOINT=https://hf-mirror.com`。手动下载可 `export IMPRINT_HF_ENDPOINT=https://hf-mirror.com`。

## ~/.imprint/embed.yaml 示例

```yaml
enabled: true
package: /path/to/imprint-embed-sidecar
config:
  port: 4174
  duplicate_threshold: 0.65
  relate_contradiction_min: 0.40
  relate_entailment_min: 0.35
  relate_bidirectional: true
  duplicate_high_cosine_block: 0.80   # 0 = 关闭；NLI neutral 时高 cos 仍拒写
  timeout_seconds: 5
  # relate_model: off   # 省略则用 imprint.plugin.json 的 schema 默认；off 关闭 /relate
```

embed 与 relate 的模型 id 只写在本仓库 `imprint.plugin.json` 的 `configSchema`。进程内覆盖用 `IMPRINT_EMBED_MODEL` 或 `server.py --model`。`GET /health` 与 `POST /embed` 响应里的 `model` 是当前加载的 id。`embed.yaml` 的 `model` 键不参与选择。

归属：向量在 vault `rule_vectors` · 权重 `~/.cache/imprint/models/` · venv `~/.imprint/plugins/embed/venv` · sidecar 日志 `~/.imprint/state/plugins/embed/sidecar.log`。

## HTTP 接口

```jsonc
// GET /health
{ "ok": true, "ready": true, "embed": {...}, "relate": {"enabled": true, "model": "..."} }

// POST /embed   {"texts": ["..."]}   // ≤64 条，≤4000 字符/条
{ "vectors": [[...]], "dim": 512, "model": "..." }

// POST /relate   {"pairs": [["claim a", "claim b"], ...]}   // ≤64 对
{ "model": "...", "labels": ["contradiction","entailment","neutral"], "results": [{...}] }
```

Go 判定顺序（阈值见 embed.yaml）：**contradiction → entailment → high cosine block**；`relate_bidirectional` 时在 Go 侧打两次 `/relate` 并合并 max 分数。

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
