# scripts

| 脚本 | 作用 |
| --- | --- |
| `setup.sh` | 创建 `~/.imprint/plugins/embed/venv` 并安装 `requirements.txt` |

标定与回归在仓库根目录：

- `calibrate.py` — 本地加载模型，扫 `duplicate_threshold` / `relate_*_min`
- `eval_gate.py` — 对已运行的 sidecar 做 60 对语义闸检查
- `./run.sh prepare` — 等同 `server.py --prepare`（下载 + warm，不起 HTTP）

imprint 仓库还有 `scripts/embed-eval-batch.py`（真实 `add` 路径）。
