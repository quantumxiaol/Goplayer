# GoPlay · Goplayer

一个围棋学习与实验项目：用 **React + Canvas** 在浏览器中对弈，用 **PyQt6** 运行桌面棋盘，用 **PyTorch + MCTS** 探索 AlphaZero 风格的自博弈训练。

支持 9 / 13 / 19 路棋盘。网页端已附带 **9 路 ONNX 模型**，可直接在浏览器中请求单步落子建议。

[快速开始](#快速开始) · [桌面版](#桌面版) · [训练与模型导出](#训练与模型导出) · [规则与限制](#规则与限制)

## 界面预览

### 网页版 · 9 路对弈与 AI 建议

以下为本地实际运行截图，展示落子、最近手顺、面积计分和 ONNX 建议。AI 建议需要手动落子，不会自动代下。

![GoPlay 网页版：9 路对局、AI 建议与局面信息](docs/screenshots/web-9x9-ai.png)

<details>
<summary>桌面版界面预览</summary>

![PyQt 围棋桌面版旧版界面](png/interface.png)

该图展示桌面棋盘外观；当前版本还提供棋盘大小切换与本地 AlphaZero 模式，菜单以实际运行界面为准。

</details>

## 功能一览

| 功能 | 网页版 | 桌面版 / Python |
| --- | --- | --- |
| 9 / 13 / 19 路棋盘 | 支持 | 支持 |
| 本地双人对弈 | 支持 | 支持 |
| 提子、自杀禁着、同形禁着 | 支持 | 支持 |
| Pass、悔棋、面积计分 | 支持 | 支持 |
| AI 建议 | 9 路 ONNX 单步建议 | 本地网络 + MCTS 对弈 |
| 外部模型 API 对弈 | 无 | 支持 OpenAI 兼容接口 |
| 随机对弈 / 双 AI 观战 | 无 | 支持 |
| 自博弈训练 | 无 | PyTorch 策略 / 价值双头网络 |

13 / 19 路目前还没有训完，页面会显示“未训练”。项目定位为学习与实验，尚未提供棋力评级或标准对局评测。

## 快速开始

### 网页版

环境：**Node.js 22.12+、pnpm 10**。以下命令从项目根目录执行：

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

访问 **http://localhost:5173/Goplayer/**，实际端口以终端输出为准。

1. 点击棋盘交叉点落子，黑先、双方交替。
2. 点击 **AI 建议**，首次使用时加载模型与 WASM 运行时；仅 9 路可用。
3. 点击 **Pass** 停一手，连续两次 Pass 后按当前盘面计分。
4. **悔棋** 撤销上一手；**重开**或切换棋盘尺寸会清空当前对局。

检查和构建：

```bash
pnpm lint
pnpm build
pnpm preview
```

构建产物位于 `frontend/dist/`；预览地址通常为 `http://localhost:4173/Goplayer/`。

### GitHub Pages

仓库已有 [部署工作流](.github/workflows/deploy-gh-pages.yml)。

## 桌面版

环境：**Python 3.12+、uv**。在项目根目录运行：

```bash
uv sync --extra gui
uv run --extra gui python ItisMyGo.py
```

可选模式：**人人对弈、人机对弈、挑战 AlphaZero、AlphaZero vs AlphaZero、随机对弈**。通过菜单切换棋盘尺寸，或执行新游戏、停一手、认输、判断胜负与悔棋。

### 外部 API 对弈

在项目根目录创建 `.env`，参考 [.env.template](.env.template) 填写服务商配置：

```dotenv
OPENAI_MODEL="your-model"
OPENAI_API_KEY="your-key"
OPENAI_BASE_URL="https://your-provider.example/v1"
```

此模式会把棋盘状态发送给所配置的服务商，并可能产生 API 费用。网页端与本地 AlphaZero 模式不使用这些配置。

### 本地 AlphaZero 对弈

```bash
uv sync --extra gui --extra rl
uv run --extra gui --extra rl python ItisMyGo.py
```

默认按棋盘尺寸加载 `checkpoints/{size}x{size}/best_model.pth`，例如 `checkpoints/9x9/best_model.pth`。该目录被 Git 忽略，克隆仓库后需要自行训练或准备兼容权重；网页附带的 `.onnx` 文件不能直接替代桌面版 `.pth`。

缺少依赖、权重或权重尺寸不匹配时，当前实现会在终端输出原因，并回退到随机合法落子。

## 训练与模型导出

### 1. 安装训练依赖

```bash
uv sync --extra rl
```

### 2. 验证训练流程

以下配置仅用于快速检查自博弈、反向传播和权重保存，不用于评估棋力。独立输出目录可避免覆盖已有模型：

```bash
uv run --extra rl python scripts/train.py \
  --board-size 9 --iterations 1 \
  --games-per-iteration 1 --num-simulations 2 \
  --max-moves 6 --batch-size 2 \
  --train-steps-per-iteration 1 --save-interval 1 \
  --device cpu --checkpoint-dir checkpoints/smoke \
  --log-dir logs/smoke --run-name smoke --no-tensorboard
```

### 3. 运行自博弈训练

```bash
uv run --extra rl python scripts/train.py \
  --board-size 9 --iterations 200 --tensorboard
```

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--games-per-iteration` | `8` | 每轮自博弈局数 |
| `--num-simulations` | `80` | 每步 MCTS 模拟次数；增大后计算开销也增加 |
| `--batch-size` | `128` | 回放训练批大小；样本不足时跳过更新 |
| `--min-moves-before-pass` | `30` | 前 N 手限制 Pass；可由 `.env` 覆盖 |
| `--save-interval` | `10` | 每 N 轮保存模型 |
| `--device` | `auto` | `auto / cpu / cuda / mps`；可由 `.env` 覆盖 |
| `--run-name` | 时间戳 | 区分训练日志目录 |

完整参数：`uv run --extra rl python scripts/train.py --help`。

输出结构：

```text
checkpoints/9x9/
├── model_v10.pth
└── best_model.pth
logs/9x9/<run_name>/
├── train_metrics.csv
└── events.out.tfevents.*   # 启用 TensorBoard 时生成
```

`--checkpoint-dir` 与 `--log-dir` 都是根目录，脚本会自动追加棋盘尺寸。`best_model.pth` 根据保存轮次的平均训练损失选取，未经过对局胜率选拔；相同权重目录中的同名文件会被后续训练覆盖，`--run-name` 只隔离日志。

```bash
uv run --extra rl tensorboard --logdir logs
```

训练配置还可通过 `.env` 中的 `RL_DEVICE`、`RL_CHECKPOINT_DIR`、`RL_LOG_DIR`、`RL_RUN_NAME`、`RL_TENSORBOARD` 和 `RL_MIN_MOVES_BEFORE_PASS` 设置，命令行参数优先。桌面版使用 `ALPHAZERO_CHECKPOINT_DIR`、`ALPHAZERO_CHECKPOINT_PATH` 与 `ALPHAZERO_MIN_MOVES_BEFORE_PASS`，详见 [.env.template](.env.template)。

### 4. 导出到浏览器

```bash
uv run --extra rl python scripts/export_onnx.py \
  --checkpoint checkpoints/9x9/best_model.pth
```

默认生成 `frontend/public/models/9x9/goplayer_v1.onnx` 和同名 `.json` 元数据，导出时会执行 ONNX 格式检查。如果训练使用了自定义网络宽度或深度，导出时需要传入匹配的 `--num-channels` 和 `--num-res-blocks`。

不同棋盘尺寸需要分别训练。导出 13 / 19 路模型后，还需要修改 [modelConfig.ts](frontend/src/game/modelConfig.ts) 中对应尺寸的 `trained`、`modelPath` 和说明文字，前端才会启用 AI 按钮。

## 规则与限制

- 落子后先提掉无气敌块，再判断己方是否有气；禁止自杀着。
- 使用棋盘历史签名检查 **Positional Superko（全局同形禁着）**。
- 连续两次 Pass 终局，按盘上棋子与单色围住的空点进行面积计分。不会自动识别或移除死子，应先完成争议区域的对弈。
- 默认白方贴目：9 路 **5.5**、13 路 **2.0**、19 路 **7.5**。
- 网页 AI 使用策略网络单次推理，不含 MCTS。界面的“置信度”是合法候选着法中的策略概率，不代表胜率；价值输出也未经棋力校准。
- 网页对局保存在内存中，刷新页面会丢失，尚无棋谱导入 / 导出与在线联机功能。

规则背景见 [GoRules.md](GoRules.md)。

## 项目结构

```text
ItisMyGo.py               桌面版入口
src/Goplayer/            Python 规则环境、棋盘与棋手
src/rl/                  编码器、策略价值网络、MCTS、回放缓冲
scripts/train.py         自博弈训练
scripts/export_onnx.py   模型导出
frontend/src/            React 页面、Canvas 棋盘、TypeScript 规则引擎
frontend/public/models/ 浏览器模型与元数据
docs/screenshots/        界面截图
tests/                   Python 规则、AI 与 Qt 线程回归测试
frontend/tests/          前端 AI 异步请求回归测试
```

## 回归测试

前端（在 `frontend/` 中运行）：

```bash
pnpm test
pnpm lint
pnpm build
```

Python（在项目根目录运行，需要 GUI 与训练依赖）：

```bash
uv run --extra gui --extra rl python -m unittest discover -s tests -v
```

测试覆盖 Pass 悔棋、API 坐标转换、MCTS 合法动作、Qt 工作线程生命周期，以及网页端重开 / 切盘 / 悔棋时的过期 AI 请求。外部 API 使用模拟响应，测试不会请求真实服务。
