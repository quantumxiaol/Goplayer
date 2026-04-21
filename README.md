# Goplayer
一个基于 **PyQt6** 的围棋程序，支持：

- GUI 对弈（人人 / 人机 / 随机 / 挑战本地 AlphaZero）
- 9/13/19 路棋盘切换
- 基于 `GoEnv` 的 AlphaZero 风格自博弈训练
- `frontend/` 下的 React + Canvas + TypeScript 前端展示页

# 前端展示（GitHub Pages / 本地预览）

`frontend/` 是一个独立的前端演示页，主要用于展示围棋规则与后续接主项目。

当前已实现：

- 9/13/19 路棋盘切换
- 双人本地对弈
- 气、提子、自杀禁着、打劫/同形禁着
- Pass、悔棋、面积计分、手顺显示
- 9 路浏览器侧 ONNX 单步落子建议

当前限制：

- 只有 `9x9` 已训练并导出 ONNX
- `13x13` / `19x19` 暂无模型，前端会明确显示为“未训练”
- 前端 AI 目前是“单步建议”，还不是完整 MCTS 自动对弈

## 1. 本地启动前端

```bash
cd frontend
pnpm install
pnpm dev
```

打开：

```text
http://localhost:5173/goplayer/
```

说明：Vite 的 `base` 已固定为 `/goplayer/`，以匹配 GitHub Pages 项目页路径。

## 2. 导出 ONNX 模型

将训练好的 checkpoint 导出到前端静态目录：

```bash
uv sync --extra rl
uv run --extra rl python scripts/export_onnx.py --checkpoint checkpoints/9x9/best_model.pth
```

默认输出到：

- `frontend/public/models/9x9/goplayer_v1.onnx`
- `frontend/public/models/9x9/goplayer_v1.json`

后续如果训练出 13/19 路模型，也应分别导出到各自目录：

- `frontend/public/models/13x13/`
- `frontend/public/models/19x19/`

## 3. GitHub Pages 部署

该前端作为项目页部署到：

```text
https://quantumxiaol.github.io/goplayer/
```

需要在 **Goplayer 仓库本身** 启用 Pages，并选择 `GitHub Actions` 作为发布来源。

# 普通用户：GUI 下棋

## 1. 安装（仅 GUI）

```bash
uv sync --extra gui
```

如果你要玩「人机对弈（OpenAI API）」模式，再配置 `.env`（可参考 `.env.template`）：

```bash
OPENAI_MODEL="your-model"
OPENAI_API_KEY="your-key"
OPENAI_BASE_URL="your-base-url"
```

## 2. 启动 GUI

```bash
uv run python ItisMyGo.py
```

## 3. 在界面里怎么用

- `对弈模式`
- `人人对弈`：双人本地落子
- `人机对弈`：调用 OpenAI API 的 AI
- `挑战 AlphaZero`：加载本地训练权重对弈
- `AlphaZero vs AlphaZero`：双 AI 观战
- `随机对弈`：双方随机落子

- `棋盘大小`
- `9路 / 13路 / 19路` 可随时切换

- `选项`
- `新游戏 / 停一手 / 认输 / 判断胜负 / 悔棋`

# 强化学习训练（AlphaZero 风格）

## 1. 安装训练依赖

仅训练（无 GUI）：

```bash
uv sync --extra rl
```

完整环境（GUI + RL）：

```bash
uv sync --extra all
# 或
uv sync --extra gui --extra rl
# 或
uv sync --all-extras
```

## 2. 开始训练

示例（9 路）：

```bash
uv run python scripts/train.py --board-size 9 --iterations 200
```

9 路推荐（更稳，但更慢）：

```bash
uv run python scripts/train.py \
  --board-size 9 \
  --iterations 400 \
  --games-per-iteration 16 \
  --num-simulations 160 \
  --min-moves-before-pass 50 \
  --run-name bs9_stable_v1 \
  --tensorboard
```

如果传 `--checkpoint-dir`，它是根目录，实际会保存到：
`<checkpoint-dir>/<board_size>x<board_size>/`

快速冒烟（先确认流程能跑通）：

```bash
uv run python scripts/train.py --board-size 9 --iterations 5 --games-per-iteration 2 --train-steps-per-iteration 2
```

常用参数说明：

- `--iterations`：训练轮数
- `--games-per-iteration`：每轮自博弈局数
- `--num-simulations`：每步 MCTS 模拟次数（越大越强，但越慢）
- `--min-moves-before-pass`：前 N 手不允许 pass（减少过早双 pass）
- `--run-name`：本次训练日志目录名

## 3. 训练输出

- 模型权重默认保存到 `checkpoints/{board_size}x{board_size}/`
  - 例如：9路保存到 `checkpoints/9x9/`
- 训练日志根目录默认是 `logs/`
- 每次训练会自动落到独立 run 目录：
  - `logs/{board_size}x{board_size}/{run_name}/train_metrics.csv`
  - `logs/{board_size}x{board_size}/{run_name}/events.out.tfevents.*`（启用 TensorBoard 时）

TensorBoard 查看：

```bash
uv run tensorboard --logdir logs
```

## 4. 训练配置（`.env`）

```bash
RL_DEVICE=auto
RL_CHECKPOINT_DIR=checkpoints
RL_LOG_DIR=logs
RL_RUN_NAME=
RL_TENSORBOARD=0
RL_MIN_MOVES_BEFORE_PASS=30
ALPHAZERO_CHECKPOINT_DIR=checkpoints
# ALPHAZERO_CHECKPOINT_PATH=/abs/path/to/model.pth
ALPHAZERO_MIN_MOVES_BEFORE_PASS=18
```

说明：

- `RL_DEVICE` 支持 `auto / cpu / cuda / mps`
- 命令行参数优先级高于 `.env`
- 训练时 `--checkpoint-dir` 是根目录，程序会自动写入 `{root}/{size}x{size}/`
- 训练时 `--log-dir` 是根目录，程序会自动写入 `{root}/{size}x{size}/{run_name}/`
- `--run-name` / `RL_RUN_NAME` 可用于手动命名本次训练日志目录（不填则自动时间戳）
- GUI 默认按棋盘大小从 `ALPHAZERO_CHECKPOINT_DIR/{size}x{size}/best_model.pth` 自动加载
- `ALPHAZERO_CHECKPOINT_PATH` 可选，设置后会覆盖自动路径
- `RL_MIN_MOVES_BEFORE_PASS`：训练时前 N 手不允许 pass（除非无合法落子），减少“白方吃komi+早早双pass”塌缩
- `ALPHAZERO_MIN_MOVES_BEFORE_PASS`：GUI 对弈时前 N 手不允许 pass（除非无合法落子）
- `AlphaZero` 模式会检查 checkpoint 与当前棋盘大小是否匹配（不匹配会拒绝加载并回退）

## 5. 棋盘大小与模型关系

不同棋盘大小需要分别训练并分别保存模型。

建议目录结构：

- `checkpoints/9x9/best_model.pth`
- `checkpoints/13x13/best_model.pth`
- `checkpoints/19x19/best_model.pth`

# 规则说明

- 终局：双 Pass
- 禁入点：Suicide Rule
- 打劫：Positional Superko
- 计分：Tromp-Taylor（白方含贴目 Komi）

更多规则见：[GoRules.md](GoRules.md)
