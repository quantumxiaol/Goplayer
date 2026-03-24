# Goplayer
一个基于 **PyQt6** 的围棋程序，支持：

- GUI 对弈（人人 / 人机 / 随机 / 挑战本地 AlphaZero）
- 9/13/19 路棋盘切换
- 基于 `GoEnv` 的 AlphaZero 风格自博弈训练

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
- `新游戏 / 判断胜负 / 悔棋`

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

快速冒烟（先确认流程能跑通）：

```bash
uv run python scripts/train.py --board-size 9 --iterations 5 --games-per-iteration 2 --train-steps-per-iteration 2
```

## 3. 训练输出

- 模型权重默认保存到 `checkpoints/`
- 训练日志默认保存到 `logs/`
- `logs/train_metrics.csv`：每轮指标
- `logs/events.out.tfevents.*`：TensorBoard 事件（启用后）

TensorBoard 查看：

```bash
uv run tensorboard --logdir logs
```

## 4. 训练配置（`.env`）

```bash
RL_DEVICE=auto
RL_CHECKPOINT_DIR=checkpoints
RL_LOG_DIR=logs
RL_TENSORBOARD=0
ALPHAZERO_CHECKPOINT_PATH=checkpoints/best_model.pth
```

说明：

- `RL_DEVICE` 支持 `auto / cpu / cuda / mps`
- 命令行参数优先级高于 `.env`
- `ALPHAZERO_CHECKPOINT_PATH` 用于 GUI 的「挑战 AlphaZero」模式加载权重
- `AlphaZero` 模式会检查 checkpoint 与当前棋盘大小是否匹配（不匹配会拒绝加载并回退）

## 5. 棋盘大小与模型关系

不同棋盘大小需要分别训练并分别保存模型。

建议命名：

- `checkpoints/best_model_9.pth`
- `checkpoints/best_model_13.pth`
- `checkpoints/best_model_19.pth`

# 规则说明

- 终局：双 Pass
- 禁入点：Suicide Rule
- 打劫：Positional Superko
- 计分：Tromp-Taylor（白方含贴目 Komi）

更多规则见：[GoRules.md](GoRules.md)
