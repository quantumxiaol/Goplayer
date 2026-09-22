# 训练归档、固定数据对照与 Pass 输入实验

以下训练命令在实验室 GPU 机器执行。本机只做固定样例、推理、导出和模拟更新的测试。新增功能不意味着已经验证模型收敛或棋力提升。

## 1. 先生成一份可复用的数据

保持旧输入与原来的搜索、学习率设置，单独采集一份基线实验：

```bash
uv sync --extra rl
uv run --extra rl python scripts/train.py \
  --config configs/train-13x13.json \
  --run-name bs13_archive_v3 --iterations 10
```

与旧版相比，这次会留出约 10% 的完整对局用于验证，因此训练缓冲区的大小不再等于所有对局手数之和。比例通过 `--validation-fraction` 控制，按游戏 ID 与种子做稳定划分，不拆开同一盘棋。验证局不进入训练回放池；数据增强只用于训练。

每批自博弈结束后，逐局落盘：

```text
logs/13x13/bs13_archive_v3/data/
├── dataset.json
├── iter-000001-game-0001/
│   ├── game.sgf
│   ├── game.json
│   └── samples.npz
└── ...
```

- `game.sgf`：所有落子和 Pass，含尺寸、贴目、已完成对局的结果。SGF 坐标为列后行，Pass 写作空坐标。
- `game.json`：逐手 `(row, col, color)`、终局棋盘、比分、终止原因、数据划分和样本数。行列从 0 开始，`(-1, -1)` 表示 Pass。
- `samples.npz`：`states [N,C,S,S]`、`policies [N,S*S+1]`、`values [N]`。保存未经增强的输入和完整访问概率，状态压缩为二值 `uint8`，加载后转回 `float32`；不使用 pickle。
- `dataset.json`：格式版本、棋盘大小、贴目、输入版本及划分参数。

截断局仍保存棋谱和终局棋盘，样本数组为空，不参与训练和验证。完整目录通过原子重命名发布，中断产生的 `.pending-*` 目录不会被加载。正在进行、尚未返回的自博弈批次不能恢复。

内存中的训练回放池默认最多 100,000 个局面，验证池最多 20,000 个；磁盘归档保留所有已保存对局，不随回放池淘汰。重新加载会按生成顺序重建指定容量的两个池。评测棋谱另存于 `evaluation_sgf/`，评测 JSONL 同样包含逐手记录和终局棋盘。

旧实验 `bs13_probe_v2` 没有保存这些样本，原来的统计日志无法补出搜索概率，不能用于下面的固定数据对照。

## 2. 在同一份固定数据上比较 20 / 80 次更新

```bash
uv run --extra rl python scripts/compare_updates.py \
  --data-dir logs/13x13/bs13_archive_v3/data \
  --checkpoint checkpoints/13x13/bs13_archive_v3/initial_model.pth \
  --output-dir logs/13x13/bs13_compare_v3 \
  --updates 20 80 --rounds 10 \
  --batch-size 256 --learning-rate 0.0003 --device cuda
```

两条分支从相同初始权重和全新的 Adam 开始，使用同一个冻结训练池、同一个验证池、同样的采样与增强随机序列前缀。10 轮结束时分别更新 200 / 800 次。这里的“轮”是固定数据上的更新分组，不会追加自博弈训练数据，也不是重新执行在线训练的十轮数据生成过程。

默认读取最近 100,000 个训练局面和 20,000 个验证局面，可通过 `--buffer-size`、`--validation-buffer-size` 调整。两个分支读取同一份内存快照，运行期间新增的源文件不会进入当前对照。

每条分支在更新前和每轮更新后计算固定验证集指标，最后保存权重，并对同一初始模型进行交换执色评测。搜索预算、贴目、Pass 限制等取自初始 checkpoint 的 `search_config`，不会因为更新次数不同而改变。默认评测 20 局，使用相同开局种子；它只用于粗略判断，不能当作棋力评级。

```text
logs/13x13/bs13_compare_v3/
├── comparison.json       # 参数、数据成员/校验值、初始权重校验值
├── results.json          # 两条分支的最终指标与评测结果
├── updates-20/
│   ├── metrics.csv       # round=0 为更新前的验证结果
│   ├── model.pth
│   ├── evaluation_games.jsonl
│   └── eval-*.sgf
└── updates-80/
    └── ...
```

如果只检查拟合与过拟合，可加 `--skip-arena`，此时不会进行对局评测，也不能据此断言棋力提升。输出目录必须不存在，避免覆盖实验。

先比较固定验证集的 `val_policy_kl`、`val_value_loss`：若训练指标下降而验证指标变差，应怀疑过拟合。若验证指标改善，再结合交换执色评测决定是否把在线训练更新次数改为 80；80 不是已经验证过的最佳值。

### 两个已训练模型直接交手，并更换开局种子

下面命令只评测，不重新训练。将 80 次分支设为候选、20 次分支设为对手：

```bash
uv run --extra rl python scripts/evaluate_checkpoints.py \
  --candidate logs/13x13/bs13_compare_v3/updates-80/model.pth \
  --opponent logs/13x13/bs13_compare_v3/updates-20/model.pth \
  --seeds 20261 20262 --games-per-seed 20 \
  --num-simulations 320 --device cuda \
  --output-dir logs/13x13/bs13_head_to_head_v3
```

每个种子 20 局，包含 10 个开局、每个开局交换执色各下一局；两个种子总计 40 局。种子不同于之前固定的 `10042`。两边使用相同搜索预算、贴目和 Pass 限制，不加根节点噪声。除显式指定的搜索预算外，共同规则与搜索参数取自候选 checkpoint，脚本检查对手棋盘尺寸和贴目兼容性。

`results.json` 保存按种子和总体统计，**所有胜负、得分率均从 candidate（这里为 80 次分支）的视角统计**，和棋计半分。`games.jsonl` 和 `seed-*-game-*.sgf` 保存每局记录，`evaluation_config.json` 保存两份权重的校验值与评测参数。截断局单独统计并从得分分母排除；存在截断时不要只看得分率。每个种子完成后更新结果，已有输出目录不会被覆盖。

```bash
cat logs/13x13/bs13_head_to_head_v3/results.json
```

40 局适合初步对照，不足以确认微小的棋力差异。后续可使用新种子或增加 `--games-per-seed`（必须为偶数），并使用新的输出目录。

## 3. Pass 输入单独实验

```bash
uv run --extra rl python scripts/train.py \
  --config configs/train-13x13-pass.json \
  --run-name bs13_pass_v2 --iterations 10
```

两个配置除 `input_features` 外相同，仍使用 20 次更新、320 次搜索和 `0.0003` 学习率，避免同时改变输入与更新预算。

| 输入版本 | 通道 | 兼容约定 |
| --- | --- | --- |
| `stones-v1` | 己方棋子、对方棋子、执色 | 默认版本，兼容原来的三通道模型 |
| `pass-v2` | 上述三通道 + 上一手是否 Pass | 单独训练的四通道版本 |

Pass 通道在上一手是 Pass 时全为 1；落子后归零，悔棋按恢复后的局面重新编码。旋转 / 翻转不会改变这个常量平面。当前版本没有增加完整历史、劫禁着或贴目输入，不声称状态信息已经完备。

训练初始化和数据加载会拒绝跨输入版本混用，不会静默补零或裁剪权重。`pass-v2` 默认从头开始训练，不能直接使用原来的三通道 checkpoint 或三通道样本。桌面推理、MCTS、ONNX 导出共用 checkpoint 输入版本识别；无版本元数据的旧权重可从第一层形状识别三通道输入。

导出时自动识别网络宽度、深度和输入通道：

```bash
uv run --extra rl python scripts/export_onnx.py \
  --checkpoint checkpoints/13x13/bs13_pass_v2/latest_model.pth
```

`pass-v2` 默认输出 `goplayer_v2.onnx`，`stones-v1` 输出 `goplayer_v1.onnx`，避免两个版本默认写入同一个文件。JSON 元数据写入输入版本与通道数。网页从 ONNX 输入形状识别 3 / 4 通道并构造对应输入，未知通道或错误棋盘尺寸会报错。仍需在 `frontend/src/game/modelConfig.ts` 中配置已验证模型的路径和 `trained` 标记。

## 4. 新指标的含义

| 字段 | 定义 |
| --- | --- |
| `policy_loss` | 当前训练批次的目标分布对网络预测的交叉熵 |
| `target_entropy` | **同一训练批次、同一次增强后的目标分布**的熵 |
| `policy_kl` | 上述两项之差，即 KL(target ∥ prediction)，可能有极小浮点误差 |
| `value_loss` | 当前批次胜负标签的均方误差 |
| `val_*` | 独立验证局面上相同定义的指标，不增强、不反向传播 |
| `validation_samples` | 本轮实际参与验证的局面数；为 0 时验证指标留空 |

验证使用 `eval()`，不会更新 BatchNorm 统计；最后不足一个批次的样本按实际数量加权。训练指标是每次更新前的批次指标的均值；验证指标在本轮更新完成后计算，两者模式与样本不同，不应要求数值一致。

在线训练的验证池会随新对局增长或淘汰，因此不同轮次的 `val_*` 不是固定数据上的严格对照。`compare_updates.py` 的验证池在整个对照期间固定，适合判断增加更新次数的影响。`mean_policy_entropy` 仍指本轮新生成自博弈局面的搜索目标统计，不能与混合回放池的 `policy_loss` 直接相减。

## 5. 在新实验中重新加载数据

```bash
uv run --extra rl python scripts/train.py \
  --config configs/train-13x13.json --run-name bs13_reuse_v3 \
  --init-checkpoint checkpoints/13x13/bs13_archive_v3/latest_model.pth \
  --replay-dir logs/13x13/bs13_archive_v3/data --iterations 10
```

已有对局保持原来的训练 / 验证划分。新实验会继续产生新数据，保存到自己的目录；导入的数据仍保留在源目录，因此应保留源归档。`--init-checkpoint` 仅加载权重，`--replay-dir` 加载数据；**这不是精确断点续训**，优化器、轮次与随机状态重新开始。只想复用固定数据追加更新时，应使用第 2 节的离线对照脚本。
