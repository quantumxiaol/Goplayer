# GoPlay 网页版

React + TypeScript + Canvas 围棋界面，支持 9 / 13 / 19 路本地双人对弈，以及 9 路浏览器 ONNX 单步建议。

项目介绍、截图、训练和部署说明见 [根目录 README](../README.md)。

## 本地开发

使用 Node.js 22.12+ 和 pnpm 10，在本目录执行：

```bash
pnpm install --frozen-lockfile
pnpm dev
```

打开 `http://localhost:5173/Goplayer/`，端口以 Vite 输出为准。`base` 区分大小写，配置为 `/Goplayer/`。

```bash
pnpm test       # AI 异步请求回归测试
pnpm lint       # ESLint
pnpm build      # TypeScript 检查与生产构建，输出到 dist/
pnpm preview    # 本地预览构建结果
```

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `src/game/goGame.ts` | 落子、提子、同形禁着、Pass、悔棋与面积计分 |
| `src/game/modelConfig.ts` | 各棋盘尺寸的模型路径与启用状态 |
| `src/hooks/useAI.ts` | ONNX Runtime Web 加载、编码与单步推理 |
| `src/components/GoBoardCanvas.tsx` | 棋盘绘制与指针交互 |
| `public/models/9x9/` | 随仓库提供的模型与导出元数据 |

AI 在用户点击按钮后加载，推理在浏览器中执行，无需 Python 后端或 API Key。13 / 19 路暂未提供模型。当前 AI 不包含 MCTS 或自动对弈。
