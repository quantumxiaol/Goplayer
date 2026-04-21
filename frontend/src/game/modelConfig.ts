import type { BoardSize } from './goGame'

export interface BoardAIConfig {
  trained: boolean
  label: string
  note: string
  modelPath: string | null
}

export const BOARD_AI_CONFIG: Record<BoardSize, BoardAIConfig> = {
  9: {
    trained: true,
    label: '已训练',
    note: '9 路已有训练权重和 ONNX 导出，可在浏览器中给出单步落子建议。',
    modelPath: `${import.meta.env.BASE_URL}models/9x9/goplayer_v1.onnx`,
  },
  13: {
    trained: false,
    label: '未训练',
    note: '13 路暂时没有训练好的模型，当前页面只提供双人本地对弈。',
    modelPath: null,
  },
  19: {
    trained: false,
    label: '未训练',
    note: '19 路暂时没有训练好的模型，当前页面只提供双人本地对弈。',
    modelPath: null,
  },
}
