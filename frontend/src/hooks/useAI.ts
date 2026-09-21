import { useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web/wasm'
import ortWasmModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import ortWasmBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { encodeBoard } from '../game/encoder'
import type { GoGame, GoSnapshot, Point, StoneColor } from '../game/goGame'

ort.env.wasm.proxy = false
ort.env.wasm.numThreads =
  typeof navigator === 'undefined'
    ? 1
    : Math.min(Math.max(navigator.hardwareConcurrency || 1, 1), 4)
ort.env.wasm.wasmPaths = {
  mjs: ortWasmModuleUrl,
  wasm: ortWasmBinaryUrl,
}

export interface AIMoveSuggestion {
  type: 'stone' | 'pass'
  row: number | null
  col: number | null
  moveIndex: number
  confidence: number
  value: number
}

export interface UseAIOptions {
  game: GoGame
  playerColor?: StoneColor
  modelPath?: string | null
  allowPass?: boolean
  enabled?: boolean
  executionProviders?: ort.InferenceSession.SessionOptions['executionProviders']
}

export interface UseAIResult {
  status: 'idle' | 'loading' | 'ready' | 'thinking' | 'error'
  modelReady: boolean
  error: string | null
  suggestion: AIMoveSuggestion | null
  value: number | null
  legalMoves: Point[]
  suggestMove: () => Promise<AIMoveSuggestion | null>
  clearSuggestion: () => void
}

const DEFAULT_EXECUTION_PROVIDERS: ort.InferenceSession.SessionOptions['executionProviders'] = ['wasm']

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits)
  const exps = logits.map((logit) => Math.exp(logit - maxLogit))
  const sum = exps.reduce((acc, value) => acc + value, 0)
  return exps.map((value) => value / sum)
}

function selectSuggestion(
  snapshot: GoSnapshot,
  legalMoves: Point[],
  logits: Float32Array | number[],
  value: number,
  allowPass: boolean,
): { suggestion: AIMoveSuggestion | null } {
  const moveIndexes = legalMoves.map((point) => point.row * snapshot.size + point.col)
  if (allowPass) {
    moveIndexes.push(snapshot.size * snapshot.size)
  }

  if (moveIndexes.length === 0) {
    return { suggestion: null }
  }

  const candidateLogits = moveIndexes.map((index) => Number(logits[index] ?? Number.NEGATIVE_INFINITY))
  const probabilities = softmax(candidateLogits)

  let bestIndex = 0
  for (let index = 1; index < candidateLogits.length; index += 1) {
    if (candidateLogits[index] > candidateLogits[bestIndex]) {
      bestIndex = index
    }
  }

  const selectedMoveIndex = moveIndexes[bestIndex]
  if (selectedMoveIndex === snapshot.size * snapshot.size) {
    return {
      suggestion: {
        type: 'pass',
        row: null,
        col: null,
        moveIndex: selectedMoveIndex,
        confidence: probabilities[bestIndex] ?? 0,
        value,
      },
    }
  }

  return {
    suggestion: {
      type: 'stone',
      row: Math.floor(selectedMoveIndex / snapshot.size),
      col: selectedMoveIndex % snapshot.size,
      moveIndex: selectedMoveIndex,
      confidence: probabilities[bestIndex] ?? 0,
      value,
    },
  }
}

export function useAI({
  game,
  playerColor,
  modelPath = null,
  allowPass = true,
  enabled = true,
  executionProviders = DEFAULT_EXECUTION_PROVIDERS,
}: UseAIOptions): UseAIResult {
  const requestId = useRef(0)
  const [loadState, setLoadState] = useState<{
    key: string
    session: ort.InferenceSession | null
    error: string | null
  }>({
    key: '',
    session: null,
    error: null,
  })
  const [thinking, setThinking] = useState(false)
  const [hasRequestedLoad, setHasRequestedLoad] = useState(false)
  const [runtimeError, setRuntimeError] = useState<string | null>(null)
  const [suggestion, setSuggestion] = useState<AIMoveSuggestion | null>(null)
  const [value, setValue] = useState<number | null>(null)
  const snapshot = game.toSnapshot()
  const activeColor = playerColor ?? snapshot.currentPlayer
  const legalMoves = game.legalMoves(activeColor)
  const providerKey = executionProviders?.join(',') ?? 'wasm'
  const resolvedModelPath = enabled ? modelPath : null
  const loadKey = `${resolvedModelPath ?? 'no-model'}::${providerKey}`
  const session = loadState.key === loadKey ? loadState.session : null
  const loadError = loadState.key === loadKey ? loadState.error : null
  useEffect(() => () => {
    requestId.current += 1
  }, [game, loadKey, activeColor, allowPass])
  useEffect(() => () => {
    void loadState.session?.release()
  }, [loadState.session])
  const status: UseAIResult['status'] = !enabled
    ? 'idle'
    : thinking
      ? 'thinking'
      : runtimeError || loadError
        ? 'error'
        : session
          ? 'ready'
          : hasRequestedLoad
            ? 'loading'
            : 'idle'
  const error = enabled ? runtimeError ?? loadError : null

  const createSession = async (id: number): Promise<ort.InferenceSession | null> => {
    if (!enabled || !resolvedModelPath) {
      return null
    }
    if (session) {
      return session
    }
    setHasRequestedLoad(true)

    const normalizedExecutionProviders =
      providerKey.length > 0
        ? (providerKey.split(',') as ort.InferenceSession.SessionOptions['executionProviders'])
        : DEFAULT_EXECUTION_PROVIDERS

    try {
      const nextSession = await ort.InferenceSession.create(resolvedModelPath, {
        executionProviders: normalizedExecutionProviders,
        graphOptimizationLevel: 'all',
      })
      if (id !== requestId.current) {
        await nextSession.release()
        return null
      }
      setLoadState({
        key: loadKey,
        session: nextSession,
        error: null,
      })
      setRuntimeError(null)
      return nextSession
    } catch (cause) {
      if (id !== requestId.current) {
        return null
      }
      setLoadState({
        key: loadKey,
        session: null,
        error: cause instanceof Error ? cause.message : String(cause),
      })
      return null
    }
  }

  const clearSuggestion = () => {
    requestId.current += 1
    setThinking(false)
    setSuggestion(null)
    setValue(null)
    setRuntimeError(null)
    setHasRequestedLoad(false)
  }

  const suggestMove = async (): Promise<AIMoveSuggestion | null> => {
    if (!enabled || !resolvedModelPath || snapshot.gameOver) {
      return null
    }
    const id = ++requestId.current
    setThinking(true)
    setRuntimeError(null)

    try {
      const currentSession = await createSession(id)
      if (!currentSession || id !== requestId.current) {
        return null
      }
      const metadata = currentSession.inputMetadata[0]
      if (!metadata?.isTensor || metadata.shape.length !== 4 ||
          (metadata.shape[1] !== 3 && metadata.shape[1] !== 4) ||
          metadata.shape[2] !== snapshot.size || metadata.shape[3] !== snapshot.size) {
        throw new Error('模型输入版本或棋盘尺寸不匹配；仅支持 stones-v1 / pass-v2。')
      }
      const channels = metadata.shape[1]
      const inputData = encodeBoard(snapshot, activeColor, channels)
      const inputTensor = new ort.Tensor('float32', inputData, [
        1,
        channels,
        snapshot.size,
        snapshot.size,
      ])
      const outputs = await currentSession.run({
        [currentSession.inputNames[0]]: inputTensor,
      })
      if (id !== requestId.current) {
        return null
      }

      const policyOutput = outputs.policy_logits ?? outputs[currentSession.outputNames[0]]
      const valueOutput = outputs.value ?? outputs[currentSession.outputNames[1]]
      if (!policyOutput || !valueOutput) {
        throw new Error('ONNX outputs do not match expected names: policy_logits, value.')
      }

      const scoreValue = Number(Array.isArray(valueOutput.data) ? valueOutput.data[0] : valueOutput.data[0])
      const result = selectSuggestion(
        snapshot,
        legalMoves,
        policyOutput.data as Float32Array | number[],
        scoreValue,
        allowPass,
      )

      setSuggestion(result.suggestion)
      setValue(scoreValue)
      return result.suggestion
    } catch (cause) {
      if (id !== requestId.current) {
        return null
      }
      setSuggestion(null)
      setValue(null)
      setRuntimeError(cause instanceof Error ? cause.message : String(cause))
      return null
    } finally {
      if (id === requestId.current) {
        setThinking(false)
      }
    }
  }

  return {
    status,
    modelReady: enabled && (status === 'ready' || status === 'thinking'),
    error,
    suggestion,
    value,
    legalMoves,
    suggestMove,
    clearSuggestion,
  }
}
