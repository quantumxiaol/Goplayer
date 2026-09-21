// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'

const runtime = vi.hoisted(() => ({ create: vi.fn() }))
vi.mock('onnxruntime-web/wasm', () => ({
  env: { wasm: {} },
  InferenceSession: { create: runtime.create },
  Tensor: class {
    type: string; data: Float32Array; dims: number[]
    constructor(type: string, data: Float32Array, dims: number[]) {
      this.type = type; this.data = data; this.dims = dims
    }
  },
}))
// Keep the real App and AI hook; Canvas drawing is unrelated to request races.
vi.mock('../src/components/GoBoardCanvas', () => ({
  GoBoardCanvas: ({ onPlay }: { onPlay: (row: number, col: number) => void }) => (
    <button onClick={() => onPlay(4, 4)}>测试落子</button>
  ),
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function outputs(index = 80) {
  const data = new Float32Array(82)
  data[index] = 10
  return { policy_logits: { data }, value: { data: new Float32Array([0.5]) } }
}

function session() {
  return {
    inputMetadata: [{ isTensor: true, shape: [1, 3, 9, 9] }],
    inputNames: ['board_state'], outputNames: ['policy_logits', 'value'],
    run: vi.fn().mockResolvedValue(outputs()),
    release: vi.fn().mockResolvedValue(undefined),
  }
}

const click = (name: string) => fireEvent.click(screen.getByRole('button', { name, exact: true }))
const notice = () => document.querySelector('.notice-bar')?.textContent

afterEach(cleanup)
beforeEach(() => { runtime.create.mockReset() })

describe('AI request lifetime', () => {
  it('still displays a normal model suggestion', async () => {
    runtime.create.mockResolvedValue(session())
    render(<App />)
    click('AI 建议')
    await waitFor(() => expect(notice()).toContain('J1'))
    expect(document.querySelector('.ai-suggestion')?.textContent).toContain('J1')
  })

  it('feeds the pass plane to a four-channel model after Pass', async () => {
    const model = session()
    model.inputMetadata[0].shape[1] = 4
    runtime.create.mockResolvedValue(model)
    render(<App />)
    click('Pass')
    click('AI 建议')
    await waitFor(() => expect(model.run).toHaveBeenCalledOnce())
    const tensor = model.run.mock.calls[0][0].board_state
    expect(tensor.dims).toEqual([1, 4, 9, 9])
    expect(Array.from(tensor.data.slice(243))).toEqual(Array(81).fill(1))
  })

  it('rejects an unsupported input shape before inference', async () => {
    const model = session()
    model.inputMetadata[0].shape[1] = 5
    runtime.create.mockResolvedValue(model)
    render(<App />)
    click('AI 建议')
    await waitFor(() => expect(document.querySelector('.ai-error')?.textContent).toContain('模型输入版本'))
    expect(model.run).not.toHaveBeenCalled()
  })

  it.each(['重开', '13 x 13', 'Pass', '测试落子', '悔棋'])(
    'discards a delayed model load after %s', async (action) => {
      const loading = deferred<ReturnType<typeof session>>()
      const model = session()
      runtime.create.mockReturnValue(loading.promise)
      render(<App />)
      if (action === '悔棋') click('测试落子')
      click('AI 建议')
      click(action)
      const nextNotice = notice()
      await act(async () => { loading.resolve(model) })
      expect(notice()).toBe(nextNotice)
      expect(document.querySelector('.ai-suggestion')).toBeNull()
      expect(model.run).not.toHaveBeenCalled()
      expect(model.release).toHaveBeenCalledOnce()
      const button = screen.getByRole('button', { name: 'AI 建议', exact: true }) as HTMLButtonElement
      expect(button.disabled).toBe(action === '13 x 13')
    },
  )

  it.each([false, true])('does not overwrite a newer result when stale inference fails=%s', async (fails) => {
    const old = deferred<ReturnType<typeof outputs>>()
    const model = session()
    model.run.mockReturnValueOnce(old.promise)
    runtime.create.mockResolvedValue(model)
    render(<App />)
    click('AI 建议')
    await waitFor(() => expect(model.run).toHaveBeenCalledOnce())
    click('重开')
    click('AI 建议')
    await waitFor(() => expect(notice()).toContain('J1'))
    const currentNotice = notice()
    await act(async () => {
      if (fails) old.reject(new Error('stale inference error'))
      else old.resolve(outputs(0))
    })
    expect(notice()).toBe(currentNotice)
    expect(document.querySelector('.ai-error')).toBeNull()
    expect(document.querySelector('.ai-suggestion')?.textContent).toContain('J1')
  })

  it('does not overwrite a newer result with a stale loading error', async () => {
    const old = deferred<ReturnType<typeof session>>()
    runtime.create.mockReturnValueOnce(old.promise).mockResolvedValue(session())
    render(<App />)
    click('AI 建议')
    click('重开')
    click('AI 建议')
    await waitFor(() => expect(notice()).toContain('J1'))
    const currentNotice = notice()
    await act(async () => { old.reject(new Error('stale loading error')) })
    expect(notice()).toBe(currentNotice)
    expect(document.querySelector('.ai-error')).toBeNull()
  })

  it('keeps a cached session usable after changing away from 9x9 and back', async () => {
    const model = session()
    runtime.create.mockResolvedValue(model)
    render(<App />)
    click('AI 建议')
    await waitFor(() => expect(notice()).toContain('J1'))
    click('13 x 13')
    click('9 x 9')
    click('AI 建议')
    await waitFor(() => expect(model.run).toHaveBeenCalledTimes(2))
    expect(model.release).not.toHaveBeenCalled()
    expect(runtime.create).toHaveBeenCalledOnce()
  })

  it('releases a session whose loading completes after unmount', async () => {
    const loading = deferred<ReturnType<typeof session>>()
    const model = session()
    runtime.create.mockReturnValue(loading.promise)
    const view = render(<App />)
    click('AI 建议')
    view.unmount()
    await act(async () => { loading.resolve(model) })
    expect(model.run).not.toHaveBeenCalled()
    expect(model.release).toHaveBeenCalledOnce()
  })
})
