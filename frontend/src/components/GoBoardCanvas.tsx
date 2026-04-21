import { useEffect, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { GoGame } from '../game/goGame'

interface BoardMetrics {
  boardSize: number
  margin: number
  cell: number
  stoneRadius: number
}

interface HoverPoint {
  row: number
  col: number
  valid: boolean
}

interface GoBoardCanvasProps {
  game: GoGame
  onPlay: (row: number, col: number) => void
}

function getBoardMetrics(canvasSize: number, boardSize: number): BoardMetrics {
  const margin = Math.max(28, Math.min(48, canvasSize * 0.09))
  const cell = (canvasSize - margin * 2) / (boardSize - 1)
  return {
    boardSize,
    margin,
    cell,
    stoneRadius: cell * 0.46,
  }
}

function getStarPointIndices(boardSize: number): number[] {
  if (boardSize === 19) {
    return [3, 9, 15]
  }
  if (boardSize === 13) {
    return [3, 6, 9]
  }
  if (boardSize === 9) {
    return [2, 4, 6]
  }
  return []
}

function getColumnLabels(boardSize: number): string[] {
  const letters = 'ABCDEFGHJKLMNOPQRST'
  return letters.slice(0, boardSize).split('')
}

export function GoBoardCanvas({ game, onPlay }: GoBoardCanvasProps) {
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [canvasSize, setCanvasSize] = useState(640)
  const [hoverPoint, setHoverPoint] = useState<HoverPoint | null>(null)
  const snapshot = game.toSnapshot()

  useEffect(() => {
    const element = wrapperRef.current
    if (!element) {
      return
    }

    const updateSize = () => {
      const nextWidth = Math.floor(element.getBoundingClientRect().width)
      setCanvasSize(Math.max(300, Math.min(nextWidth, 820)))
    }

    updateSize()
    const observer = new ResizeObserver(() => {
      updateSize()
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.floor(canvasSize * dpr)
    canvas.height = Math.floor(canvasSize * dpr)
    canvas.style.width = `${canvasSize}px`
    canvas.style.height = `${canvasSize}px`

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    context.clearRect(0, 0, canvasSize, canvasSize)

    const metrics = getBoardMetrics(canvasSize, snapshot.size)
    const labels = getColumnLabels(snapshot.size)

    const background = context.createLinearGradient(0, 0, canvasSize, canvasSize)
    background.addColorStop(0, '#f1d7a0')
    background.addColorStop(1, '#c38d4f')
    context.fillStyle = background
    context.fillRect(0, 0, canvasSize, canvasSize)

    context.fillStyle = 'rgba(255, 255, 255, 0.08)'
    for (let index = 0; index < 7; index += 1) {
      context.beginPath()
      context.ellipse(
        canvasSize * (0.12 + index * 0.11),
        canvasSize * 0.12,
        canvasSize * 0.08,
        canvasSize * 0.035,
        Math.PI / 7,
        0,
        Math.PI * 2,
      )
      context.fill()
    }

    context.strokeStyle = 'rgba(64, 39, 15, 0.72)'
    context.lineWidth = 1.2
    for (let line = 0; line < snapshot.size; line += 1) {
      const offset = metrics.margin + line * metrics.cell
      context.beginPath()
      context.moveTo(metrics.margin, offset)
      context.lineTo(canvasSize - metrics.margin, offset)
      context.stroke()

      context.beginPath()
      context.moveTo(offset, metrics.margin)
      context.lineTo(offset, canvasSize - metrics.margin)
      context.stroke()
    }

    context.fillStyle = '#503116'
    context.font = `${Math.max(12, metrics.cell * 0.22)}px "Avenir Next", "PingFang SC", sans-serif`
    context.textAlign = 'center'
    context.textBaseline = 'middle'

    for (let index = 0; index < snapshot.size; index += 1) {
      const axis = metrics.margin + index * metrics.cell
      const rowLabel = String(snapshot.size - index)
      const colLabel = labels[index]
      context.fillText(colLabel, axis, metrics.margin * 0.45)
      context.fillText(colLabel, axis, canvasSize - metrics.margin * 0.45)
      context.fillText(rowLabel, metrics.margin * 0.45, axis)
      context.fillText(rowLabel, canvasSize - metrics.margin * 0.45, axis)
    }

    for (const row of getStarPointIndices(snapshot.size)) {
      for (const col of getStarPointIndices(snapshot.size)) {
        const x = metrics.margin + col * metrics.cell
        const y = metrics.margin + row * metrics.cell
        context.beginPath()
        context.arc(x, y, Math.max(2.5, metrics.cell * 0.08), 0, Math.PI * 2)
        context.fill()
      }
    }

    const drawStone = (row: number, col: number, color: 'black' | 'white') => {
      const x = metrics.margin + col * metrics.cell
      const y = metrics.margin + row * metrics.cell
      const radius = metrics.stoneRadius
      const gradient = context.createRadialGradient(
        x - radius * 0.35,
        y - radius * 0.45,
        radius * 0.2,
        x,
        y,
        radius,
      )

      if (color === 'black') {
        gradient.addColorStop(0, '#5a5a5a')
        gradient.addColorStop(0.45, '#191919')
        gradient.addColorStop(1, '#050505')
      } else {
        gradient.addColorStop(0, '#ffffff')
        gradient.addColorStop(0.5, '#ececec')
        gradient.addColorStop(1, '#c9c9c9')
      }

      context.save()
      context.shadowColor = 'rgba(0, 0, 0, 0.28)'
      context.shadowBlur = radius * 0.45
      context.shadowOffsetY = radius * 0.12
      context.fillStyle = gradient
      context.beginPath()
      context.arc(x, y, radius, 0, Math.PI * 2)
      context.fill()
      context.restore()

      context.strokeStyle = color === 'black' ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.12)'
      context.lineWidth = 1
      context.beginPath()
      context.arc(x, y, radius, 0, Math.PI * 2)
      context.stroke()
    }

    for (let row = 0; row < snapshot.size; row += 1) {
      for (let col = 0; col < snapshot.size; col += 1) {
        const stone = snapshot.grid[row][col]
        if (stone) {
          drawStone(row, col, stone)
        }
      }
    }

    if (hoverPoint && snapshot.grid[hoverPoint.row][hoverPoint.col] === null) {
      const x = metrics.margin + hoverPoint.col * metrics.cell
      const y = metrics.margin + hoverPoint.row * metrics.cell
      context.save()
      context.globalAlpha = hoverPoint.valid ? 0.4 : 0.18
      drawStone(hoverPoint.row, hoverPoint.col, snapshot.currentPlayer)
      context.restore()
      context.lineWidth = 2
      context.strokeStyle = hoverPoint.valid ? 'rgba(17, 78, 38, 0.9)' : 'rgba(167, 29, 42, 0.88)'
      context.beginPath()
      context.arc(x, y, metrics.stoneRadius + 2.5, 0, Math.PI * 2)
      context.stroke()
    }

    if (snapshot.lastMove?.type === 'stone') {
      const x = metrics.margin + snapshot.lastMove.col! * metrics.cell
      const y = metrics.margin + snapshot.lastMove.row! * metrics.cell
      context.fillStyle =
        snapshot.lastMove.color === 'black' ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.82)'
      context.beginPath()
      context.arc(x, y, Math.max(3, metrics.cell * 0.11), 0, Math.PI * 2)
      context.fill()
    }
  }, [canvasSize, hoverPoint, snapshot])

  const getPointFromEvent = (
    event: ReactPointerEvent<HTMLCanvasElement>,
  ): { row: number; col: number } | null => {
    const canvas = canvasRef.current
    if (!canvas) {
      return null
    }

    const rect = canvas.getBoundingClientRect()
    const metrics = getBoardMetrics(canvasSize, snapshot.size)
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const col = Math.round((x - metrics.margin) / metrics.cell)
    const row = Math.round((y - metrics.margin) / metrics.cell)

    if (row < 0 || row >= snapshot.size || col < 0 || col >= snapshot.size) {
      return null
    }

    const targetX = metrics.margin + col * metrics.cell
    const targetY = metrics.margin + row * metrics.cell
    if (Math.hypot(targetX - x, targetY - y) > metrics.cell * 0.52) {
      return null
    }

    return { row, col }
  }

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = getPointFromEvent(event)
    if (!point) {
      setHoverPoint(null)
      return
    }
    setHoverPoint({
      ...point,
      valid: game.isValidMove(point.row, point.col),
    })
  }

  const handlePointerLeave = () => {
    setHoverPoint(null)
  }

  const handleClick = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = getPointFromEvent(event)
    if (!point) {
      return
    }
    onPlay(point.row, point.col)
  }

  return (
    <div className="board-shell" ref={wrapperRef}>
      <canvas
        ref={canvasRef}
        className="go-board-canvas"
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onClick={handleClick}
      />
    </div>
  )
}
