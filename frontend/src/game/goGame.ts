export const BOARD_SIZES = [9, 13, 19] as const

export type BoardSize = (typeof BOARD_SIZES)[number]
export type StoneColor = 'black' | 'white'
export type Intersection = StoneColor | null
export type MoveError =
  | 'game_over'
  | 'wrong_player'
  | 'out_of_bounds'
  | 'occupied'
  | 'suicide'
  | 'superko'

export interface Point {
  row: number
  col: number
}

export interface GoMove {
  type: 'stone' | 'pass'
  color: StoneColor
  moveNumber: number
  captured: Point[]
  row?: number
  col?: number
  boardSignature: string
}

export interface ScoreSummary {
  blackScore: number
  whiteScore: number
  winner: StoneColor
}

export interface PlayResult {
  ok: boolean
  capturedCount?: number
  error?: MoveError
}

export interface PassResult {
  ok: boolean
  gameOver: boolean
}

export interface GoSnapshot {
  size: number
  grid: Intersection[][]
  currentPlayer: StoneColor
  captures: Record<StoneColor, number>
  consecutivePasses: number
  moveCount: number
  komi: number
  gameOver: boolean
  boardSignature: string
  lastMove: GoMove | null
  moves: GoMove[]
}

interface SimulatedMove {
  grid: Intersection[][]
  captured: Point[]
}

interface HistorySnapshot {
  grid: Intersection[][]
  captures: Record<StoneColor, number>
  positionHistory: Set<string>
  consecutivePasses: number
  currentPlayer: StoneColor
  gameOver: boolean
  moves: GoMove[]
  lastMove: GoMove | null
}

const DIRECTIONS: Point[] = [
  { row: -1, col: 0 },
  { row: 1, col: 0 },
  { row: 0, col: -1 },
  { row: 0, col: 1 },
]

function cloneGrid(grid: Intersection[][]): Intersection[][] {
  return grid.map((line) => [...line])
}

function cloneCaptures(
  captures: Record<StoneColor, number>,
): Record<StoneColor, number> {
  return {
    black: captures.black,
    white: captures.white,
  }
}

function pointKey(row: number, col: number): string {
  return `${row},${col}`
}

function copyMove(move: GoMove): GoMove {
  return {
    ...move,
    captured: move.captured.map((point) => ({ ...point })),
  }
}

export function opponentOf(color: StoneColor): StoneColor {
  return color === 'black' ? 'white' : 'black'
}

export function serializeGrid(grid: Intersection[][]): string {
  return grid
    .map((row) =>
      row
        .map((cell) =>
          cell === 'black' ? 'B' : cell === 'white' ? 'W' : '.',
        )
        .join(''),
    )
    .join('|')
}

export class GoGame {
  size: number
  komi: number
  grid: Intersection[][]
  captures: Record<StoneColor, number>
  positionHistory: Set<string>
  consecutivePasses: number
  currentPlayer: StoneColor
  gameOver: boolean
  moves: GoMove[]
  lastMove: GoMove | null
  private undoStack: HistorySnapshot[]

  constructor(size: BoardSize = 19) {
    this.size = size
    this.komi = GoGame.defaultKomi(size)
    this.grid = []
    this.captures = { black: 0, white: 0 }
    this.positionHistory = new Set<string>()
    this.consecutivePasses = 0
    this.currentPlayer = 'black'
    this.gameOver = false
    this.moves = []
    this.lastMove = null
    this.undoStack = []
    this.reset(size)
  }

  static defaultKomi(size: number): number {
    if (size === 19) {
      return 7.5
    }
    if (size === 13) {
      return 2
    }
    if (size === 9) {
      return 5.5
    }
    return 0
  }

  reset(size = this.size): void {
    this.size = size
    this.komi = GoGame.defaultKomi(size)
    this.grid = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => null),
    )
    this.captures = { black: 0, white: 0 }
    this.positionHistory = new Set([this.boardSignature()])
    this.consecutivePasses = 0
    this.currentPlayer = 'black'
    this.gameOver = false
    this.moves = []
    this.lastMove = null
    this.undoStack = []
  }

  clone(): GoGame {
    const next = new GoGame(this.size as BoardSize)
    next.komi = this.komi
    next.grid = cloneGrid(this.grid)
    next.captures = cloneCaptures(this.captures)
    next.positionHistory = new Set(this.positionHistory)
    next.consecutivePasses = this.consecutivePasses
    next.currentPlayer = this.currentPlayer
    next.gameOver = this.gameOver
    next.moves = this.moves.map(copyMove)
    next.lastMove = this.lastMove ? copyMove(this.lastMove) : null
    next.undoStack = this.undoStack.map((snapshot) => ({
      ...snapshot,
      grid: cloneGrid(snapshot.grid),
      captures: cloneCaptures(snapshot.captures),
      positionHistory: new Set(snapshot.positionHistory),
      moves: snapshot.moves.map(copyMove),
      lastMove: snapshot.lastMove ? copyMove(snapshot.lastMove) : null,
    }))
    return next
  }

  boardSignature(grid = this.grid): string {
    return serializeGrid(grid)
  }

  toSnapshot(): GoSnapshot {
    return {
      size: this.size,
      grid: cloneGrid(this.grid),
      currentPlayer: this.currentPlayer,
      captures: cloneCaptures(this.captures),
      consecutivePasses: this.consecutivePasses,
      moveCount: this.moves.length,
      komi: this.komi,
      gameOver: this.gameOver,
      boardSignature: this.boardSignature(),
      lastMove: this.lastMove ? copyMove(this.lastMove) : null,
      moves: this.moves.map(copyMove),
    }
  }

  getNeighbors(row: number, col: number): Point[] {
    return DIRECTIONS.map((direction) => ({
      row: row + direction.row,
      col: col + direction.col,
    })).filter(
      (point) =>
        point.row >= 0 &&
        point.row < this.size &&
        point.col >= 0 &&
        point.col < this.size,
    )
  }

  getGroupAndLiberties(
    row: number,
    col: number,
    grid = this.grid,
  ): { group: Point[]; liberties: Point[] } {
    const color = grid[row]?.[col]
    if (!color) {
      return { group: [], liberties: [] }
    }

    const group: Point[] = []
    const liberties: Point[] = []
    const stack: Point[] = [{ row, col }]
    const groupSet = new Set<string>()
    const libertySet = new Set<string>()

    while (stack.length > 0) {
      const point = stack.pop()
      if (!point) {
        continue
      }

      const key = pointKey(point.row, point.col)
      if (groupSet.has(key)) {
        continue
      }

      groupSet.add(key)
      group.push(point)

      for (const neighbor of this.getNeighbors(point.row, point.col)) {
        const stone = grid[neighbor.row][neighbor.col]
        if (stone === null) {
          const libertyKey = pointKey(neighbor.row, neighbor.col)
          if (!libertySet.has(libertyKey)) {
            libertySet.add(libertyKey)
            liberties.push(neighbor)
          }
          continue
        }
        if (stone === color && !groupSet.has(pointKey(neighbor.row, neighbor.col))) {
          stack.push(neighbor)
        }
      }
    }

    return { group, liberties }
  }

  isValidMove(row: number, col: number, color = this.currentPlayer): boolean {
    return this.validateMove(row, col, color).ok
  }

  explainMove(row: number, col: number, color = this.currentPlayer): MoveError | null {
    const result = this.validateMove(row, col, color)
    return result.ok ? null : (result.error ?? null)
  }

  legalMoves(color = this.currentPlayer): Point[] {
    const legal: Point[] = []
    for (let row = 0; row < this.size; row += 1) {
      for (let col = 0; col < this.size; col += 1) {
        if (this.isValidMove(row, col, color)) {
          legal.push({ row, col })
        }
      }
    }
    return legal
  }

  play(row: number, col: number, color = this.currentPlayer): PlayResult {
    if (this.gameOver) {
      return { ok: false, error: 'game_over' }
    }
    if (color !== this.currentPlayer) {
      return { ok: false, error: 'wrong_player' }
    }

    const validation = this.validateMove(row, col, color)
    if (!validation.ok) {
      return validation
    }

    const simulated = this.simulateMove(row, col, color)
    if (!simulated) {
      return { ok: false, error: 'suicide' }
    }

    this.undoStack.push(this.createHistorySnapshot())
    this.grid = simulated.grid

    if (simulated.captured.length > 0) {
      this.captures[opponentOf(color)] += simulated.captured.length
    }

    const move: GoMove = {
      type: 'stone',
      color,
      row,
      col,
      moveNumber: this.moves.length + 1,
      captured: simulated.captured.map((point) => ({ ...point })),
      boardSignature: this.boardSignature(),
    }

    this.positionHistory.add(move.boardSignature)
    this.moves.push(move)
    this.lastMove = move
    this.consecutivePasses = 0
    this.currentPlayer = opponentOf(color)
    return {
      ok: true,
      capturedCount: simulated.captured.length,
    }
  }

  pass(color = this.currentPlayer): PassResult {
    if (this.gameOver) {
      return { ok: false, gameOver: true }
    }
    if (color !== this.currentPlayer) {
      return { ok: false, gameOver: this.gameOver }
    }

    this.undoStack.push(this.createHistorySnapshot())
    this.consecutivePasses += 1

    const move: GoMove = {
      type: 'pass',
      color,
      moveNumber: this.moves.length + 1,
      captured: [],
      boardSignature: this.boardSignature(),
    }

    this.moves.push(move)
    this.lastMove = move
    this.currentPlayer = opponentOf(color)

    if (this.consecutivePasses >= 2) {
      this.gameOver = true
    }

    return { ok: true, gameOver: this.gameOver }
  }

  undo(): boolean {
    const snapshot = this.undoStack.pop()
    if (!snapshot) {
      return false
    }

    this.grid = cloneGrid(snapshot.grid)
    this.captures = cloneCaptures(snapshot.captures)
    this.positionHistory = new Set(snapshot.positionHistory)
    this.consecutivePasses = snapshot.consecutivePasses
    this.currentPlayer = snapshot.currentPlayer
    this.gameOver = snapshot.gameOver
    this.moves = snapshot.moves.map(copyMove)
    this.lastMove = snapshot.lastMove ? copyMove(snapshot.lastMove) : null
    return true
  }

  calculateAreaScore(): ScoreSummary {
    let blackStones = 0
    let whiteStones = 0
    let blackTerritory = 0
    let whiteTerritory = 0

    for (const row of this.grid) {
      for (const cell of row) {
        if (cell === 'black') {
          blackStones += 1
        }
        if (cell === 'white') {
          whiteStones += 1
        }
      }
    }

    const visited = new Set<string>()
    for (let row = 0; row < this.size; row += 1) {
      for (let col = 0; col < this.size; col += 1) {
        if (this.grid[row][col] !== null || visited.has(pointKey(row, col))) {
          continue
        }
        const { region, borderColors } = this.exploreEmptyRegion(row, col, visited)
        if (borderColors.size !== 1) {
          continue
        }
        const owner = [...borderColors][0]
        if (owner === 'black') {
          blackTerritory += region.length
        } else if (owner === 'white') {
          whiteTerritory += region.length
        }
      }
    }

    const blackScore = blackStones + blackTerritory
    const whiteScore = whiteStones + whiteTerritory + this.komi
    return {
      blackScore,
      whiteScore,
      winner: blackScore > whiteScore ? 'black' : 'white',
    }
  }

  private createHistorySnapshot(): HistorySnapshot {
    return {
      grid: cloneGrid(this.grid),
      captures: cloneCaptures(this.captures),
      positionHistory: new Set(this.positionHistory),
      consecutivePasses: this.consecutivePasses,
      currentPlayer: this.currentPlayer,
      gameOver: this.gameOver,
      moves: this.moves.map(copyMove),
      lastMove: this.lastMove ? copyMove(this.lastMove) : null,
    }
  }

  private validateMove(
    row: number,
    col: number,
    color: StoneColor,
  ): PlayResult {
    if (this.gameOver) {
      return { ok: false, error: 'game_over' }
    }
    if (row < 0 || row >= this.size || col < 0 || col >= this.size) {
      return { ok: false, error: 'out_of_bounds' }
    }
    if (this.grid[row][col] !== null) {
      return { ok: false, error: 'occupied' }
    }

    const simulated = this.simulateMove(row, col, color)
    if (!simulated) {
      return { ok: false, error: 'suicide' }
    }

    const nextSignature = serializeGrid(simulated.grid)
    if (this.positionHistory.has(nextSignature)) {
      return { ok: false, error: 'superko' }
    }

    return { ok: true }
  }

  private simulateMove(
    row: number,
    col: number,
    color: StoneColor,
  ): SimulatedMove | null {
    const grid = cloneGrid(this.grid)
    grid[row][col] = color

    const opponent = opponentOf(color)
    const checked = new Set<string>()
    const captured: Point[] = []

    for (const neighbor of this.getNeighbors(row, col)) {
      if (grid[neighbor.row][neighbor.col] !== opponent) {
        continue
      }
      const neighborKey = pointKey(neighbor.row, neighbor.col)
      if (checked.has(neighborKey)) {
        continue
      }

      const groupState = this.getGroupAndLiberties(neighbor.row, neighbor.col, grid)
      for (const point of groupState.group) {
        checked.add(pointKey(point.row, point.col))
      }
      if (groupState.liberties.length > 0) {
        continue
      }
      for (const point of groupState.group) {
        grid[point.row][point.col] = null
        captured.push(point)
      }
    }

    const ownGroup = this.getGroupAndLiberties(row, col, grid)
    if (ownGroup.liberties.length === 0) {
      return null
    }

    return { grid, captured }
  }

  private exploreEmptyRegion(
    startRow: number,
    startCol: number,
    visited: Set<string>,
  ): { region: Point[]; borderColors: Set<StoneColor> } {
    const stack: Point[] = [{ row: startRow, col: startCol }]
    const region: Point[] = []
    const borderColors = new Set<StoneColor>()

    while (stack.length > 0) {
      const point = stack.pop()
      if (!point) {
        continue
      }
      const key = pointKey(point.row, point.col)
      if (visited.has(key)) {
        continue
      }

      visited.add(key)
      if (this.grid[point.row][point.col] !== null) {
        continue
      }

      region.push(point)
      for (const neighbor of this.getNeighbors(point.row, point.col)) {
        const stone = this.grid[neighbor.row][neighbor.col]
        if (stone === null) {
          if (!visited.has(pointKey(neighbor.row, neighbor.col))) {
            stack.push(neighbor)
          }
          continue
        }
        borderColors.add(stone)
      }
    }

    return { region, borderColors }
  }
}
