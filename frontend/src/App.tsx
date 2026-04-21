import { useState } from 'react'
import './App.css'
import { GoBoardCanvas } from './components/GoBoardCanvas'
import {
  BOARD_SIZES,
  GoGame,
  type BoardSize,
  type GoMove,
  type MoveError,
  type StoneColor,
} from './game/goGame'
import { BOARD_AI_CONFIG } from './game/modelConfig'
import { useAI } from './hooks/useAI'

const PLAYER_LABEL: Record<StoneColor, string> = {
  black: '黑棋',
  white: '白棋',
}

const MOVE_ERROR_TEXT: Record<MoveError, string> = {
  game_over: '对局已经结束，先重开或悔棋再继续。',
  wrong_player: '当前不是这个颜色的回合。',
  out_of_bounds: '落点超出棋盘范围。',
  occupied: '这个交叉点已经有棋子。',
  suicide: '这是自杀着，除非能提子，否则不能下。',
  superko: '这步会重复历史局面，触发打劫/同形禁着。',
}

function formatCoordinate(size: number, move: GoMove): string {
  if (move.type === 'pass') {
    return 'Pass'
  }
  const letters = 'ABCDEFGHJKLMNOPQRST'
  const column = letters[move.col ?? 0] ?? '?'
  return `${column}${size - (move.row ?? 0)}`
}

function getMoveSummary(game: GoGame, move: GoMove): string {
  if (move.type === 'pass') {
    return `${PLAYER_LABEL[move.color]}停一手`
  }

  const detail = `${PLAYER_LABEL[move.color]}落子 ${formatCoordinate(game.size, move)}`
  if (move.captured.length === 0) {
    return detail
  }
  return `${detail}，提子 ${move.captured.length} 枚`
}

function formatSuggestion(row: number | null, col: number | null, size: number): string {
  if (row === null || col === null) {
    return 'Pass'
  }
  const letters = 'ABCDEFGHJKLMNOPQRST'
  return `${letters[col] ?? '?'}${size - row}`
}

function App() {
  const [game, setGame] = useState(() => new GoGame(9))
  const [notice, setNotice] = useState('黑先。点击棋盘交叉点开始对弈。')
  const snapshot = game.toSnapshot()
  const activeBoardSize = snapshot.size as BoardSize
  const activeAISupport = BOARD_AI_CONFIG[activeBoardSize]
  const ai = useAI({
    game,
    playerColor: snapshot.currentPlayer,
    enabled: activeAISupport.trained,
    modelPath: activeAISupport.modelPath,
    allowPass: true,
  })
  const score = game.calculateAreaScore()
  const blackCaptures = snapshot.captures.white
  const whiteCaptures = snapshot.captures.black
  const recentMoves = [...snapshot.moves].slice(-10).reverse()

  const replaceGame = (nextGame: GoGame, message: string) => {
    ai.clearSuggestion()
    setGame(nextGame)
    setNotice(message)
  }

  const startNewGame = (size: BoardSize) => {
    const nextGame = new GoGame(size)
    replaceGame(nextGame, `${size}x${size} 新对局开始，黑先。`)
  }

  const handlePlay = (row: number, col: number) => {
    const nextGame = game.clone()
    const result = nextGame.play(row, col)
    if (!result.ok) {
      setNotice(MOVE_ERROR_TEXT[result.error ?? 'occupied'])
      return
    }

    const lastMove = nextGame.lastMove
    if (!lastMove) {
      return
    }

    if (nextGame.gameOver) {
      const finalScore = nextGame.calculateAreaScore()
      replaceGame(
        nextGame,
        `${getMoveSummary(nextGame, lastMove)}。终局：${PLAYER_LABEL[finalScore.winner]}胜，黑 ${finalScore.blackScore.toFixed(
          1,
        )} : 白 ${finalScore.whiteScore.toFixed(1)}。`,
      )
      return
    }

    replaceGame(nextGame, `${getMoveSummary(nextGame, lastMove)}。轮到${PLAYER_LABEL[nextGame.currentPlayer]}。`)
  }

  const handlePass = () => {
    const nextGame = game.clone()
    const result = nextGame.pass()
    if (!result.ok) {
      setNotice('当前不能停一手。')
      return
    }

    const lastMove = nextGame.lastMove
    if (!lastMove) {
      return
    }

    if (result.gameOver) {
      const finalScore = nextGame.calculateAreaScore()
      replaceGame(
        nextGame,
        `双方连续停一手，终局。${PLAYER_LABEL[finalScore.winner]}胜，黑 ${finalScore.blackScore.toFixed(
          1,
        )} : 白 ${finalScore.whiteScore.toFixed(1)}。`,
      )
      return
    }

    replaceGame(nextGame, `${getMoveSummary(nextGame, lastMove)}。轮到${PLAYER_LABEL[nextGame.currentPlayer]}。`)
  }

  const handleUndo = () => {
    const nextGame = game.clone()
    if (!nextGame.undo()) {
      setNotice('当前没有可悔的手数。')
      return
    }
    replaceGame(nextGame, `已悔棋。轮到${PLAYER_LABEL[nextGame.currentPlayer]}。`)
  }

  const handleReset = () => {
    startNewGame(activeBoardSize)
  }

  const handleAISuggest = async () => {
    if (!activeAISupport.trained) {
      setNotice(`${activeBoardSize}x${activeBoardSize} 还没有训练好的 AI 模型，当前仅支持双人对弈。`)
      return
    }
    if (snapshot.gameOver) {
      setNotice('对局已经结束，不能再请求 AI 建议。')
      return
    }

    const suggestion = await ai.suggestMove()
    if (!suggestion) {
      setNotice(ai.error ?? 'AI 暂时没有返回建议落点。')
      return
    }

    setNotice(
      suggestion.type === 'pass'
        ? `AI 建议 ${PLAYER_LABEL[snapshot.currentPlayer]} Pass，局面价值 ${suggestion.value.toFixed(3)}。`
        : `AI 建议 ${PLAYER_LABEL[snapshot.currentPlayer]} 落在 ${formatSuggestion(
            suggestion.row,
            suggestion.col,
            snapshot.size,
          )}，置信度 ${(suggestion.confidence * 100).toFixed(1)}%，局面价值 ${suggestion.value.toFixed(3)}。`,
    )
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <span className="eyebrow">React + Canvas + TypeScript</span>
          <h1>Goplayer Frontend Demo</h1>
          <p className="hero-text">
            规则引擎按 Python 版 GoEnv 重构，保留气、提子、打劫/同形禁着、Pass、悔棋和面积计分接口，方便后续接入 GPU 推理或 AI 对弈模块。
          </p>
        </div>
        <div className="hero-stat-grid">
          <article className="hero-stat">
            <span>棋盘</span>
            <strong>{snapshot.size} x {snapshot.size}</strong>
          </article>
          <article className="hero-stat">
            <span>当前回合</span>
            <strong>{PLAYER_LABEL[snapshot.currentPlayer]}</strong>
          </article>
          <article className="hero-stat">
            <span>总手数</span>
            <strong>{snapshot.moveCount}</strong>
          </article>
          <article className="hero-stat">
            <span>贴目</span>
            <strong>{snapshot.komi.toFixed(1)}</strong>
          </article>
        </div>
      </section>

      <section className="workspace">
        <div className="board-panel">
          <div className="board-panel-header">
            <div>
              <h2>双人对弈棋盘</h2>
              <p>Canvas 渲染，支持 9x9 / 13x13 / 19x19，最近一手会标记。</p>
            </div>
            <div className={`status-pill ${snapshot.gameOver ? 'is-over' : ''}`}>
              {snapshot.gameOver ? '已终局' : `${PLAYER_LABEL[snapshot.currentPlayer]}落子`}
            </div>
          </div>
          <GoBoardCanvas game={game} onPlay={handlePlay} />
          <p className="notice-bar">{notice}</p>
        </div>

        <aside className="side-panel">
          <section className="panel-card">
            <h3>棋盘尺寸</h3>
            <div className="size-switcher">
              {BOARD_SIZES.map((size) => (
                <button
                  key={size}
                  type="button"
                  className={size === snapshot.size ? 'is-active' : ''}
                  onClick={() => startNewGame(size)}
                >
                  {size} x {size}
                </button>
              ))}
            </div>
            <div className="size-ai-list">
              {BOARD_SIZES.map((size) => (
                <div
                  key={`ai-${size}`}
                  className={`size-ai-item ${size === activeBoardSize ? 'is-active' : ''}`}
                >
                  <strong>{size} 路</strong>
                  <span className={BOARD_AI_CONFIG[size].trained ? 'ai-trained' : 'ai-untrained'}>
                    {BOARD_AI_CONFIG[size].label}
                  </span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel-card">
            <h3>操作</h3>
            <div className="action-grid">
              <button type="button" onClick={handlePass}>Pass</button>
              <button type="button" onClick={handleUndo}>悔棋</button>
              <button type="button" onClick={handleReset}>重开</button>
              <button
                type="button"
                disabled={!activeAISupport.trained || snapshot.gameOver || ai.status === 'loading' || ai.status === 'thinking'}
                onClick={() => {
                  void handleAISuggest()
                }}
              >
                {ai.status === 'thinking' ? 'AI 思考中' : 'AI 建议'}
              </button>
            </div>
          </section>

          <section className="panel-card">
            <h3>AI 状态</h3>
            <div className="ai-status-card">
              <div className="ai-status-row">
                <strong>{activeBoardSize}x{activeBoardSize}</strong>
                <span className={activeAISupport.trained ? 'ai-trained' : 'ai-untrained'}>
                  {activeAISupport.label}
                </span>
              </div>
              <p>{activeAISupport.note}</p>
              {activeAISupport.trained ? (
                <p className="panel-footnote">
                  当前接入的是浏览器侧 ONNX 单步建议，不包含完整 MCTS 搜索与自动对弈流程。
                </p>
              ) : null}
              {activeAISupport.trained && ai.error ? (
                <p className="ai-error">AI 加载失败：{ai.error}</p>
              ) : null}
              {activeAISupport.trained && ai.suggestion ? (
                <div className="ai-suggestion">
                  <strong>最新建议</strong>
                  <span>
                    {ai.suggestion.type === 'pass'
                      ? 'Pass'
                      : formatSuggestion(ai.suggestion.row, ai.suggestion.col, snapshot.size)}
                  </span>
                  <small>
                    置信度 {(ai.suggestion.confidence * 100).toFixed(1)}%，价值 {ai.suggestion.value.toFixed(3)}
                  </small>
                </div>
              ) : null}
            </div>
          </section>

          <section className="panel-card">
            <h3>局面信息</h3>
            <div className="score-grid">
              <article className="score-card dark">
                <span>黑棋</span>
                <strong>{score.blackScore.toFixed(1)}</strong>
                <small>提白 {blackCaptures}</small>
              </article>
              <article className="score-card light">
                <span>白棋</span>
                <strong>{score.whiteScore.toFixed(1)}</strong>
                <small>提黑 {whiteCaptures}</small>
              </article>
            </div>
            <p className="panel-footnote">
              这里展示的是面积计分预估；连续两次 Pass 后自动按当前局面判胜。
            </p>
          </section>

          <section className="panel-card">
            <h3>最近手顺</h3>
            <ol className="move-list">
              {recentMoves.length === 0 ? (
                <li className="move-empty">尚未落子</li>
              ) : (
                recentMoves.map((move) => (
                  <li key={`${move.moveNumber}-${move.boardSignature}`}>
                    <span>{move.moveNumber}.</span>
                    <strong>{PLAYER_LABEL[move.color]}</strong>
                    <span>{formatCoordinate(snapshot.size, move)}</span>
                    <em>{move.captured.length > 0 ? `提 ${move.captured.length}` : ''}</em>
                  </li>
                ))
              )}
            </ol>
          </section>

          <section className="panel-card">
            <h3>规则内核</h3>
            <ul className="rule-list">
              <li>落子先模拟提掉四邻无气敌块，再判断己方新连通块是否有气。</li>
              <li>局面签名会记录历史盘面，阻止打劫与重复局面。</li>
              <li>引擎层提供 `play / pass / undo / legalMoves / calculateAreaScore`，可直接接 AI。</li>
              <li>前端当前完成的是本地双人对弈；AI 仅 9 路提供浏览器侧单步建议，13/19 路明确标记为未训练。</li>
            </ul>
          </section>
        </aside>
      </section>
    </main>
  )
}

export default App
