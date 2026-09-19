import { describe, expect, it } from 'vitest'
import { GoGame } from '../src/game/goGame'

describe('13x13 draw', () => {
  it('scores a legal double-Pass tied game as a draw', () => {
    const game = new GoGame(13)
    const moves: ([number, number] | null)[] = [
      [0, 2], [0, 0], [1, 1], [0, 1], [1, 0], [0, 0], null, [8, 8], [8, 7], null, null,
    ]
    for (const move of moves) {
      expect(move ? game.play(...move).ok : game.pass().ok).toBe(true)
    }
    expect(game.gameOver).toBe(true)
    expect(game.calculateAreaScore()).toEqual({ blackScore: 4, whiteScore: 4, winner: 'draw' })
  })
})
