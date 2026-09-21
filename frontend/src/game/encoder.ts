import type { GoSnapshot, StoneColor } from './goGame'

export function encodeBoard(snapshot: GoSnapshot, playerColor: StoneColor, channels: 3 | 4 = 3): Float32Array {
  const size = snapshot.size
  const state = new Float32Array(channels * size * size)
  const opponent = playerColor === 'black' ? 'white' : 'black'
  const planeSize = size * size

  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const cell = snapshot.grid[row][col]
      const offset = row * size + col
      if (cell === playerColor) {
        state[offset] = 1
      } else if (cell === opponent) {
        state[planeSize + offset] = 1
      }
    }
  }

  if (playerColor === 'black') {
    state.fill(1, planeSize * 2, planeSize * 3)
  }

  if (channels === 4 && snapshot.consecutivePasses > 0) {
    state.fill(1, planeSize * 3)
  }

  return state
}
