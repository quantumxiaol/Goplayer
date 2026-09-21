import { expect, it } from 'vitest'
import { GoGame } from '../src/game/goGame'
import { encodeBoard } from '../src/game/encoder'

it('keeps legacy inputs and distinguishes pass without filling the color plane into pass', () => {
  const game = new GoGame(9)
  const legacy = encodeBoard(game.toSnapshot(), 'black')
  const initial = encodeBoard(game.toSnapshot(), 'black', 4)
  expect(initial.slice(0, 243)).toEqual(legacy)
  expect(Array.from(initial.slice(243))).toEqual(Array(81).fill(0))
  game.pass()
  const after = encodeBoard(game.toSnapshot(), 'white', 4)
  expect(Array.from(after.slice(162, 243))).toEqual(Array(81).fill(0))
  expect(Array.from(after.slice(243))).toEqual(Array(81).fill(1))
  game.undo()
  expect(encodeBoard(game.toSnapshot(), 'black', 4)).toEqual(initial)
})
