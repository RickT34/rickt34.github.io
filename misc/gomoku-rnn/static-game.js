"use strict";

class BrowserGame {
  constructor(size = 19) {
    if (!Number.isInteger(size) || size < 5 || size > 25) throw new Error("棋盘大小无效");
    this.size = size;
    this.reset();
  }

  reset() {
    this.board = Array.from({length:this.size}, () => Array(this.size).fill(0));
    this.to_play = 1;
    this.winner = 0;
    this.done = false;
  }

  step(action) {
    if (this.done) throw new Error("本局已结束");
    if (!Number.isInteger(action) || action < 0 || action >= this.size ** 2) throw new Error("落子位置超出棋盘");
    const row = Math.floor(action / this.size), col = action % this.size;
    if (this.board[row][col]) throw new Error("该位置已有棋子");
    const player = this.to_play;
    this.board[row][col] = player;
    for (const [dr, dc] of [[0,1], [1,0], [1,1], [1,-1]]) {
      let count = 1;
      for (const sign of [-1,1]) {
        let r = row + dr * sign, c = col + dc * sign;
        while (r >= 0 && c >= 0 && r < this.size && c < this.size && this.board[r][c] === player) {
          count++;
          r += dr * sign;
          c += dc * sign;
        }
      }
      if (count >= 5) this.winner = player;
    }
    this.done = !!this.winner || this.board.every(line => line.every(Boolean));
    this.to_play *= -1;
  }

  observation() {
    return Float32Array.from(this.board.flat(), stone => stone * this.to_play);
  }
}

function maskedProbabilities(logits, state) {
  if (logits.length !== state.length) throw new Error("模型输出大小不匹配");
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (!state[i]) {
      if (!Number.isFinite(logits[i])) throw new Error("模型输出无效");
      max = Math.max(max, logits[i]);
    }
  }
  if (!Number.isFinite(max)) throw new Error("棋盘已满");
  const probs = Array.from(logits, (value, i) => state[i] ? 0 : Math.exp(value - max));
  const total = probs.reduce((sum, p) => sum + p, 0);
  return probs.map(p => p / total);
}

// Mulberry32 gives each game its own reproducible browser sampling sequence.
function randomFloat(game) {
  game.random_state = (game.random_state + 0x6D2B79F5) >>> 0;
  let value = game.random_state;
  value = Math.imul(value ^ value >>> 15, value | 1);
  value ^= value + Math.imul(value ^ value >>> 7, value | 61);
  return ((value ^ value >>> 14) >>> 0) / 4294967296;
}

if (typeof module !== "undefined") module.exports = {BrowserGame, maskedProbabilities, randomFloat};
