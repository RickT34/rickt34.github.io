"use strict";

window.gomokuBackend = (() => {
  let manifest, current, loadedHash;
  let progressHandler = () => {};
  const progress = details => progressHandler(details);
  const pending = new Map();
  let nextId = 0;
  const worker = new Worker(new URL("./inference-worker.js", document.baseURI));
  worker.onmessage = ({data}) => {
    const task = pending.get(data.id);
    if (!task) return;
    pending.delete(data.id);
    if (data.error) task.reject(new Error(data.error));
    else task.resolve(data.result);
  };
  worker.onerror = () => {
    for (const task of pending.values()) task.reject(new Error("推理线程失败，请刷新页面重试"));
    pending.clear();
  };

  function call(command, data, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, {resolve, reject});
      worker.postMessage({id, command, ...data}, transfer);
    });
  }

  async function models() {
    progress({label:"正在读取模型列表…"});
    const response = await fetch(new URL("./models.json", document.baseURI), {cache:"no-cache"});
    if (!response.ok) throw new Error("无法读取模型列表");
    const parsed = await response.json();
    if (!Array.isArray(parsed.models)) throw new Error("模型列表无效");
    manifest = parsed;
    return {models:manifest.models.map(model => model.name),
      details:manifest.models.map(({name, onnx_bytes, stage, description}) => ({name, onnx_bytes, stage, description}))};
  }

  async function load(entry) {
    if (loadedHash === entry.sha256) return;
    const url = new URL(entry.file, document.baseURI);
    if (url.origin !== location.origin) throw new Error("模型地址无效");
    progress({label:"正在下载模型…", detail:entry.name, loaded:0});
    const response = await fetch(url);
    if (!response.ok) throw new Error("模型加载失败");
    // Content-Length may describe compressed transfer bytes. Prefer the manifest's
    // uncompressed ONNX size, and use an indeterminate bar if neither is reliable.
    const encoded = response.headers.get("Content-Encoding");
    const declared = Number(response.headers.get("Content-Length"));
    const total = entry.onnx_bytes > 0 ? entry.onnx_bytes
      : !encoded || encoded === "identity" ? declared : 0;
    let bytes;
    if (response.body) {
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;
      try {
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          chunks.push(value);
          loaded += value.byteLength;
          progress({label:"正在下载模型…", detail:entry.name, loaded,
            total:loaded <= total ? total : 0});
        }
      } finally { reader.releaseLock(); }
      const joined = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) { joined.set(chunk, offset); offset += chunk.byteLength; }
      bytes = joined.buffer;
    } else {
      progress({label:"正在下载模型…", detail:entry.name});
      bytes = await response.arrayBuffer();
    }
    progress({label:"正在校验模型…", detail:entry.name});
    if (crypto.subtle) {
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
        byte => byte.toString(16).padStart(2, "0")).join("");
      if (hash !== entry.sha256) throw new Error("模型文件与发布清单不一致");
    }
    progress({label:"正在初始化模型…", detail:"首次加载可能需要稍等片刻"});
    await call("load", {bytes}, [bytes]);
    loadedHash = entry.sha256;
  }

  async function probabilities(game, label = "正在分析棋盘…") {
    if (game.env.done) return null;
    const state = game.env.observation();
    progress({label});
    const logits = await call("infer", {state, size:game.env.size});
    return maskedProbabilities(logits, state);
  }

  function place(game, action, decision = null) {
    const player = game.env.to_play;
    game.env.step(action);
    game.history.push({action, row:Math.floor(action / game.env.size), col:action % game.env.size,
      player, decision});
  }

  async function aiTurn(game) {
    if (game.env.done || game.env.to_play === game.human) return;
    const start = performance.now();
    const probs = await probabilities(game, "AI 正在思考…");
    const occupied = game.env.board.flat();
    const legal = probs.map((probability, action) => ({action, probability}))
      .filter(item => !occupied[item.action]);
    legal.sort((a,b) => b.probability - a.probability || a.action - b.action);
    let action = legal[0].action;
    if (game.mode === "sample") {
      const target = randomFloat(game);
      let cumulative = 0;
      action = legal.at(-1).action;
      for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (target < cumulative) { action = i; break; }
      }
    }
    place(game, action, {probability:probs[action], milliseconds:performance.now() - start,
      top:legal.slice(0,5)});
  }

  function snapshot(game) {
    return structuredClone({session:"browser", board:game.env.board, size:game.env.size,
      human:game.human, to_play:game.env.to_play, done:game.env.done, winner:game.env.winner,
      version:game.version, history:game.history, probabilities:game.probs,
      model:game.entry.name, mode:game.mode, seed:game.seed,
      can_undo:game.history.some(move => move.player === game.human)});
  }

  function validateSettings(human, mode, seed) {
    if (human !== 1 && human !== -1) throw new Error("执棋方无效");
    if (mode !== "greedy" && mode !== "sample") throw new Error("落子模式无效");
    if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32) throw new Error("随机种子无效");
  }

  async function restore(serialized) {
    const record = JSON.parse(serialized);
    validateSettings(record.human, record.mode, record.seed);
    const entry = manifest.models.find(model => model.name === record.model && model.sha256 === record.sha256);
    if (!entry || !Array.isArray(record.history) || record.history.length > entry.size ** 2
      || !Number.isInteger(record.version) || record.version < 0
      || !Number.isInteger(record.random_state) || record.random_state < 0 || record.random_state >= 2 ** 32)
      throw new Error("棋局记录无效");
    const game = {env:new BrowserGame(entry.size), entry, history:[], human:record.human,
      mode:record.mode, seed:record.seed, version:record.version, random_state:record.random_state};
    for (const move of record.history) {
      if (move.player !== game.env.to_play) throw new Error("棋局记录无效");
      place(game, move.action, move.decision);
    }
    if (!game.env.done && game.env.to_play !== game.human) throw new Error("棋局记录不完整");
    await load(entry);
    game.probs = await probabilities(game);
    current = game;
    return snapshot(game);
  }

  async function request(path, data) {
    if (path === "/api/new") {
      validateSettings(data.human, data.mode, data.seed);
      const entry = manifest.models.find(model => model.name === data.model);
      if (!entry) throw new Error("模型不存在");
      const game = {env:new BrowserGame(entry.size), entry, human:data.human, mode:data.mode,
        seed:data.seed, random_state:data.seed >>> 0, history:[], version:0};
      await load(entry);
      await aiTurn(game);
      game.probs = await probabilities(game);
      current = game;
      return snapshot(game);
    }
    if (path === "/api/state" && !current) return restore(data.session);
    if (!current) throw new Error("请开始新对局");
    const game = current;
    if (path === "/api/export") {
      return {format_version:1, size:game.env.size, human:game.human, mode:game.mode, seed:game.seed,
        done:game.env.done, winner:game.env.winner, rules:"freestyle: five or more; no forbidden moves",
        coordinates:"zero-based row and col; action = row * size + col",
        inference:"onnxruntime-web/wasm; browser Mulberry32 sampler",
        model:{model:game.entry.name, sha256:game.entry.sha256, source_sha256:game.entry.source_sha256,
          channels:game.entry.channels, training_config:{}}, moves:structuredClone(game.history)};
    }
    if (path === "/api/state") return snapshot(game);
    if (path !== "/api/move" && path !== "/api/undo") throw new Error("未知操作");
    if (data.version !== game.version) throw new Error("棋盘已更新");
    const history = structuredClone(game.history), random = game.random_state;
    try {
      if (path === "/api/move") {
        if (game.env.to_play !== game.human) throw new Error("还未轮到你落子");
        // A failed new-game load may have changed the worker's active model.
        await load(game.entry);
        place(game, data.action);
        await aiTurn(game);
      } else {
        const index = game.history.findLastIndex(move => move.player === game.human);
        if (index < 0) throw new Error("还没有可以撤回的落子");
        game.history = game.history.slice(0,index);
        game.env.reset();
        for (const move of game.history) game.env.step(move.action);
        await load(game.entry);
      }
      game.probs = await probabilities(game);
      game.version++;
      return snapshot(game);
    } catch (error) {
      game.history = history;
      game.random_state = random;
      game.env.reset();
      for (const move of history) game.env.step(move.action);
      throw error;
    }
  }

  function persist() {
    if (!current) return;
    try {
      sessionStorage.setItem("gomoku-static-session", JSON.stringify({model:current.entry.name,
        sha256:current.entry.sha256, human:current.human, mode:current.mode, seed:current.seed,
        random_state:current.random_state, version:current.version, history:current.history}));
    } catch { /* Play remains available when browser storage is disabled. */ }
  }

  return {models, request, persist, setProgressHandler(handler) { progressHandler = handler; }};
})();
