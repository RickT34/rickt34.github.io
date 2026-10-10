"use strict";

window.gomokuBackend = (() => {
  let manifest, current, loadedHash, loadedHasValue = false;
  let progressHandler = () => {};
  let stateHandler = () => {};
  let runtimeHandler = () => {};
  const progress = details => progressHandler(details);
  const pending = new Map();
  let nextId = 0;
  let worker,workerMode,workerFailure,runtimeInfo=null,downloadCache=null;

  function runtimeMode(value="auto") {
    if(!["auto","wasm-1","wasm-4","webgpu"].includes(value))throw new Error("未知计算模式");
    return value;
  }

  function startWorker(mode) {
    if(worker && workerMode===mode && !workerFailure)return;
    worker?.terminate();
    for(const task of pending.values())task.reject(new Error("计算引擎已重启"));
    pending.clear();
    workerMode=mode;workerFailure=null;loadedHash=null;loadedHasValue=false;runtimeInfo=null;
    worker=new Worker(new URL("./inference-worker.js",document.baseURI));
    worker.onmessage = ({data}) => {
    const task = pending.get(data.id);
    if (!task) return;
    if(data.runtime){runtimeInfo=data.runtime;runtimeHandler(runtimeInfo);}
    if(data.notice){progress({label:data.notice});return;}
    if (data.progress) {
      const stats=data.progress;
      progress({label:"AI 正在搜索…",detail:`第 ${stats.depth} 层 · 已完成 ${stats.completed_depth} 层 · ${stats.nodes} 节点 · ${stats.evaluations} 次评估`});
      return;
    }
    pending.delete(data.id);
    if (data.error) {
      const error=new Error(data.error);
      if(task.command==="load"){loadedHash=null;workerFailure=error;}
      task.reject(error);
    }
    else task.resolve(data.result);
  };
  worker.onerror = () => {
    workerFailure=new Error("推理线程失败，请开始新局重试");
    loadedHash=null;
    for (const task of pending.values()) task.reject(workerFailure);
    pending.clear();
  };
  }
  startWorker("auto");

  function call(command, data, transfer = []) {
    if(workerFailure)return Promise.reject(workerFailure);
    return new Promise((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, {resolve, reject, command});
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
      details:manifest.models.map(({name, onnx_bytes, stage, description, has_value}) => ({name, onnx_bytes, stage, description, has_value}))};
  }

  async function load(entry, mode="auto") {
    startWorker(runtimeMode(mode));
    if (loadedHash === entry.sha256) return;
    const url = new URL(entry.file, document.baseURI);
    if (url.origin !== location.origin) throw new Error("模型地址无效");
    let bytes;
    if(downloadCache?.sha256===entry.sha256)bytes=downloadCache.bytes;
    else {
    progress({label:"正在下载模型…", detail:entry.name, loaded:0});
    const response = await fetch(url);
    if (!response.ok) throw new Error("模型加载失败");
    // Content-Length may describe compressed transfer bytes. Prefer the manifest's
    // uncompressed ONNX size, and use an indeterminate bar if neither is reliable.
    const encoded = response.headers.get("Content-Encoding");
    const declared = Number(response.headers.get("Content-Length"));
    const total = entry.onnx_bytes > 0 ? entry.onnx_bytes
      : !encoded || encoded === "identity" ? declared : 0;
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
    downloadCache={sha256:entry.sha256,bytes};
    }
    progress({label:"正在初始化模型…", detail:"首次加载可能需要稍等片刻"});
    const input=bytes.slice(0);
    const capabilities=await call("load", {bytes:input,runtime:mode}, [input]);
    loadedHasValue=capabilities?.has_value === true;
    loadedHash = entry.sha256;
  }

  async function probabilities(game, label = "正在分析棋盘…") {
    if (game.env.done) return null;
    if (game.probs) return game.probs;
    const state = game.env.observation();
    progress({label});
    const logits = await call("infer", {state, size:game.env.size});
    game.probs = GomokuSearch.distribution(logits, state, game.temperature || 1);
    return game.probs;
  }

  function place(game, action, decision = null) {
    const player = game.env.to_play;
    game.env.step(action);
    game.probs = null;
    game.history.push({action, row:Math.floor(action / game.env.size), col:action % game.env.size,
      player, decision});
  }

  async function aiTurn(game) {
    if (game.env.done || game.env.to_play === game.human) return;
    const start = performance.now();
    let probs, searchResult, choices;
    if (game.search.depth > 0) {
      if (!loadedHasValue) throw new Error("此模型没有价值头，无法搜索");
      progress({label:"AI 正在搜索…"});
      const result=await call("search",{board:game.env.board.flat(),size:game.env.size,
        player:game.env.to_play,config:game.search});
      probs=result.probabilities;
      searchResult=result.search;
      choices=result.choices;
    } else probs = await probabilities(game, "AI 正在思考…");
    const occupied = game.env.board.flat();
    const legal = probs.map((probability, action) => ({action, probability}))
      .filter(item => !occupied[item.action]);
    legal.sort((a,b) => b.probability - a.probability || a.action - b.action);
    let action = searchResult ? searchResult.action : legal[0].action;
    if (game.temperature > 0) {
      const candidates=choices || legal;
      const target=randomFloat(game);
      let cumulative=0;
      action=candidates.at(-1).action;
      for(const item of candidates){
        cumulative+=item.probability;
        if(target<cumulative){action=item.action;break;}
      }
      if(searchResult)searchResult={...searchResult,action,score:choices.find(item=>item.action===action).score};
    }
    place(game, action, {probability:probs[action], milliseconds:performance.now() - start,
      top:legal.slice(0,5), runtime:runtimeInfo, ...(searchResult ? {search:searchResult} : {})});
  }

  function snapshot(game) {
    return structuredClone({session:"browser", board:game.env.board, size:game.env.size,
      human:game.human, to_play:game.env.to_play, done:game.env.done, winner:game.env.winner,
      version:game.version, history:game.history, probabilities:game.probs ?? null,
      model:game.entry.name, mode:game.mode, seed:game.seed, temperature:game.temperature,
      search:game.search ?? null,
      runtime:runtimeInfo, runtime_request:game.runtime_request,
      can_undo:game.history.some(move => move.player === game.human)});
  }

  function validateSettings(human, mode, seed) {
    if (human !== 1 && human !== -1) throw new Error("执棋方无效");
    if (!["unified","greedy","sample","minimax"].includes(mode)) throw new Error("落子模式无效");
    if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32) throw new Error("随机种子无效");
  }

  function strategy(game, data) {
    const temperature=data.temperature ?? (data.mode==="sample" ? 1 : 0);
    game.search=GomokuSearch.config({...((data.mode==="minimax" || data.mode==="unified") ? data.search : {depth:0}),temperature});
    game.temperature=game.search.temperature;
    game.mode="unified";
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
    strategy(game,record);
    game.runtime_request=runtimeMode(record.runtime_request ?? "auto");
    for (const move of record.history) {
      if (move.player !== game.env.to_play) throw new Error("棋局记录无效");
      place(game, move.action, move.decision);
    }
    if (!game.env.done && game.env.to_play !== game.human) throw new Error("棋局记录不完整");
    await load(entry,game.runtime_request);
    if (game.search.depth > 0 && !loadedHasValue) throw new Error("此模型没有价值头，无法搜索");
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
      strategy(game,data);
      game.runtime_request=runtimeMode(data.runtime ?? "auto");
      await load(entry,game.runtime_request);
      if (game.search.depth > 0 && !loadedHasValue) throw new Error("此模型没有价值头，无法搜索");
      await aiTurn(game);
      current = game;
      return snapshot(game);
    }
    if (path === "/api/state" && !current) return restore(data.session);
    if (!current) throw new Error("请开始新对局");
    const game = current;
    if (path === "/api/export") {
      return {format_version:1, size:game.env.size, human:game.human, mode:game.mode, seed:game.seed, temperature:game.temperature,
        search:game.search ?? null,
        done:game.env.done, winner:game.env.winner, rules:"freestyle: five or more; no forbidden moves",
        coordinates:"zero-based row and col; action = row * size + col",
        inference:`onnxruntime-web/${runtimeInfo?.backend || "wasm"}; browser Mulberry32 sampler`,
        runtime:runtimeInfo,
        model:{model:game.entry.name, sha256:game.entry.sha256, source_sha256:game.entry.source_sha256,
          channels:game.entry.channels, training_config:{}}, moves:structuredClone(game.history)};
    }
    if (path === "/api/state") return snapshot(game);
    if (path === "/api/probabilities") {
      if (data.version !== game.version) throw new Error("棋盘已更新");
      if (!game.env.done && !game.probs) {
        await load(game.entry,game.runtime_request);
        await probabilities(game);
      }
      return snapshot(game);
    }
    if (path !== "/api/move" && path !== "/api/undo") throw new Error("未知操作");
    if (data.version !== game.version) throw new Error("棋盘已更新");
    const history = structuredClone(game.history), random = game.random_state, probs = game.probs;
    try {
      if (path === "/api/move") {
        if (game.env.to_play !== game.human) throw new Error("还未轮到你落子");
        place(game, data.action);
        // Publish the validated human move without persisting the unfinished turn.
        stateHandler(snapshot(game));
        if (!game.env.done) {
          // A failed new-game load may have changed the worker's active model.
          await load(game.entry,game.runtime_request);
          await aiTurn(game);
        }
      } else {
        const index = game.history.findLastIndex(move => move.player === game.human);
        if (index < 0) throw new Error("还没有可以撤回的落子");
        game.history = game.history.slice(0,index);
        game.env.reset();
        for (const move of game.history) game.env.step(move.action);
        game.probs = null;
      }
      game.version++;
      return snapshot(game);
    } catch (error) {
      game.history = history;
      game.random_state = random;
      game.env.reset();
      for (const move of history) game.env.step(move.action);
      game.probs = probs;
      stateHandler(snapshot(game));
      throw error;
    }
  }

  function persist() {
    if (!current) return;
    try {
      sessionStorage.setItem("gomoku-static-session", JSON.stringify({model:current.entry.name,
        sha256:current.entry.sha256, human:current.human, mode:current.mode, seed:current.seed, temperature:current.temperature,
        search:current.search ?? null,
        runtime_request:current.runtime_request,
        random_state:current.random_state, version:current.version, history:current.history}));
    } catch { /* Play remains available when browser storage is disabled. */ }
  }

  return {models, request, persist,
    setProgressHandler(handler) { progressHandler = handler; },
    setRuntimeHandler(handler) { runtimeHandler=handler; },
    setStateHandler(handler) { stateHandler = handler; }};
})();
