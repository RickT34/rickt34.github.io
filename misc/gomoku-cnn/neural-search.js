"use strict";

// Serial Negamax + Alpha-Beta. Leaves use the value head; policy selects candidates.
globalThis.GomokuSearch = (() => {
  const WIN = 2;
  const lineCache = new Map();
  class SearchTimeout extends Error {}

  function config(input = {}) {
    if (!input || typeof input !== "object" || Array.isArray(input)
      || Object.keys(input).some(key => !["depth","top_p","time_limit"].includes(key)))
      throw new Error("搜索设置仅支持 depth、top_p、time_limit");
    const result = {depth:2, top_p:0.9, time_limit:3, ...input};
    if (!Number.isInteger(result.depth) || result.depth < 1 || result.depth > 6)
      throw new Error("搜索深度须为 1 到 6 的整数");
    if (typeof result.top_p !== "number" || !Number.isFinite(result.top_p) || result.top_p <= 0 || result.top_p > 1)
      throw new Error("Top-p 须在 (0, 1] 内");
    if (typeof result.time_limit !== "number" || !Number.isFinite(result.time_limit) || result.time_limit < 0 || result.time_limit > 30)
      throw new Error("搜索时限须在 0 到 30 秒内，0 表示不限时");
    return result;
  }

  function lines(size) {
    if (!lineCache.has(size)) {
      const result = [];
      for (let r=0;r<size;r++) for (let c=0;c<size;c++)
        for (const [dr,dc] of [[0,1],[1,0],[1,1],[1,-1]]) {
          const endR=r+4*dr,endC=c+4*dc;
          if (endR>=0 && endR<size && endC>=0 && endC<size)
            result.push(Array.from({length:5},(_,i)=>(r+i*dr)*size+c+i*dc));
        }
      lineCache.set(size,result);
    }
    return lineCache.get(size);
  }

  function won(board, size, action, player) {
    const row=Math.floor(action/size),col=action%size;
    for (const [dr,dc] of [[0,1],[1,0],[1,1],[1,-1]]) {
      let count=1;
      for (const sign of [-1,1]) {
        let r=row+sign*dr,c=col+sign*dc;
        while (r>=0 && r<size && c>=0 && c<size && board[r*size+c]===player) {
          count++;r+=sign*dr;c+=sign*dc;
        }
      }
      if (count>=5) return true;
    }
    return false;
  }

  function distribution(logits, board) {
    const legal=Array.from(board,(_,i)=>i).filter(i=>board[i]===0);
    if (!legal.length) throw new Error("棋盘已满");
    const max=Math.max(...legal.map(i=>logits[i]));
    const probs=Array.from(board,(stone,i)=>stone?0:Math.exp(logits[i]-max));
    const total=probs.reduce((a,b)=>a+b,0);
    return probs.map(p=>p/total);
  }

  function actions(board, size, player, logits, topP) {
    let ordered=Array.from(board,(_,i)=>i).filter(i=>board[i]===0)
      .sort((a,b)=>logits[b]-logits[a] || a-b);
    if (topP<1) {
      const probs=distribution(logits,board);
      let sum=0,count=0;
      do { sum+=probs[ordered[count++]]; } while(sum<topP && count<ordered.length);
      ordered=ordered.slice(0,count);
    }
    const wins=new Set(),blocks=new Set();
    for (const line of lines(size)) {
      let empty=-1,emptyCount=0,own=0,opponent=0;
      for (const a of line) {
        if (!board[a]) {empty=a;emptyCount++;}
        else if (board[a]===player) own++;
        else opponent++;
      }
      if (emptyCount===1 && own===4) wins.add(empty);
      if (emptyCount===1 && opponent===4) blocks.add(empty);
    }
    const priority=a=>wins.has(a)?0:blocks.has(a)?1:2;
    return [...new Set([...ordered,...wins,...blocks])]
      .sort((a,b)=>priority(a)-priority(b) || logits[b]-logits[a] || a-b);
  }

  class Engine {
    constructor(evaluate, options, {now=()=>performance.now(), onProgress=()=>{}} = {}) {
      this.evaluate=evaluate;this.config=config(options);this.now=now;this.onProgress=onProgress;
    }

    notify(force=false) {
      const now=this.now();
      if (force || now-this.lastProgress>=100) {
        this.lastProgress=now;
        this.onProgress({depth:this.iteration, completed_depth:this.completed,
          requested_depth:this.config.depth, milliseconds:now-this.started, ...this.stats});
      }
    }

    async network(board, player) {
      const key=player+":"+String.fromCharCode(...board);
      if (this.cache.has(key)) { this.stats.cache_hits++; return this.cache.get(key); }
      const result=await this.evaluate(Float32Array.from(board,value=>value*player));
      if (!result || result.logits.length!==board.length || !Array.from(result.logits).every(Number.isFinite)
        || !Number.isFinite(result.value) || result.value < -1 || result.value > 1)
        throw new Error("模型输出无效：搜索需要策略 logits 和 [-1, 1] 内的价值评分");
      this.stats.evaluations++;
      this.cache.set(key,result);
      this.notify();
      return result;
    }

    async visit(board, player, depth, alpha, beta, lastAction) {
      if (this.iteration>1 && this.now()>=this.deadline) throw new SearchTimeout();
      this.stats.nodes++;
      if (won(board,this.size,lastAction,-player)) return -WIN;
      if (!board.includes(0)) return 0;
      if (depth===0) return (await this.network(board,player)).value;
      const candidates=actions(board,this.size,player,(await this.network(board,player)).logits,this.config.top_p);
      let best=-Infinity;
      for (const action of candidates) {
        board[action]=player;
        let score;
        try { score=-await this.visit(board,-player,depth-1,-beta,-alpha,action); }
        finally { board[action]=0; }
        best=Math.max(best,score);alpha=Math.max(alpha,score);
        if (alpha>=beta) {this.stats.cutoffs++;break;}
      }
      return best;
    }

    async search(source, player, size) {
      if (!Number.isInteger(size) || size<5 || size>25 || source.length!==size*size
        || !Array.from(source).every(value=>[-1,0,1].includes(value)) || ![-1,1].includes(player))
        throw new Error("棋盘或当前行棋方无效");
      const board=Int8Array.from(source);
      if (!board.includes(0) || lines(size).some(line=>board[line[0]] && line.every(a=>board[a]===board[line[0]])))
        throw new Error("不能在终局上搜索");
      this.size=size;this.started=this.now();this.lastProgress=-Infinity;
      this.deadline=this.config.time_limit ? this.started+this.config.time_limit*1000 : Infinity;
      this.stats={nodes:0,cutoffs:0,evaluations:0,cache_hits:0};
      this.cache=new Map();this.completed=0;this.iteration=1;
      try {
        const root=await this.network(board,player);
        const candidates=actions(board,size,player,root.logits,this.config.top_p);
        let action=candidates[0],score=-Infinity,timedOut=false;
        for (this.iteration=1;this.iteration<=this.config.depth;this.iteration++) {
          this.notify(true);
          let alpha=-Infinity,best=candidates[0];
          try {
            for (const candidate of candidates) {
              board[candidate]=player;
              let value;
              try { value=-await this.visit(board,-player,this.iteration-1,-Infinity,-alpha,candidate); }
              finally { board[candidate]=0; }
              if (value>alpha) {alpha=value;best=candidate;}
            }
          } catch (error) {
            if (!(error instanceof SearchTimeout)) throw error;
            timedOut=true;break;
          }
          action=best;score=alpha;this.completed=this.iteration;
          candidates.splice(candidates.indexOf(best),1);candidates.unshift(best);
          this.notify(true);
        }
        return {probabilities:distribution(root.logits,board),
          search:{action,score,depth:this.completed,requested_depth:this.config.depth,timed_out:timedOut,
            milliseconds:this.now()-this.started,config:{...this.config},...this.stats}};
      } finally { this.cache.clear(); }
    }
  }

  return {Engine,config,actions,won,WIN};
})();
if (typeof module !== "undefined") module.exports=globalThis.GomokuSearch;
