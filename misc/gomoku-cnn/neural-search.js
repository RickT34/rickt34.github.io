"use strict";

// Serial Negamax + Alpha-Beta. Leaves use the value head; policy selects candidates.
globalThis.GomokuSearch = (() => {
  const WIN = 2;
  const lineCache = new Map();
  class SearchTimeout extends Error {}

  class LRU extends Map {
    constructor(limit) {
      super();
      if (!Number.isInteger(limit) || limit<1) throw new Error("Invalid cache capacity");
      this.limit=limit;
    }
    get(key) {
      const value=super.get(key);
      if (value!==undefined) {super.delete(key);super.set(key,value);}
      return value;
    }
    set(key,value) {
      super.delete(key);super.set(key,value);
      while (this.size>this.limit) super.delete(this.keys().next().value);
      return this;
    }
  }

  function createCaches({network=4096,transpositions=20000,candidates=4096,ordering=10000}={}) {
    const caches={network:new LRU(network),transpositions:new LRU(transpositions),
      candidates:new LRU(candidates),ordering:new LRU(ordering)};
    caches.clear=()=>{for(const name of ["network","transpositions","candidates","ordering"]) caches[name].clear();};
    caches.sizes=()=>Object.fromEntries(["network","transpositions","candidates","ordering"].map(name=>[name,caches[name].size]));
    return caches;
  }

  // Full position keys avoid hash collisions. Network input is current-player relative.
  function stateKey(board,player=1) {
    return board.length+":"+String.fromCharCode(...Array.from(board,value=>value*player));
  }

  function config(input = {}) {
    if (!input || typeof input !== "object" || Array.isArray(input)
      || Object.keys(input).some(key => !["depth","top_p","top_k","time_limit","temperature"].includes(key)))
      throw new Error("搜索设置仅支持 depth、top_p、top_k、time_limit、temperature");
    const result = {depth:2, top_p:0.9, top_k:0, time_limit:3, temperature:0, ...input};
    if (!Number.isInteger(result.depth) || result.depth < 0 || result.depth > 100)
      throw new Error("搜索深度须为 0 到 100 的整数");
    if (typeof result.top_p !== "number" || !Number.isFinite(result.top_p) || result.top_p <= 0 || result.top_p > 1)
      throw new Error("Top-p 须在 (0, 1] 内");
    if (!Number.isInteger(result.top_k) || result.top_k < 0 || result.top_k > 625)
      throw new Error("Top-k 须为 0 到 625 的整数，0 表示不启用");
    if (typeof result.time_limit !== "number" || !Number.isFinite(result.time_limit) || result.time_limit < 0 || result.time_limit > 30)
      throw new Error("搜索时限须在 0 到 30 秒内，0 表示不限时");
    if (typeof result.temperature !== "number" || !Number.isFinite(result.temperature) || result.temperature < 0)
      throw new Error("温度 T 须为非负有限数");
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

  function distribution(logits, board, temperature=1) {
    if(logits.length!==board.length || !Number.isFinite(temperature) || temperature<0)
      throw new Error("模型输出或温度无效");
    const legal=Array.from(board,(_,i)=>i).filter(i=>board[i]===0);
    if(!legal.every(i=>Number.isFinite(logits[i])))throw new Error("模型输出无效");
    if (!legal.length) throw new Error("棋盘已满");
    const max=Math.max(...legal.map(i=>logits[i]));
    const probs=Array.from(board,(stone,i)=>stone?0:(temperature===0 ? Number(logits[i]===max) : Math.exp((logits[i]-max)/temperature)));
    const total=probs.reduce((a,b)=>a+b,0);
    return probs.map(p=>p/total);
  }

  function actions(board, size, player, logits, topP, topK = 0) {
    let ordered=Array.from(board,(_,i)=>i).filter(i=>board[i]===0)
      .sort((a,b)=>logits[b]-logits[a] || a-b);
    if (topP<1) {
      const probs=distribution(logits,board);
      let sum=0,count=0;
      do { sum+=probs[ordered[count++]]; } while(sum<topP && count<ordered.length);
      ordered=ordered.slice(0,count);
    }
    // Both filters select prefixes of the same ordering: the first threshold wins.
    if (topK>0) ordered=ordered.slice(0,topK);
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
    constructor(evaluate, options, {now=()=>performance.now(), onProgress=()=>{},
      caches=createCaches(), optimizations={}} = {}) {
      this.evaluate=evaluate;this.config=config(options);this.now=now;this.onProgress=onProgress;
      this.caches=caches;this.cache=caches.network;
      this.opt={transpositions:true,ordering:true,earlyExit:true,candidates:true,...optimizations};
      this.rulePrefix=this.config.top_p+"/"+this.config.top_k+":";
    }

    notify(force=false) {
      const now=this.now();
      if (force || now-this.lastProgress>=100) {
        this.lastProgress=now;
        this.onProgress({depth:this.iteration, completed_depth:this.completed,
          requested_depth:this.config.depth, milliseconds:now-this.started, ...this.stats});
      }
    }

    async network(board, player, key=stateKey(board,player)) {
      const cached=this.cache.get(key);
      if (cached) { this.stats.cache_hits++; return cached; }
      const result=await this.evaluate(Float32Array.from(board,value=>value*player));
      if (!result || result.logits.length!==board.length || !Array.from(result.logits).every(Number.isFinite)
        || !Number.isFinite(result.value) || result.value < -1 || result.value > 1)
        throw new Error("模型输出无效：搜索需要策略 logits 和 [-1, 1] 内的价值评分");
      this.stats.evaluations++;
      this.cache.set(key,result);
      this.notify();
      return result;
    }

    async candidates(board,player,key) {
      const ruleKey=this.rulePrefix+key;
      const cached=this.opt.candidates && this.caches.candidates.get(ruleKey);
      if (cached) {this.stats.candidate_cache_hits++;return [...cached];}
      const result=actions(board,this.size,player,(await this.network(board,player,key)).logits,this.config.top_p,this.config.top_k);
      if(this.opt.candidates)this.caches.candidates.set(ruleKey,result);
      return [...result];
    }

    reorder(candidates,key) {
      if(!this.opt.ordering)return;
      const previous=this.caches.ordering.get(this.rulePrefix+key);
      if(!previous)return;
      const index=candidates.indexOf(previous.action);
      if(index>0){candidates.splice(index,1);candidates.unshift(previous.action);this.stats.order_hits++;}
    }

    tableKey(key,depth) {return this.rulePrefix+depth+":"+key;}

    remember(key,depth,score,action,flag) {
      if(this.opt.transpositions) this.caches.transpositions.set(this.tableKey(key,depth),{score,action,flag,depth});
      if(this.opt.ordering){
        const orderKey=this.rulePrefix+key,previous=this.caches.ordering.get(orderKey);
        if(!previous || previous.depth<=depth)this.caches.ordering.set(orderKey,{action,depth});
      }
    }

    async visit(board, player, depth, alpha, beta, lastAction) {
      if (this.iteration>1 && this.now()>=this.deadline) throw new SearchTimeout();
      this.stats.nodes++;
      if (won(board,this.size,lastAction,-player)) return -WIN;
      if (!board.includes(0)) return 0;
      const key=stateKey(board,player);
      if (depth===0) return (await this.network(board,player,key)).value;
      // Exact depth is deliberate: a deeper horizon may have a different value.
      const saved=this.opt.transpositions && this.caches.transpositions.get(this.tableKey(key,depth));
      if(saved){
        this.stats.tt_hits++;
        if(saved.flag==="EXACT")return saved.score;
        if(saved.flag==="LOWER")alpha=Math.max(alpha,saved.score);
        else beta=Math.min(beta,saved.score);
        if(alpha>=beta){this.stats.cutoffs++;this.stats.tt_cutoffs++;return saved.score;}
      }
      const originalAlpha=alpha,originalBeta=beta;
      const candidates=await this.candidates(board,player,key);
      this.reorder(candidates,key);
      let best=-Infinity,bestAction=candidates[0];
      for (const action of candidates) {
        board[action]=player;
        let score;
        try { score=-await this.visit(board,-player,depth-1,-beta,-alpha,action); }
        finally { board[action]=0; }
        if(score>best){best=score;bestAction=action;}
        alpha=Math.max(alpha,score);
        if(this.opt.earlyExit && best===WIN){this.stats.cutoffs++;this.stats.win_cutoffs++;break;}
        if (alpha>=beta) {this.stats.cutoffs++;break;}
      }
      const flag=best===WIN ? "EXACT" : best<=originalAlpha ? "UPPER" : best>=originalBeta ? "LOWER" : "EXACT";
      this.remember(key,depth,best,bestAction,flag);
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
      this.stats={nodes:0,cutoffs:0,evaluations:0,cache_hits:0,tt_hits:0,tt_cutoffs:0,
        order_hits:0,win_cutoffs:0,candidate_cache_hits:0};
      this.completed=0;this.iteration=1;
        const key=stateKey(board,player);
        const root=await this.network(board,player,key);
        const candidates=await this.candidates(board,player,key);
        this.reorder(candidates,key);
        let action=candidates[0],score=-Infinity,timedOut=false,solved=false,rootScores=[];
        const sampling=this.config.temperature>0;
        if(this.config.depth===0)throw new Error("深度 0 应使用直接策略推理");
        for (this.iteration=1;this.iteration<=this.config.depth;this.iteration++) {
          this.notify(true);
          if(this.iteration>1 && this.now()>=this.deadline){timedOut=true;break;}
          const saved=this.opt.transpositions && this.caches.transpositions.get(this.tableKey(key,this.iteration));
          if(!sampling && saved?.flag==="EXACT" && candidates.includes(saved.action)){
            this.stats.tt_hits++;action=saved.action;score=saved.score;this.completed=this.iteration;
            candidates.splice(candidates.indexOf(action),1);candidates.unshift(action);
            if(!sampling && this.opt.earlyExit && score===WIN){solved=true;break;}
            continue;
          }
          let alpha=-Infinity,best=candidates[0],iterationScores=[];
          try {
            for (const candidate of candidates) {
              board[candidate]=player;
              let value;
              try { value=-await this.visit(board,-player,this.iteration-1,-Infinity,sampling ? Infinity : -alpha,candidate); }
              finally { board[candidate]=0; }
              iterationScores.push({action:candidate,score:value});
              if (value>alpha) {alpha=value;best=candidate;}
              if(!sampling && this.opt.earlyExit && alpha===WIN){this.stats.win_cutoffs++;break;}
            }
          } catch (error) {
            if (!(error instanceof SearchTimeout)) throw error;
            timedOut=true;break;
          }
          action=best;score=alpha;rootScores=iterationScores;this.completed=this.iteration;
          this.remember(key,this.iteration,score,action,"EXACT");
          candidates.splice(candidates.indexOf(best),1);candidates.unshift(best);
          this.notify(true);
          if(!sampling && this.opt.earlyExit && score===WIN){solved=true;break;}
        }
        const selection=sampling ? distribution(rootScores.map(item=>item.score),new Int8Array(rootScores.length),this.config.temperature) : null;
        return {probabilities:distribution(root.logits,board),
          choices:sampling ? rootScores.map((item,i)=>({...item,probability:selection[i]})) : null,
          search:{action,score,depth:this.completed,requested_depth:this.config.depth,timed_out:timedOut,
            milliseconds:this.now()-this.started,config:{...this.config},solved,...this.stats,
            cache_sizes:this.caches.sizes()}};
    }
  }

  return {Engine,config,actions,won,WIN,LRU,createCaches,stateKey,distribution};
})();
if (typeof module !== "undefined") module.exports=globalThis.GomokuSearch;
