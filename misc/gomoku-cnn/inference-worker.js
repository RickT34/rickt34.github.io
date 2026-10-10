"use strict";
importScripts("./neural-search.js");

let session, runtime, modelBytes;
let caches=GomokuSearch.createCaches();

function updateThreadCount() {
  const actual=ort.env.wasm.numThreads;
  if(runtime.backend==="wasm" && actual<runtime.threads)
    runtime.note+=(runtime.note?"；":"")+"当前浏览器仅支持单线程计算";
  runtime.threads=actual;
}

async function initialize(mode="wasm-1") {
  if(runtime){
    if(runtime.requested!==mode)throw new Error("更换计算模式需要重启 Worker");
    return;
  }
  if(!["auto","wasm-1","wasm-4","webgpu"].includes(mode))throw new Error("未知计算模式");
  let adapter=null,note="",software=false;
  if(mode==="auto" || mode==="webgpu") {
    try {
      adapter=await navigator.gpu?.requestAdapter({powerPreference:"high-performance"});
      software=!!(adapter?.info?.isFallbackAdapter ?? adapter?.isFallbackAdapter)
        || /swiftshader|llvmpipe/i.test(adapter?.info?.description || "");
      if(mode==="auto" && software)adapter=null;
    } catch { adapter=null; }
    if(!adapter)note="GPU 不可用，已使用 CPU";
  }
  const backend=adapter?"webgpu":"wasm";
  const wanted=mode==="wasm-1"?1:Math.min(4,navigator.hardwareConcurrency || 1);
  const threads=backend==="webgpu" || !self.crossOriginIsolated ? 1 : wanted;
  if(backend==="wasm" && wanted>1 && threads===1)note+=(note?"；":"")+"站点未启用跨源隔离，使用单线程";
  importScripts(backend==="webgpu" ? "./vendor/ort.webgpu.min.js" : "./vendor/ort.wasm.min.js");
  ort.env.wasm.numThreads=threads;
  ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths=new URL("./vendor/",self.location.href).href;
  if(adapter)ort.env.webgpu.adapter=adapter;
  runtime={requested:mode,backend,threads,software:backend==="webgpu"&&software,
    isolated:!!self.crossOriginIsolated,note};
}

async function useCPU(reason) {
  console.warn("WebGPU fallback:",reason);
  const replacement=await ort.InferenceSession.create(modelBytes,{executionProviders:["wasm"]});
  const old=session;session=replacement;
  if(old)try {await old.release();}catch{}
  caches.clear();
  runtime={...runtime,backend:"wasm",software:false,note:"GPU 初始化或计算失败，已切换 CPU"};
  updateThreadCount();
}

async function evaluate(state,size) {
  if(!session)throw new Error("模型尚未加载");
  const tensor=new ort.Tensor("float32",state,[1,size,size]);
  const outputs=await session.run({[session.inputNames[0]]:tensor});
  const logits=outputs[session.outputNames[0]].data.slice();
  const value=session.outputNames.includes("value") ? Number(outputs.value.data[0]) : null;
  if(!Array.from(logits).every(Number.isFinite) || (value!==null && (!Number.isFinite(value)||Math.abs(value)>1)))
    throw new Error("模型输出无效");
  return {logits,value};
}

async function execute(data) {
  if(data.command==="infer") {
    const key=GomokuSearch.stateKey(data.state);
    let output=caches.network.get(key);
    if(!output){output=await evaluate(data.state,data.size);caches.network.set(key,output);}
    return output.logits.slice();
  }
  if(data.command==="search") {
    if(!session?.outputNames.includes("value"))throw new Error("此模型没有价值头，无法搜索");
    const engine=new GomokuSearch.Engine(state=>evaluate(state,data.size),data.config,{
      caches:data.persistent===false?GomokuSearch.createCaches():caches,
      optimizations:data.optimizations,
      onProgress:stats=>self.postMessage({id:data.id,progress:stats,runtime})});
    return engine.search(data.board,data.player,data.size);
  }
  if(data.command==="clear-cache"){caches.clear();return true;}
  if(data.command==="diagnostics")return {runtime,cache_sizes:caches.sizes()};
  throw new Error("未知推理操作");
}

self.onmessage=async ({data})=>{
  try {
    if(data.command==="infer" && (!Number.isInteger(data.size) || data.size<5 || data.size>25
      || data.state?.length!==data.size**2 || !Array.from(data.state).every(value=>[-1,0,1].includes(value))))
      throw new Error("输入棋盘无效");
    if(data.command==="load") {
      await initialize(data.runtime);
      const bytes=new Uint8Array(data.bytes);
      let replacement;
      try {replacement=await ort.InferenceSession.create(bytes,{executionProviders:[runtime.backend]});}
      catch(error) {
        if(runtime.backend!=="webgpu")throw error;
        modelBytes=bytes;
        await useCPU(error.message);
        replacement=session;
      }
      const old=session;session=replacement;modelBytes=bytes;
      if(old && old!==replacement)await old.release();
      caches.clear();
      updateThreadCount();
      self.postMessage({id:data.id,result:{has_value:session.outputNames.includes("value"),runtime},runtime});
      return;
    }
    let result;
    try {result=await execute(data);}
    catch(error) {
      if(runtime?.backend!=="webgpu" || !["infer","search"].includes(data.command))throw error;
      await useCPU(error.message);
      self.postMessage({id:data.id,notice:"GPU 暂不可用，正在使用 CPU 重试",runtime});
      result=await execute(data);
    }
    self.postMessage({id:data.id,result,runtime},result instanceof Float32Array?[result.buffer]:[]);
  } catch(error) {
    self.postMessage({id:data.id,error:error.message || String(error),runtime});
  }
};
