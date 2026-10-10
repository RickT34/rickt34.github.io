"use strict";
importScripts("./vendor/ort.wasm.min.js");
importScripts("./neural-search.js");

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL("./vendor/", self.location.href).href;
let session;

self.onmessage = async ({data}) => {
  const {id, command} = data;
  try {
    if (command === "load") {
      const replacement = await ort.InferenceSession.create(data.bytes, {executionProviders:["wasm"]});
      const old = session;
      session = replacement;
      if (old) await old.release();
      self.postMessage({id, result:{has_value:session.outputNames.includes("value")}});
    } else if (command === "infer") {
      if (!session) throw new Error("模型尚未加载");
      const tensor = new ort.Tensor("float32", data.state, [1, data.size, data.size]);
      const outputs = await session.run({[session.inputNames[0]]:tensor});
      const result = outputs[session.outputNames[0]].data.slice();
      self.postMessage({id, result}, [result.buffer]);
    } else if (command === "search") {
      if (!session || !session.outputNames.includes("value")) throw new Error("此模型没有价值头，无法搜索");
      const engine=new GomokuSearch.Engine(async state => {
        const tensor=new ort.Tensor("float32",state,[1,data.size,data.size]);
        const outputs=await session.run({[session.inputNames[0]]:tensor});
        return {logits:outputs[session.outputNames[0]].data.slice(),value:Number(outputs.value.data[0])};
      }, data.config, {onProgress:stats=>self.postMessage({id,progress:stats})});
      const result=await engine.search(data.board,data.player,data.size);
      self.postMessage({id,result});
    } else throw new Error("未知推理操作");
  } catch (error) {
    self.postMessage({id, error:error.message || String(error)});
  }
};
