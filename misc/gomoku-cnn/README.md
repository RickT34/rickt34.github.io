# 五子棋纯静态网站

此目录可以直接部署到静态网站托管平台。用户仅需现代浏览器，无需安装 Python、PyTorch、Node.js 或其他软件。模型在浏览器的后台 Worker 中使用 CPU / WebAssembly 推理。页面、推理运行库和模型均随包提供，不使用 CDN，不调用后端 API 或外部推理服务。

## 部署

解压 ZIP，将其中包含 `index.html` 的目录内容上传到静态托管平台。可以部署在域名根目录或子目录，例如 `https://example.com/gomoku/`。目录访问需使用尾部斜杠，或直接访问 `index.html`。

保持 `vendor/` 中的 JS、MJS 和 WASM 文件配套；托管平台应正常提供 `.mjs`（JavaScript）和 `.wasm`（WebAssembly）文件。默认单线程，无需配置跨源隔离响应头。推荐 HTTPS。

本地预览仅需任意静态文件服务器，例如在此目录运行：

```bash
python -m http.server 8003 --bind 127.0.0.1
```

然后打开 <http://127.0.0.1:8003/>。这里 Python 只用于开发者预览，部署后的用户不需要它。此版本通过 HTTP/HTTPS 加载模型和 Worker，不能直接双击 HTML 使用。

## 模型与对局

模型下拉列表来自 `models.json`。支持执黑/执白、最大概率/采样、悔一轮、落子概率热力图、JSON 棋谱导出和刷新页面后恢复本局。每个新对局使用当时选择的模型。页面刷新后如模型哈希已改变，会开始新局。

浏览器版的贪心／采样模式每次 AI 决策只进行一次网络推理，右侧候选列表复用这次结果。点击落子后，人类棋子立即显示，再等待 AI 应手；推理失败时回滚这一轮。概率图关闭时，不计算下一局面的概率：执黑新局、悔棋、恢复对局均不额外推理。开启概率图才计算当前行棋方的偏好，同一局面反复开关复用缓存；若保持开启，每轮 AI 应手后会为新的棋盘另计算一次概率图，因为这与 AI 决策前是不同的局面。

v0.4.0 新增浏览器内的“Minimax + αβ · 神经网络评分”。选择模式后可设置 Top-p（1–100%）、深度（1–6 手）和软预算（0–30 秒，0 为不限时）。策略头筛选候选，价值头评估叶节点；搜索会进行多次网络前向，根节点结果复用于候选列表。Top-p 之外仍保留一步成五和防五落点。

预算为软时限：至少完整计算一层，之后超时采用上一完整层。因此第一层或单次推理可能使实际耗时超过预算，100% Top-p 尤其可能较慢。搜索在 Worker 中串行计算，页面保持响应；进度显示实际完成层数与评估次数。终局评分 ±2 优先于网络价值 [-1,1]，评分不是胜率。旧的纯策略 ONNX 不支持此模式，需重新从 `.pt` 导出带价值头的模型。

操作超过约 500 毫秒才在右下角显示悬浮进度卡片，快速操作不显示，也不会推动棋盘或改变页面布局。模型下载时显示实际下载大小与百分比；无法获知总大小时仅显示已下载量。校验、初始化、AI 思考和棋盘分析使用不定进度条及已用时间，不估算虚假的计算完成率。完成或失败后立即收起，加载失败时可以再次开始新局重试。

模型输入是当前行棋方视角的棋盘：己方 +1、对手 -1、空位 0。已落子的交叉点在浏览器中屏蔽，再对网络 logits 做 softmax。规则为自由五子棋，连续五子及以上获胜，无禁手。贪心／采样模式完全按策略头落子；Minimax 模式结合终局规则和价值头搜索，不加入 C 教师或手工局面评分。

浏览器采样使用独立的 Mulberry32 随机数序列，同模型、同种子、同操作可复现；该序列与 PyTorch 不同，因此采样棋谱不保证与 Python 版一致。FP32 跨推理引擎有浮点误差，极接近的候选概率仍可能改变排序。`models.json` 记录发布时数值对照及贪心落点验证结果。

已加载资源后，当前页面的对弈操作不依赖网络。离线刷新或重新打开页面仍需资源缓存或静态服务器；本版本不包含离线安装功能。

## 从训练仓库重新生成

### 给已有网站添加新模型（推荐）

在训练仓库根目录运行，首次使用需要在同一 Python 环境安装 `onnx` 和 `onnxruntime`：

```bash
uv pip install --python .venv/bin/python onnx onnxruntime
.venv/bin/python scripts/add_static_model.py runs/new-model.pt \
  --site dist/gomoku-static-v0.1.0 \
  --output dist/gomoku-static-v0.2.0
```

将 `runs/new-model.pt` 换成实际路径，可以在同一条命令中依次给出多个 `.pt`。只支持本项目的 CNN/residual 单模型 checkpoint；完整群体断点需要先导出成员。

脚本会剥离优化器和训练配置，转换为 FP32 ONNX，并对照原模型验证 16 个局面的输出和贪心落点。新版默认保留残差模型的价值头以支持搜索，同时验证价值输出；旧 CNN 模型仍只有策略头。它复制现有网站，保留原来的页面及模型，更新 `models.json`，并生成：

- `gomoku-static-v0.2.0/`：可直接部署的完整网站。
- `gomoku-static-v0.2.0.zip`：完整网站的无损 ZIP 压缩包，解压内容直接作为网站根目录。
- `gomoku-static-v0.2.0-models.zip`：只包含本次转换的模型和完整的新 `models.json`，用于更新已有网站。

**更新已部署的网站：** 解压 `-models.zip`，先上传其中的 `models/` 文件，再上传 `models.json`；然后在页面刷新模型列表、选择新模型并开始新局。增量包必须应用到 `--site` 对应的原网站；它不包含旧模型或运行库，不能单独部署。下次添加模型时，将 `--site` 指向上一次生成的网站目录。

使用 `--name stronger-v2` 可指定网页显示的名称（仅限一次添加一个模型）。同名模型默认拒绝替换；需要替换时加 `--replace`，替换仅发生在新的输出网站中。模型文件名带内容哈希，减少更新后的浏览器缓存冲突。原 `.pt` 和原网站目录保持原样，已有输出也不会被覆盖。

新版选择框显示 ONNX 文件大小及训练阶段。单个模型导入时可加 `--stage "历史对手池强化学习" --description "从蒸馏模型继续训练，与历史模型及当前模型对弈。"`，也可直接编辑 `models.json` 中对应模型的 `stage` 和 `description`。说明使用显式标注，不根据文件名推断；大小读取 `onnx_bytes`，单位为 MiB（1 MiB = 1,048,576 字节）。

要使用浏览器搜索，请以 v0.4.0 或更新的网站作为 `--site`。给旧站增量添加带价值头的模型不会自动升级其页面和 Worker。

这里的“压缩”包括剥离训练内容和 ZIP 无损压缩，**不做 FP16/INT8 量化**。若源文件本来就只有推理权重，ONNX 文件大小通常接近原文件；ZIP 的缩小比例取决于权重内容。ZIP 必须解压后部署，网页实际加载 `.onnx` 文件。无需修改网页 JS，也无需重新安装 Node.js 或重新下载 WASM 运行库。

### 从头打包整个网站

以下工具只用于开发者打包，不属于分发包的运行依赖：

```bash
uv pip install --python .venv/bin/python onnx onnxruntime
npm ci --prefix web
.venv/bin/python -m gomoku.publish_static \
  --models-dir dist/gomoku-play-v0.1.0/models \
  --output dist/gomoku-static-v0.1.0
```

也可以重复传入 `--checkpoint path/to/model.pt`。导出使用 FP32，保留策略推理图和残差模型的价值头，不打包训练状态或配置。默认保留文件名，将 `.pt` 改为 `.onnx`。已有发布目录或 ZIP 会拒绝覆盖，请指定新的输出路径。

生成独立静态目录与同名 ZIP。增加或替换模型时推荐重新打包，让 `models.json` 的模型列表、大小和 SHA256 与文件对应。网站访问者可下载模型权重。

第三方推理库的 MIT 许可证位于 `vendor/LICENSE`。ONNX Runtime Web 的部署要求参见 <https://onnxruntime.ai/docs/tutorials/web/deploy.html>。

## 可选：进一步压缩权重（有损）

默认导出保留 FP32。`scripts/compress_onnx_weights.py` 可以把大权重张量用 FP16 保存，在图中用 Cast 恢复为 FP32 后计算。归一化参数、偏置和小常量保留 FP32，输入输出也仍为 FP32。它不要求浏览器使用 GPU，也不保证减少推理内存或加速计算。

```bash
.venv/bin/python scripts/compress_onnx_weights.py \
  dist/gomoku-static-v0.2.0/models/原模型.onnx \
  --output dist/compact-models/轻量模型.onnx
```

脚本默认比较 128 个局面（随机合法局面和原策略采样局面各半），输出 `.validation.json`，记录概率误差和改变的贪心落点。已有输出拒绝覆盖。压缩会舍入权重，有限局面对照不等于棋力完全不变；请查看报告，并用目标浏览器验证后再采用。

压缩文件的大小和 SHA256 会变化，部署时须同步更新 `models.json` 的 `file`、`onnx_bytes`、`sha256`，保留原 FP32 文件作为基线。此次 `gomoku-static-v0.2.1-fp16` 是单独的轻量试验包，未替换原 v0.2.0。

更激进的试验可以添加 `--storage int8`，按输出通道对大权重量化为对称 INT8，图中用 DequantizeLinear 恢复为 FP32；它只减小权重存储，不是全 INT8 推理，也不承诺加速。默认仍为 FP16。

更充分的贪心对照使用 `scripts/evaluate_compression.py`：

```bash
.venv/bin/python scripts/evaluate_compression.py \
  --original 原始.onnx --fp16 半精度权重.onnx --int8 八位权重.onnx \
  --output dist/新的评测目录 --positions 1024 --games 16
```

评测保存去重后的局面、原版输出、每个局面的 CSV 误差、完整贪心棋谱和 JSON 汇总。同局面落点分歧与整局棋谱分歧分开统计；原策略概率损失不等于胜率损失。原生 CPU 与浏览器 WASM 在临界局面可能有不同的舍入结果，网页发布还须通过 `scripts/check_compression_browser.cjs` 复核。此次完整结果保存在 `dist/compression-evaluation/`。
