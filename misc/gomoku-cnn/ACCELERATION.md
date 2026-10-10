# 浏览器加速实现与实测

日期：2026-10-10。模型保持原来的两个 FP16 存储、FP32 计算模型。

## 已实现

- 跨回合、贪心与搜索共用的有界网络输出缓存。
- 有界置换表，区分 EXACT / LOWER / UPPER，按相同深度和 Top-p/Top-k 条件复用。
- 缓存候选集合，并优先访问过去搜索发现的好着法。
- 达到评分上界 +2 时提前结束，避免继续计算无法提高分数的分支。
- CPU 单线程、多线程（最多 4）及 WebGPU；不可用或计算失败时回退 CPU。
- 静态托管隔离响应头、实际运行后端展示、参数保存恢复及缓存命中统计。

本版使用既有模型，性能变化来自浏览器计算和搜索实现。

## 本机初测

设备：RTX 5070 Ti；Chromium Headless 测试浏览器，20 个逻辑 CPU。为测试硬件 WebGPU，浏览器显式启用了 Vulkan/WebGPU 支持；普通环境若未提供 GPU 适配器，自动模式使用 CPU。以下不是跨设备性能保证。

推理测试使用 residual-history.onnx 的 8 个固定局面，各运行两次；预热计算内核后清空局面缓存。时间包括 Worker 往返，不包括模型下载、加载和首次编译。

| 计算后端 | 推理中位数 | 推理均值 | 同局面 3 层冷搜索 | 1 秒预算完成深度 |
|---|---:|---:|---:|---:|
| CPU 单线程 | 141.28 ms | 141.64 ms | 2724.98 ms | 1 |
| CPU 4 线程 | 41.52 ms | 42.18 ms | 783.43 ms | 3 |
| 硬件 WebGPU | 10.32 ms | 9.77 ms | 216.09 ms | 5 |

搜索使用同一局面、Top-p=90%、Top-k=3，固定深度测试不设时限；预算测试的目标深度为 100。时限仍为软限制，观察到总耗时略超过 1 秒。

这个浅层冷搜索例子中，优化前后都需要 19 次网络评估，搜索排序/置换表没有带来明显冷启动收益。多线程和 GPU 的收益来自更快的网络计算，不能把缓存命中的极小耗时推广到新局面。

## 缓存与跨回合

同局面重复搜索命中已完成层的置换表，额外网络评估降为 0，返回相同分数和落点。

另一个真实后续局面中，保留上一回合缓存时需要 9 次网络评估；清空缓存后需要 13 次。硬件 WebGPU 搜索耗时分别为 98.98 ms 与 128.91 ms，返回相同落点 160 和分数 -0.28225243。

切换模型、计算后端时清空缓存。UI 测试确认改变计算方式不会重复下载相同模型，刷新会恢复所选模式；缓存本身仅存在于当前页面内存中。

## 正确性与回退

- 与独立、不剪枝的 Max/Min 穷举对照，检查双方视角、奇偶深度、终局与 Top-p/Top-k。
- 验证置换表上下界重用、深度隔离、参数隔离、未完成迭代不写成根节点精确结果、容量淘汰。
- 第一个模型的 8 个推理局面上，CPU 4 线程与 CPU 单线程概率一致；WebGPU 的最大概率绝对误差约 1.01e-6，贪心落点一致。
- 第二个模型额外测试 8 个一层搜索局面，CPU 与 WebGPU 选择一致，价值分数最大误差约 4.97e-6。
- 在 GPU 搜索的第二次前向中注入故障，已确认清空 GPU 缓存、回退 CPU 并完成有效搜索。
- 未提供隔离响应头的普通静态服务器上，多线程选项会退到单线程，对弈仍可用。
- 桌面和手机界面、引擎切换、参数恢复、模型切换、悔棋复用已验证。

这些检查证明实现与有限样本的行为，不保证所有设备或临界局面完全同值。搜索顺序变化可能改变完全并列候选的选择；GPU/CPU 浮点差异也可能改变接近的排名。

## 复现与原始数据

```bash
python scripts/serve_static.py --directory dist --port 8005
node tests/test_browser_search.cjs
node tests/test_static_backend.cjs
GOMOKU_GPU_TEST=1 node scripts/benchmark_browser_acceleration.cjs
node scripts/check_accelerated_ui.cjs
```

浏览器脚本需要单独安装 Playwright 及 Chromium。GPU 测试使用显式 Vulkan 标志，仅改变测试浏览器，不修改用户浏览器设置。

原始数据：`acceleration-report.json`、`acceleration-cross-turn.json`、`acceleration-fallback.json`、`acceleration-second-model.json`。

运行后端配置参考：https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html 。
