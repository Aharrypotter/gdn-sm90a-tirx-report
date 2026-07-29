# 让 TIRx 跑上 Hopper：从有限 SM90a 编译器能力到产品化 GDN Prefill

这是一个面向 Hopper SM90a 的非官方 TIRx 个人 fork 实验。

这个项目最值得讲的，并不是“某个 kernel 跑出了一个不错的数字”，而是从编译器能力到公开算子的完整闭环：在编译器层补齐 Hopper 所需机制，把递归语义与任何具体调度解耦并冻结下来，实现多条经过资格限定的 TIRx 调度路径，通过公开 API 做无隐藏 fallback 的精确 dispatch，最后把源码与正确性、安全性、codegen 和可复核性能证据绑定。

它也是一个边界刻意收紧的结果。编译器 fork 提供该算子所需的 SM90a WGMMA/TMA 相关能力；这不是完整 SM90 支持声明。 当前算子契约是 BF16、head dimension 128；测量只覆盖一套 NVIDIA H20 环境中的冻结矩阵。它不代表所有 Hopper GPU、所有 GDN 形状，也不是端到端模型吞吐或延迟结论。

## 为什么必须先做编译器

在 DSL 里写 Hopper kernel，不是把 target 改成 `sm_90a` 就结束了。编译器必须把一组彼此关联的约束一直带到生成的 CUDA：

- 面向架构专用路径的 fail-closed target 判定；
- 本算子使用的 shared/shared 与 register/shared WGMMA lowering；
- 明确的 warpgroup accumulator 和寄存器 layout；
- 支持 K-major 形式的 swizzle-aware shared-memory descriptor；
- 面向显式 stride 全局视图的 TMA TensorMap；
- ragged tail 所需的有界 global/shared copy；
- host codegen 中正确对齐的 TensorMap 存储。

公开编译器标签只实现这个算子需要的能力切片。它并不覆盖任意 WGMMA 形状、所有 TMA 模式、Blackwell TCGEN05 或全部 SM90 工作负载。具体边界见[编译器能力说明](../../docs/compiler-capability.md)。

## 调度优化之前，先冻结语义

GDN 是递归算子。一个实现可能输出“看起来差不多”的 tensor，却悄悄改变 state 的方向、把 gate 放到矩阵的错误一侧、移动精度舍入点，或者让 packed tail 穿过序列边界。

因此，公开语义契约独立于 GPU 实现存在。State 是 FP32、采用 V-first 方向；Q/K/V 是 BF16；MHA、GQA、GVA 的 head 映射均被明确规定。每条序列按 64-token chunk 处理，精度可见的 chunk algebra 固定了：

1. alpha prefix；
2. QK 和 KK 的 transfer scaling；
3. 求下三角逆之前的 row-side beta；
4. FP16 可见的 inverse ladder；
5. inverse 发布为 BF16 operand 时的 column-side beta；
6. prior-state output、state projection、corrected value、chunk 内 output 与 terminal-state update 的次序。

Ragged tail 中对应位置使用恒等值或零，并且任何 replay 都不能跨越 `cu_seqlens`。字面递归和精度契约见 [GDN 语义说明](../../docs/gdn-semantics.md)。

## 一个公开 API，三条产品调度

TIRx fork 发布了产品化 GDN prefill 算子，包含精确优化路径和有文档说明的 pipeline 路径。

通用路径是一条 two-stage pipeline：先并行准备 chunk，再只对真正递归的部分做 scan/output。任何合法但不在优化白名单中的 specialization，都回到这条安全的产品路径。

另外两条路径只按精确 metadata 选择：

- 面向一个限定 packed 形状的 fused short register-replay；
- 最多 replay 一个前驱 chunk、使用 co-resident value warpgroups 和 consumer-relative barrier phase 的 tail-predecessor。

优化 dispatch 是精确白名单。形状近似但不完全一致、state 模式不同、或没有显式 gate，都会回到 pipeline。Wrapper 只读取 host-visible tensor metadata 来选路径，不读取 CUDA 侧的序列边界值，也不会转调 FLA、Triton、CuTeDSL 或 C++ GDN fallback。

完整 route map 见[调度与产品 dispatch](../../docs/schedules-and-dispatch.md)。

## 验证不是一个绿色勾，而是一组分层门禁

发布证据把下面这些层次分开：

- CPU 语义与 auxiliary reference；
- 公开 GPU 语义及精确 route/near-miss 行为；
- 重复执行与 non-default-stream liveness；
- packed redzone 与输入不变性；
- dispatcher-visible host synchronization；
- Compute Sanitizer 的 memcheck、racecheck 和 synccheck；
- 与源码绑定的 WGMMA/TMA codegen 和资源清单；
- 通过最终公开 callable 采集的隔离计时回执。

这种分层很重要。源码阅读不是 GPU 正确性；输出正确不是内存安全；sanitizer 干净不等于生成了原生 WGMMA；内部 PrimFunc 很快，也不等于公开 dispatch 很快。

历史 release 因为完整 device-artifact baseline 不具备 timing inheritance 资格，最终使用了 fresh full canonical timing matrix。验证摘要见[验证说明](../../docs/validation.md)。

## 历史性能证据到底说明了什么

历史基准覆盖一套 NVIDIA H20 环境上的 6 个精确 BF16/D128 测试行。

比值方向：tirx_latency / comparator_latency; lower is faster。

在 6 个冻结测试行中，TIRx 有 5 行的延迟低于 CuTeDSL。

packed-n10 的精确 TIRx/CuTeDSL 比值为 1.0146，即延迟高 1.46%，仍处于预先登记的 2.0% 噪声区间内；该行并非性能胜出。

在 6 个冻结测试行中，TIRx 有 6 行的延迟低于 FLA。

上面这些句子都由 canonical performance JSON 生成。包含逐行延迟和几何平均值的完整表格见[自动生成的历史性能报告](../../reports/historical-performance.md)。计时区域是冻结 benchmark contract 下的最终公开 callable，不是端到端模型 benchmark。

## Comparator 身份必须说清楚

CuTeDSL 对照实现绑定到 88737e9d906cf313995a092624656a89d74dd65e 上的 gdn-sm90a-comparator-r1，入口为 cula.gdn.prefill.chunk_gated_delta_rule；本报告明确排除 gdn2-sm90a-comparator-r0。

这个修正不能省略。历史回执明确记录了 GDN callable、backend、commit 和 CuTe DSL 版本；较早的 GDN2 发布标签指向不同源码，不能用于解释本次 GDN 测量。FLA 对照也绑定到一个精确 upstream commit 和 callable。

完整映射见[证据溯源](../../docs/evidence-provenance.md)与机器可读的 [link map](../../contracts/link-map.json)。

## 历史与 fresh 两类证据

历史证据包状态为 HISTORICAL_EVIDENCE_BOUND；unofficial-personal-fork 为 true，upstream-merge 为 false。

脱敏 evidence bundle 是确定性、隐私安全的，并绑定到不可变历史 release seal。它保留数值回执和紧凑的正确性、安全性、codegen、release 摘要，同时排除原始日志、私有 host/device 标识、cache 路径、profiler artifact 与进程元数据。

这个 bundle 永远保持历史证据身份，不会被改名或覆盖。Public tags 的运行属于另一份加性证据：

精确 public tags 的 H20 6 行表征：bundle 派生 66 receipts，CHARACTERIZATION；TIRx/CuTeDSL 0.834812、TIRx/FLA 0.453700、packed-n10 基础三进程 1.017445。未复现历史 host-sync/sanitizer/完整 codegen/resource reseal；非官方 fork，无 upstream merge。

[Fresh evidence root](../../evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1)
和[来源派生的性能报告](../../reports/fresh-public-tag-performance.md)把结论绑定到精确 source/build/runtime identity、独立进程、物理 H20、receipt correctness 与 timing。历史 host-sync、sanitizer 和完整 codegen/resource 结果仍只属于历史证据。

## 公开资产

- [报告与证据仓库](https://github.com/Aharrypotter/gdn-sm90a-tirx-report)
- [TVM compiler tag](https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0)
- [TIRx GDN kernel tag](https://github.com/Aharrypotter/tirx-kernels/tree/gdn-sm90a-kernel-r0)
- [修正后的 CuTeDSL GDN comparator tag](https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1)
- [精确 FLA comparator commit](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9)
- [Fresh public-tag evidence](../../evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1)
- [Fresh 性能 characterization](../../reports/fresh-public-tag-performance.md)

这些都是非官方个人 fork 产物。没有任何上游项目合并、背书或发布这项工作。
