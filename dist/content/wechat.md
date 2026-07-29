# 从“能生成代码”到“能公开发布”：一次 TIRx SM90a GDN 实验

如果只看最后的 kernel，人们很容易把这项工作理解成一次普通的性能优化：换一套 tile、调一组 warp、跑一张表。

但真正耗时的部分，其实发生在 kernel 之前和之后。

之前，是让编译器具备足够精确的 Hopper 能力；之后，是证明公开 callable 的语义、路径、安全性、codegen 和性能都属于同一份源码。

这是一个面向 Hopper SM90a 的非官方 TIRx 个人 fork 实验。

## 先把边界写在开头

历史基准覆盖一套 NVIDIA H20 环境上的 6 个精确 BF16/D128 测试行。

编译器 fork 提供该算子所需的 SM90a WGMMA/TMA 相关能力；这不是完整 SM90 支持声明。

也就是说，这是一项有明确边界的 operator experiment。它不代表全部 Hopper GPU，不代表任意 dtype 或 head dimension，不代表所有 GDN shape，更不是端到端模型性能。

边界写清楚不是“保守措辞”，而是让结果可以被复核的前提。

## 第一关：让 TIRx 真正理解 SM90a 路径

一个 DSL kernel 想使用 WGMMA 和 TMA，不能只在源码里写出对应 op。编译器必须理解 warpgroup fragment 如何落到寄存器，shared-memory swizzle 如何进入 descriptor，显式 stride 的 global view 如何构造 TensorMap，ragged tail 如何做有界 copy，host wrapper 又如何为 TensorMap 提供正确的 ABI 存储。

这次公开的 compiler fork 补齐了本算子所需的能力链：

- WGMMA shared/shared 与 register/shared lowering；
- accumulator 与 A-register layout；
- MN-major、K-major 与 swizzled shared descriptor；
- TMA explicit-stride TensorMap；
- bounded copy 与 tail masking；
- TensorMap host codegen。

它是一条完整但有限的能力链，而不是“SM90 功能大全”。

## 第二关：递归算子的语义不能交给调度猜

GDN 的 state 会跨 token、跨 chunk 延续。Gate 放在矩阵哪一侧、state 用 `[V, K]` 还是 `[K, V]`、inverse 在哪里舍入、packed tail 用零还是恒等值，都会改变最终语义。

所以我先冻结了独立 reference：

- tokenwise recurrence 负责最直观的算法定义；
- chunkwise reference 负责精度可见的 staged algebra；
- inverse reference 固定下三角 inverse ladder；
- packed reference 保证每条 sequence 独立。

GPU schedule 必须和这些 reference 对齐，而不是反过来修改 reference 来迁就 kernel。

## 第三关：优化路径必须是“产品 dispatch”

TIRx fork 发布了产品化 GDN prefill 算子，包含精确优化路径和有文档说明的 pipeline 路径。

最终 API 下有三类 schedule：

1. 面向所有合法非白名单形状的通用 pipeline；
2. 面向一个精确 packed key 的 short register-replay；
3. 面向若干精确 key 的 tail-predecessor。

Dispatch 不读取 device 侧 sequence boundary，不靠运行时试错，也没有外部 GDN fallback。只有完整命中 allowlist 才能进入 specialized route，所有 near miss 都回到 pipeline。

这让性能优化有了产品边界：我们知道什么被优化、什么没有被优化，也能为两者分别写测试。

## 第四关：一次正确运行远远不够

我把验证拆成了多层：

- CPU 语义；
- 公开 GPU correctness；
- optimized route 与 near-miss；
- stream liveness；
- packed redzone；
- host-sync audit；
- Compute Sanitizer；
- WGMMA/TMA codegen 与 resource inventory；
- 隔离 timing receipts。

任何一层都不能替代下一层。

特别是 performance：历史 release 没有因为“旧 kernel 看起来一样”就继承完整 timing，而是在 codegen inheritance 条件不足时重新跑 full canonical matrix。

## 那性能结果怎么说才准确

比值方向：tirx_latency / comparator_latency; lower is faster。

在 6 个冻结测试行中，TIRx 有 5 行的延迟低于 CuTeDSL。

packed-n10 的精确 TIRx/CuTeDSL 比值为 1.0146，即延迟高 1.46%，仍处于预先登记的 2.0% 噪声区间内；该行并非性能胜出。

在 6 个冻结测试行中，TIRx 有 6 行的延迟低于 FLA。

这里的关键不是找一个最漂亮的数字做标题，而是把 packed-n10 的例外和 noise-band 解释一起公开。完整表格由 canonical evidence 自动生成，见[历史性能报告](../../reports/historical-performance.md)。

## 一次必须公开说明的 comparator 修正

CuTeDSL 对照实现绑定到 88737e9d906cf313995a092624656a89d74dd65e 上的 gdn-sm90a-comparator-r1，入口为 cula.gdn.prefill.chunk_gated_delta_rule；本报告明确排除 gdn2-sm90a-comparator-r0。

历史回执绑定的是 `cula.gdn` 的 callable。较早的 GDN2 标签指向不同实现，因此不能作为本报告 comparator。这个修正也提醒我：性能证据必须一路绑定到 commit、entrypoint、backend 和依赖版本，不能只记录一个模糊的库名。

## 公开标签的 fresh characterization

公开包状态为 HISTORICAL_EVIDENCE_BOUND；unofficial-personal-fork 为 true，upstream-merge 为 false。

当前 historical evidence bundle 仍是从不可变历史 release seal 按字段 allowlist 派生的，不会被改名或覆盖。

另一份加性 evidence root 已从精确 public tags/commit 单独封存：

精确 public tags 的 H20 6 行表征：bundle 派生 66 receipts，CHARACTERIZATION；TIRx/CuTeDSL 0.834812、TIRx/FLA 0.453700、packed-n10 基础三进程 1.017445。未复现历史 host-sync/sanitizer/完整 codegen/resource reseal；非官方 fork，无 upstream merge。

它的[来源派生报告](../../reports/fresh-public-tag-performance.md)只覆盖 source/build/runtime identity、独立进程、物理 H20、receipt correctness 和六行 timing。历史 host-sync、sanitizer 与完整 codegen/resource 证据没有被复制为 fresh 结论。

## 公开链接

- [报告、文档与证据](https://github.com/Aharrypotter/gdn-sm90a-tirx-report)
- [TVM compiler](https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0)
- [TIRx GDN kernel](https://github.com/Aharrypotter/tirx-kernels/tree/gdn-sm90a-kernel-r0)
- [CuTeDSL GDN comparator](https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1)
- [FLA comparator](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9)
- [Fresh public-tag evidence](../../evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1)

这些代码和标签都是非官方个人 fork 产物，没有 upstream merge 或 endorsement。

这项实验最终想证明的不是“DSL 能写出一个 kernel”，而是：当编译器能力、语义契约、调度空间、公开 API 和证据门禁被分层处理后，TIRx 可以承载一条真正可审查、可复现、可继续演进的 Hopper operator 开发路径。
