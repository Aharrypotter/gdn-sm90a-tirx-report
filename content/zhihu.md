# 我给 TIRx 补了一条 SM90a 能力链，并用它实现了产品化 GDN Prefill

<!-- TEMPLATE_ONLY
知乎发布模板。解析所有 {{claim:Cxx:zh}} 后再粘贴到平台。
性能句只能来自 claim registry；表格只链接 generated report。
-->

先给结论的边界：

{{claim:C01:zh}}

{{claim:C04:zh}}

这不是“TVM 已经支持所有 SM90 workload”的结论，也不是一个端到端模型 benchmark。它是一项更具体的工程实验：为了让 TIRx 能表达并稳定编译一个 Hopper GDN prefill 算子，我补齐了一组有限但闭环的编译器能力；然后冻结语义、实现三类调度、建立公开 dispatch，最后用分层证据验证。

## 一、问题为什么不只是“写一个 kernel”

GDN 同时有三个难点。

第一，它是递归算子。当前 state 会参与本 token 的 prediction，gate 会影响衰减与写入，最终 state 还要继续传给下一个 chunk。只要 state orientation、gate 的左右位置或精度舍入点发生偏移，就可能得到“误差不大但不是同一个算法”的实现。

第二，它是 packed-variable-length。`cu_seqlens` 留在 CUDA 上，不能为了 dispatch 随手拷回 host；ragged tail 又必须保证不会读写到相邻序列。

第三，它要用 Hopper 的 WGMMA/TMA。Kernel 源码里的 tile op 只是上层意图，编译器还必须正确处理 warpgroup fragment layout、shared-memory swizzle、TensorMap、barrier、ragged copy 和 host ABI。

所以我把工程拆成了三层：

1. 编译器层只负责可复用的 SM90a lowering 能力；
2. 语义层冻结 GDN recurrence 与精度契约；
3. 调度层在不改变语义的前提下选择 pipeline、register replay 或 tail-predecessor。

架构图和各层职责见[公开架构文档](../docs/architecture.md)。

## 二、编译器到底补了什么

{{claim:C02:zh}}

这次编译器工作包含：

- 对架构专用 WGMMA 路径做 fail-closed target 判定；
- 支持本算子需要的 shared/shared 与 register/shared WGMMA；
- 明确 accumulator 与 A-register 的 warpgroup layout；
- 处理 MN-major、K-major 及 swizzled shared descriptor；
- 让显式 stride 的 global view 正确进入 TMA TensorMap；
- 对 ragged tail 做 logical-prefix bounded copy；
- 在 host codegen 中满足 TensorMap 的存储和对齐要求。

这些能力之间不能随便缺一项。比如，WGMMA instruction 选对了，但 sliced accumulator 的 register offset 错了，结果仍会错；TMA device code 没问题，但 host 侧 TensorMap ABI 不对，甚至到不了 kernel launch；tail mask 正确，但 swizzle offset 没有重新物化，也会落到错误地址。

具体支持范围与非目标见[有限 SM90a 编译器能力](../docs/compiler-capability.md)。

## 三、为什么我先写“语义参考”，再写高性能调度

公开 contract 规定：

- Q/K/V 为 BF16，state 为 FP32；
- state 使用 V-first `[V, K]`；
- 支持 MHA、GQA、GVA 的明确 head mapping；
- sequence 按 64-token chunk 处理；
- ragged tail 的 Q/K/V、alpha、beta 与 inverse diagonal 使用固定的零/恒等规则；
- 任何 replay 都不能跨 `cu_seqlens` 边界。

更重要的是，chunk 内不是只写一个“数学上等价”的公式，而是冻结精度可见顺序：alpha prefix、QK/KK transfer、row-side beta、FP16 inverse ladder、column-side beta，以及 O1、SK、NewV、O2、terminal state update 的次序。

这让 CPU tokenwise reference、chunkwise reference 与 GPU schedule 各自承担清晰职责。CPU reference 可以证明 recurrence 和方向，但不能替代 GPU 正确性、安全性或性能证据。

完整契约见 [GDN 语义文档](../docs/gdn-semantics.md)。

## 四、三类调度如何共存

{{claim:C03:zh}}

通用 pipeline 将可并行的 chunk prepare 与真正递归的 scan/output 分开。它不是“慢速 fallback”，而是所有合法非白名单 specialization 的正式产品路径。

针对冻结 workload 中的精确 key，又有两类限定优化：

- short register-replay：为一个精确 packed key 设计的 fused 路径；
- tail-predecessor：只 replay 一个前驱 chunk，并让 value warpgroups co-resident 的两阶段路径。

这里最重要的不是 schedule 名称，而是 dispatch discipline：

- 只有完整命中 key 且显式 gate 条件满足时才进优化路径；
- near miss、state 模式变化或 gate 缺失都回到 pipeline；
- dispatch 只看 host-visible metadata；
- 不调用 FLA、Triton、CuTeDSL 或 C++ GDN 实现。

详细 route map 在[调度与 dispatch 文档](../docs/schedules-and-dispatch.md)中。

## 五、我如何判断它“真的可以发布”

我没有把一次 benchmark 当作 release gate。验证按层推进：

1. CPU recurrence、inverse 与 auxiliary semantics；
2. 公开 GPU callable 的 MHA/GQA/GVA 与 optional state/gate；
3. 精确 optimized key 和 near-miss route；
4. 重复运行与 non-default stream；
5. packed redzone、相邻 sequence boundary 与输入不变性；
6. dispatcher-visible host sync；
7. Compute Sanitizer；
8. 所有冻结 specialization 的 WGMMA/TMA codegen 与资源清单；
9. 最终 public callable 的隔离 timing receipts。

历史 release 的完整 device artifact baseline 不满足 timing inheritance 条件，因此没有直接沿用旧性能，而是重新跑了 full canonical timing matrix。这一点和 sanitizer 的 superseded control attempt 都被保留在证据里。

验证分层与边界见[验证说明](../docs/validation.md)。

## 六、性能应该怎样读

{{claim:C05:zh}}

{{claim:C08:zh}}

{{claim:C09:zh}}

{{claim:C10:zh}}

这里必须保留 packed-n10 的例外，不能把结论写成“所有行都胜过 CuTeDSL”。完整逐行延迟、比值与几何平均表由 canonical JSON 自动生成，见[历史性能报告](../reports/historical-performance.md)。计时对象是冻结 benchmark contract 下的最终公开 callable，不是端到端模型。

## 七、为什么 comparator 还要专门做一次溯源修正

{{claim:C13:zh}}

历史回执实际调用的是 `cula.gdn`，而不是 `cula.gdn2`。因此，公开 comparator 必须直接绑定到历史回执记录的 commit、entrypoint、backend 与 CuTe DSL 版本。较早的 GDN2 标签保留不动，但明确不属于本报告证据。

这是一个典型的 evidence lesson：名字看起来接近，不等于 source identity 相同。Benchmark 的对照身份必须从 receipt 回溯到 callable 与 commit，不能凭目录名或开发历史猜测。

## 八、为什么现在仍然标记为 historical

{{claim:C14:zh}}

当前公开 bundle 来自一个不可变的历史 release seal。脱敏过程按字段 allowlist 重建，不复制原始日志再做正则替换；公开包保留数值样本与紧凑验证摘要，但不暴露 host、container、GPU UUID、SSH alias、文件路径、PID 或 profiler artifact。

公开 compiler/kernel/comparator tags 是在历史 seal 之后发布的。因此，它们与历史 source 做了明确映射，但还不能称为“从 public tags 独立复现”。下一步必须从这些 tags 构建，在 H20 上重跑同一冻结矩阵和验证门禁，并发布一个新的、独立封存的 fresh evidence root。

## 九、代码与证据

- [报告与证据仓库](https://github.com/Aharrypotter/gdn-sm90a-tirx-report)
- [TVM compiler tag](https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0)
- [TIRx GDN kernel tag](https://github.com/Aharrypotter/tirx-kernels/tree/gdn-sm90a-kernel-r0)
- [修正后的 CuTeDSL comparator tag](https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1)
- [FLA comparator commit](https://github.com/fla-org/flash-linear-attention/commit/d1ce07369d581813553f30a750af3b6b5f9af6a9)

这些都是非官方个人 fork 产物，没有 upstream merge 或 endorsement。对我来说，这项工作的价值也不在于提前宣称上游状态，而在于把编译器能力、kernel 产品路径与可审计证据真正闭环。
