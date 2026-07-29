# X Thread — 中文

## 1

我公开了一项 TIRx on Hopper 实验：有限的 SM90a 编译器能力、产品化 GDN prefill 算子，以及把源码、语义、codegen、安全和性能计时分层封存的 evidence package。

这是一个面向 Hopper SM90a 的非官方 TIRx 个人 fork 实验。

## 2

编译器工作远不只是选择 `sm_90a`：WGMMA SS/RS lowering、warpgroup layout、swizzled/K-major descriptor、显式 stride TMA TensorMap、ragged-tail bounded copy 与 host TensorMap ABI 必须全部一致。

## 3

编译器 fork 提供该算子所需的 SM90a WGMMA/TMA 相关能力；这不是完整 SM90 支持声明。

Compiler：
https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0

## 4

递归算子如果先调度、后语义，很容易“数值接近但契约已变”。我先冻结 token recurrence、FP32 V-first state、MHA/GQA/GVA 映射、精度可见 inverse ladder 与 packed boundary，再进入性能优化。

## 5

TIRx fork 发布了产品化 GDN prefill 算子，包含精确优化路径和有文档说明的 pipeline 路径。

## 6

Wrapper 只根据 host-visible metadata dispatch。精确白名单进入 specialized route；所有合法 near miss 都走通用 TIRx pipeline。没有路径转调外部 GDN fallback。

## 7

三类产品调度：

- 通用 prepare + recurrent scan/output pipeline；
- 一个精确 fused short register-replay；
- bounded replay 与 co-resident value warpgroups 的精确 tail-predecessor。

## 8

验证被刻意分层：CPU 语义、公开 GPU 行为、白名单与 near-miss、stream liveness、packed redzone、host-sync audit、Compute Sanitizer、与源码绑定的 WGMMA/TMA codegen/resources，最后才是隔离 public-call timing。

## 9

历史基准覆盖一套 NVIDIA H20 环境上的 6 个精确 BF16/D128 测试行。

比值方向：tirx_latency / comparator_latency; lower is faster。

## 10

这是 public-call operator latency，不是端到端模型吞吐；结论不能越过冻结矩阵。

## 11

在 6 个冻结测试行中，TIRx 有 5 行的延迟低于 CuTeDSL。

## 12

packed-n10 的精确 TIRx/CuTeDSL 比值为 1.0146，即延迟高 1.46%，仍处于预先登记的 2.0% 噪声区间内；该行并非性能胜出。

Packed 例外必须进入主叙事，不能藏在脚注。

## 13

在 6 个冻结测试行中，TIRx 有 6 行的延迟低于 FLA。

自动生成的完整表格：
https://github.com/Aharrypotter/gdn-sm90a-tirx-report

## 14

Comparator provenance 有一项必须说明的修正：

CuTeDSL 对照实现绑定到 88737e9d906cf313995a092624656a89d74dd65e 上的 gdn-sm90a-comparator-r1，入口为 cula.gdn.prefill.chunk_gated_delta_rule；本报告明确排除 gdn2-sm90a-comparator-r0。

## 15

修正后的 CuTeDSL 源码：
https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1

历史证据包状态为 HISTORICAL_EVIDENCE_BOUND；unofficial-personal-fork 为 true，upstream-merge 为 false。

## 16

精确 public tags 的 H20 6 行表征：bundle 派生 66 receipts，CHARACTERIZATION；TIRx/CuTeDSL 0.834812、TIRx/FLA 0.453700、packed-n10 基础三进程 1.017445。未复现历史 host-sync/sanitizer/完整 codegen/resource reseal；非官方 fork，无 upstream merge。

## 17

Fresh 来源派生报告：
https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r2/reports/fresh-public-tag-performance.md
