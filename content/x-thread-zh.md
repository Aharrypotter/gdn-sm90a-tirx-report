# X Thread — 中文

<!-- TEMPLATE_ONLY
发布源模板。所有 {{claim:Cxx:zh}} 从 claim registry 解析。
发布前必须在平台编辑器逐条预览解析后的内容。
-->

## 1

我公开了一项 TIRx on Hopper 实验：有限的 SM90a 编译器能力、产品化 GDN prefill 算子，以及把源码、语义、codegen、安全和性能计时分层封存的 evidence package。

{{claim:C01:zh}}

## 2

编译器工作远不只是选择 `sm_90a`：WGMMA SS/RS lowering、warpgroup layout、swizzled/K-major descriptor、显式 stride TMA TensorMap、ragged-tail bounded copy 与 host TensorMap ABI 必须全部一致。

## 3

{{claim:C02:zh}}

Compiler：
https://github.com/Aharrypotter/tvm/tree/gdn-sm90a-compiler-r0

## 4

递归算子如果先调度、后语义，很容易“数值接近但契约已变”。我先冻结 token recurrence、FP32 V-first state、MHA/GQA/GVA 映射、精度可见 inverse ladder 与 packed boundary，再进入性能优化。

## 5

{{claim:C03:zh}}

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

{{claim:C04:zh}}

{{claim:C05:zh}}

## 10

这是 public-call operator latency，不是端到端模型吞吐；结论不能越过冻结矩阵。

## 11

{{claim:C08:zh}}

## 12

{{claim:C09:zh}}

Packed 例外必须进入主叙事，不能藏在脚注。

## 13

{{claim:C10:zh}}

自动生成的完整表格：
https://github.com/Aharrypotter/gdn-sm90a-tirx-report

## 14

Comparator provenance 有一项必须说明的修正：

{{claim:C13:zh}}

## 15

修正后的 CuTeDSL 源码：
https://github.com/Aharrypotter/cuLA/tree/gdn-sm90a-comparator-r1

{{claim:C14:zh}}

## 16

{{claim:C12:zh}}

## 17

Fresh 来源派生报告：
https://github.com/Aharrypotter/gdn-sm90a-tirx-report/blob/gdn-sm90a-r0/reports/fresh-public-tag-performance.md
