# Evidence classes

`historical/` contains privacy-safe derivations from an earlier immutable
release seal. A historical bundle can preserve a valid decision without
claiming that newly published Git tags were independently rerun.

`fresh/` is reserved for results generated from the public tags in this
repository's source lock. It must not be created until source verification,
environment identity, correctness, safety, codegen, and the complete
66-receipt benchmark matrix all close.

Never merge historical and fresh receipts into one aggregate. Every figure
and article claim must identify its evidence class.
