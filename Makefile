.PHONY: \
	release-assets \
	source-locks \
	verify-all \
	verify-benchmark \
	verify-claims \
	verify-content \
	verify-fresh-evidence \
	verify-historical-evidence \
	verify-json \
	verify-release-assets \
	verify-release-tools \
	verify-static

RELEASE_TAG ?= gdn-sm90a-r0
RELEASE_DIR ?= build/releases/$(RELEASE_TAG)

verify-static:
	ruff check scripts reproduce
	ruff format --check scripts reproduce
	python3 -m compileall -q scripts reproduce
	bash -n reproduce/*.sh

verify-json:
	find . -type f -name '*.json' -not -path './.git/*' -print0 | \
		xargs -0 -n1 python3 -m json.tool >/dev/null

verify-historical-evidence:
	python3 scripts/verify_public_evidence.py \
		--bundle evidence/historical/gdn-sm90a-h20-20260728-v1
	python3 scripts/render_performance_markdown.py --check

verify-claims:
	python3 scripts/validate_claims.py

verify-benchmark:
	python3 -m unittest discover -s reproduce/benchmark/tests -v

verify-fresh-evidence:
	python3 -m reproduce.fresh_evidence.verify \
		--bundle evidence/fresh/gdn-sm90a-public-tags-h20-20260729-v1
	python3 -m unittest discover -s reproduce/fresh_evidence/tests -v

verify-release-tools:
	python3 -m unittest discover -s scripts/tests -v

verify-content:
	python3 scripts/materialize_publication_content.py --check
	python3 scripts/validate_claims.py --check-content dist/content
	python3 scripts/render_figures.py --check
	python3 scripts/check_markdown_links.py

verify-all: \
	verify-static \
	verify-json \
	verify-historical-evidence \
	verify-claims \
	verify-benchmark \
	verify-fresh-evidence \
	verify-release-tools \
	verify-content

release-assets:
	python3 scripts/build_release_assets.py \
		--tag "$(RELEASE_TAG)" \
		--require-contract-tag \
		--output "$(RELEASE_DIR)"

verify-release-assets:
	python3 scripts/verify_release_assets.py \
		--tag "$(RELEASE_TAG)" \
		--require-contract-tag \
		--assets "$(RELEASE_DIR)"

source-locks:
	@test -n "$(TVM_DIR)" || { echo "TVM_DIR is required" >&2; exit 2; }
	@test -n "$(TIRX_DIR)" || { echo "TIRX_DIR is required" >&2; exit 2; }
	@test -n "$(CUTEDSL_DIR)" || { echo "CUTEDSL_DIR is required" >&2; exit 2; }
	@test -n "$(FLA_DIR)" || { echo "FLA_DIR is required" >&2; exit 2; }
	python3 reproduce/verify_source_locks.py \
		--source-lock config/public-source-lock.json \
		--tvm-dir "$(TVM_DIR)" \
		--tirx-dir "$(TIRX_DIR)" \
		--cutedsl-dir "$(CUTEDSL_DIR)" \
		--fla-dir "$(FLA_DIR)"
