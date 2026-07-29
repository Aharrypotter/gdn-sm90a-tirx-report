# Copyright 2026 Aharrypotter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

import audit_public_release as auditor  # noqa: E402
import build_release_assets as builder  # noqa: E402
import verify_release_assets as verifier  # noqa: E402


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run(root: Path, *arguments: str) -> str:
    return subprocess.run(
        [*arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _seal(root: Path, payloads: dict[str, Any]) -> None:
    entries = []
    for relative, value in sorted(payloads.items()):
        path = root / relative
        if isinstance(value, bytes):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        else:
            _write_json(path, value)
        raw = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(raw),
                "size_bytes": len(raw),
            }
        )
    manifest = {
        "schema": "fixture.public-evidence-manifest.v1",
        "file_count": len(entries),
        "files": entries,
    }
    manifest_raw = _json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_raw)
    (root / "MANIFEST.sha256").write_text(f"{_sha256(manifest_raw)}  manifest.json\n")


class FixtureRepository:
    def __init__(self, *, expected_public_tag: str = "gdn-sm90a-local-test") -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gdn-release-fixture-")
        self.root = Path(self.temporary.name)
        self.tag = "gdn-sm90a-local-test"
        self.expected_public_tag = expected_public_tag
        self._create()

    def close(self) -> None:
        self.temporary.cleanup()

    def _create(self) -> None:
        _run(self.root, "git", "init", "-q")
        _run(self.root, "git", "config", "user.name", "Release Fixture")
        _run(self.root, "git", "config", "user.email", "fixture@example.invalid")
        (self.root / "LICENSE").write_text("fixture license\n")
        (self.root / "dist/content").mkdir(parents=True)
        (self.root / "dist/content/post.md").write_text("# Fixture content\n")
        (self.root / "reports").mkdir()
        (self.root / "reports/performance.md").write_text("# Fixture report\n")

        historical = self.root / "evidence/historical/historical-v1"
        historical.mkdir(parents=True)
        _seal(
            historical,
            {
                "PUBLICATION.json": {
                    "package_status": "HISTORICAL_EVIDENCE_BOUND",
                    "upstream_merge": False,
                },
                "payload.json": {"status": "PASS"},
            },
        )
        fresh = self.root / "evidence/fresh/fresh-v1"
        fresh.mkdir(parents=True)
        _seal(
            fresh,
            {
                "performance.json": {
                    "status": "PASS",
                    "decision_status": "CHARACTERIZATION",
                    "receipt_count": 66,
                    "primary_geomean": {
                        "tirx_over_cutedsl": 0.8,
                        "tirx_over_fla": 0.4,
                    },
                    "packed_n10_escalation_required": True,
                },
                "publication.json": {
                    "status": "PASS",
                    "evidence_kind": "fresh-public-tag-h20-rerun",
                    "decision_status": "CHARACTERIZATION",
                    "upstream_merge_claim": False,
                    "run_id": "fixture-run",
                    "receipt_count": 66,
                },
                "payload.json": {"status": "PASS"},
            },
        )

        _write_json(
            self.root / "contracts/claim-registry.json",
            {"claims": [{"id": "C12", "enabled": True}]},
        )
        _write_json(
            self.root / "config/public-source-lock.json",
            {
                "repositories": {
                    "tvm": {
                        "upstream_repository": "https://github.com/apache/tvm",
                        "public_release": {
                            "branch": "component-branch",
                            "tag": "component-r0",
                            "tag_object": "a" * 40,
                            "commit": "b" * 40,
                            "tree": "c" * 40,
                        },
                    }
                }
            },
        )
        notes = (
            "# Fixture release\n\n"
            "Source: https://github.com/fixture/component/releases/tag/component-r0\n"
        )
        (self.root / "releases").mkdir()
        (self.root / "releases/notes.md").write_text(notes)
        (self.root / "releases/tag-message.txt").write_text(
            "Fixture release tag\n\nNo upstream publication.\n"
        )

        historical_verifier = self.root / "scripts/verify_public_evidence.py"
        historical_verifier.parent.mkdir(parents=True)
        historical_verifier.write_text(
            'import json\nprint(json.dumps({"status": "PASS", "receipt_count": 66}))\n'
        )
        (self.root / "reproduce/fresh_evidence").mkdir(parents=True)
        (self.root / "reproduce/__init__.py").write_text("")
        (self.root / "reproduce/fresh_evidence/__init__.py").write_text("")
        (self.root / "reproduce/fresh_evidence/verify.py").write_text(
            'print("GDN_FRESH_PUBLIC_EVIDENCE_VERIFY_PASS '
            "run_id=fixture-run receipts=66 "
            'tirx/cutedsl=0.800000 tirx/fla=0.400000")\n'
        )

        contract = {
            "schema": builder.SCHEMA,
            "source_lock_path": "config/public-source-lock.json",
            "claim_registry_path": "contracts/claim-registry.json",
            "release": {
                "repository": "fixture/report",
                "expected_public_tag": self.expected_public_tag,
                "title": "Fixture release",
                "notes_path": "releases/notes.md",
                "tag_message_path": "releases/tag-message.txt",
            },
            "evidence": {
                "historical": {
                    "path": "evidence/historical/historical-v1",
                    "manifest_path": "evidence/historical/historical-v1/manifest.json",
                    "seal_path": "evidence/historical/historical-v1/MANIFEST.sha256",
                    "publication_path": ("evidence/historical/historical-v1/PUBLICATION.json"),
                    "evidence_class": "HISTORICAL_EVIDENCE_BOUND",
                    "required_status": "HISTORICAL_EVIDENCE_BOUND",
                },
                "fresh": {
                    "path": "evidence/fresh/fresh-v1",
                    "manifest_path": "evidence/fresh/fresh-v1/manifest.json",
                    "seal_path": "evidence/fresh/fresh-v1/MANIFEST.sha256",
                    "publication_path": "evidence/fresh/fresh-v1/publication.json",
                    "evidence_class": "fresh-public-tag-h20-characterization",
                    "required_status": "PASS",
                    "evidence_kind": "fresh-public-tag-h20-rerun",
                    "decision_status": "CHARACTERIZATION",
                },
            },
            "packages": {
                "source": {
                    "filename": "fixture-source.tar",
                    "format": "tar",
                    "include": ["**"],
                    "prefix": "fixture-source",
                },
                "evidence": {
                    "filename": "fixture-evidence.tar",
                    "format": "tar",
                    "include": ["evidence", "reproduce", "scripts/verify_public_evidence.py"],
                    "prefix": "fixture-evidence",
                },
                "content": {
                    "filename": "fixture-content.zip",
                    "format": "zip",
                    "include": ["LICENSE", "dist/content", "reports"],
                    "prefix": "fixture-content",
                },
            },
            "release_manifest_filename": "fixture-release-manifest.json",
            "checksum_filename": "SHA256SUMS",
            "required_claims": {"C12": True},
            "source_links": [
                {
                    "id": "tvm",
                    "kind": "annotated_tag_release",
                    "repository": "fixture/component",
                    "url": "https://github.com/fixture/component/tree/component-r0",
                    "release_url": (
                        "https://github.com/fixture/component/releases/tag/component-r0"
                    ),
                }
            ],
        }
        _write_json(self.root / "contracts/release-assets.json", contract)
        _run(self.root, "git", "add", ".")
        _run(self.root, "git", "commit", "-qm", "fixture release")
        _run(
            self.root,
            "git",
            "tag",
            "-a",
            self.tag,
            "-F",
            "releases/tag-message.txt",
        )

    def build(self, name: str) -> Path:
        output = self.root / name
        builder.build_release_assets(
            root=self.root,
            tag=self.tag,
            contract_path="contracts/release-assets.json",
            output=output,
        )
        return output


class FakeGitHubClient:
    def __init__(
        self,
        *,
        json_values: dict[str, Any],
        byte_values: dict[str, bytes],
    ) -> None:
        self.json_values = json_values
        self.byte_values = byte_values
        self.web_calls: list[tuple[str, bool]] = []

    def api_json(self, path: str, *, authenticated: bool) -> dict[str, Any]:
        del authenticated
        value = self.json_values[path]
        if not isinstance(value, dict):
            raise AssertionError(f"expected object fixture for {path}")
        return value

    def api_list(self, path: str, *, authenticated: bool) -> list[Any]:
        del authenticated
        value = self.json_values[path]
        if not isinstance(value, list):
            raise AssertionError(f"expected list fixture for {path}")
        return value

    def api_bytes(
        self,
        path_or_url: str,
        *,
        authenticated: bool,
        accept: str = "application/vnd.github+json",
    ) -> bytes:
        del authenticated, accept
        return self.byte_values[path_or_url]

    def web_bytes(self, url: str, *, authenticated: bool = False) -> bytes:
        self.web_calls.append((url, authenticated))
        return self.byte_values[url]


class ReleaseAssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = FixtureRepository()
        cls.assets_a = cls.fixture.build("assets-a")
        cls.assets_b = cls.fixture.build("assets-b")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.close()

    def test_double_build_is_byte_identical_and_self_verifying(self) -> None:
        names_a = sorted(path.name for path in self.assets_a.iterdir())
        names_b = sorted(path.name for path in self.assets_b.iterdir())
        self.assertEqual(names_a, names_b)
        for name in names_a:
            self.assertEqual(
                (self.assets_a / name).read_bytes(),
                (self.assets_b / name).read_bytes(),
            )
        result = verifier.verify_release_assets(
            root=self.fixture.root,
            tag=self.fixture.tag,
            contract_path="contracts/release-assets.json",
            assets=self.assets_a,
            require_contract_tag=True,
            rebuild_count=2,
            run_evidence_verifiers=True,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["byte_identical_rebuilds"])
        self.assertEqual(
            [entry["kind"] for entry in result["evidence_verifier_results"]],
            ["historical", "fresh"],
        )

    def test_lightweight_tag_is_rejected(self) -> None:
        _run(self.fixture.root, "git", "tag", "fixture-lightweight")
        with self.assertRaisesRegex(builder.ReleaseAssetError, "annotated tag"):
            builder.load_tag_identity(self.fixture.root, "fixture-lightweight")

    def test_tampered_asset_is_rejected(self) -> None:
        tampered = self.fixture.root / "tampered"
        shutil.copytree(self.assets_a, tampered)
        source = tampered / "fixture-source.tar"
        source.write_bytes(source.read_bytes() + b"tamper")
        with self.assertRaisesRegex(
            verifier.ReleaseVerificationError,
            "SHA256SUMS mismatch",
        ):
            verifier.verify_release_assets(
                root=self.fixture.root,
                tag=self.fixture.tag,
                contract_path="contracts/release-assets.json",
                assets=tampered,
                rebuild_count=1,
                run_evidence_verifiers=False,
            )

    def _fake_client(self, *, mode: str) -> FakeGitHubClient:
        identity = builder.load_tag_identity(self.fixture.root, self.fixture.tag)
        contract = json.loads((self.fixture.root / "contracts/release-assets.json").read_text())
        notes = (self.fixture.root / "releases/notes.md").read_text()
        remote_assets = []
        byte_values: dict[str, bytes] = {}
        for index, path in enumerate(sorted(self.assets_a.iterdir()), start=1):
            browser_url = f"https://downloads.invalid/{path.name}"
            api_url = f"https://api.invalid/assets/{index}"
            payload = path.read_bytes()
            remote_assets.append(
                {
                    "name": path.name,
                    "size": len(payload),
                    "digest": f"sha256:{_sha256(payload)}",
                    "browser_download_url": browser_url,
                    "url": api_url,
                }
            )
            byte_values[browser_url] = payload
            byte_values[api_url] = payload
        report_release = {
            "id": 1,
            "tag_name": self.fixture.tag,
            "draft": mode == "draft",
            "prerelease": False,
            "name": contract["release"]["title"],
            "body": notes,
            "html_url": (f"https://github.com/fixture/report/releases/tag/{self.fixture.tag}"),
            "assets": remote_assets,
        }
        encoded_pr_query = urlencode(
            {
                "state": "all",
                "head": "fixture:component-branch",
                "per_page": 100,
            }
        )
        json_values: dict[str, Any] = {
            f"repos/fixture/report/git/ref/tags/{self.fixture.tag}": {
                "object": {"type": "tag", "sha": identity.tag_object}
            },
            f"repos/fixture/report/git/tags/{identity.tag_object}": {
                "object": {"type": "commit", "sha": identity.commit}
            },
            f"repos/fixture/report/git/commits/{identity.commit}": {"tree": {"sha": identity.tree}},
            f"repos/fixture/report/releases/tags/{self.fixture.tag}": report_release,
            "repos/fixture/component/git/ref/tags/component-r0": {
                "object": {"type": "tag", "sha": "a" * 40}
            },
            f"repos/fixture/component/git/tags/{'a' * 40}": {
                "object": {"type": "commit", "sha": "b" * 40}
            },
            f"repos/fixture/component/git/commits/{'b' * 40}": {"tree": {"sha": "c" * 40}},
            "repos/fixture/component/releases/tags/component-r0": {
                "tag_name": "component-r0",
                "draft": False,
                "prerelease": False,
                "body": "Report: https://github.com/fixture/report\n",
            },
            f"repos/apache/tvm/pulls?{encoded_pr_query}": [],
        }
        if mode == "public":
            for keyword in ("gdn", "sm90"):
                search = urlencode(
                    {
                        "q": (f"repo:apache/tvm is:pr author:fixture {keyword}"),
                        "per_page": 100,
                    }
                )
                json_values[f"search/issues?{search}"] = {"total_count": 0, "items": []}
        byte_values.update(
            {
                (
                    "https://github.com/fixture/component/releases/tag/component-r0"
                ): b"source release",
                "https://github.com/fixture/component/tree/component-r0": b"source tag",
                (
                    f"https://github.com/fixture/report/releases/tag/{self.fixture.tag}"
                ): b"report release",
            }
        )
        return FakeGitHubClient(json_values=json_values, byte_values=byte_values)

    def test_public_and_draft_release_audits(self) -> None:
        public = auditor.audit_public_release(
            root=self.fixture.root,
            tag=self.fixture.tag,
            contract_path="contracts/release-assets.json",
            assets=self.assets_a,
            mode="public",
            client=self._fake_client(mode="public"),
            rebuild_count=1,
            run_evidence_verifiers=False,
        )
        self.assertEqual(public["status"], "PASS")
        self.assertTrue(public["anonymous_release_page"])
        self.assertTrue(all(asset["anonymous_download"] for asset in public["downloaded_assets"]))
        self.assertEqual(
            public["upstream_pr_audit"][0]["keyword_checks"],
            [
                {"keyword": "gdn", "match_count": 0},
                {"keyword": "sm90", "match_count": 0},
            ],
        )

        draft = auditor.audit_public_release(
            root=self.fixture.root,
            tag=self.fixture.tag,
            contract_path="contracts/release-assets.json",
            assets=self.assets_a,
            mode="draft",
            client=self._fake_client(mode="draft"),
            rebuild_count=1,
            run_evidence_verifiers=False,
        )
        self.assertEqual(draft["status"], "PASS")
        self.assertFalse(draft["anonymous_release_page"])
        self.assertTrue(
            all(not asset["anonymous_download"] for asset in draft["downloaded_assets"])
        )
        self.assertEqual(draft["upstream_pr_audit"][0]["keyword_checks"], [])

    def test_upstream_branch_pr_is_fail_closed(self) -> None:
        query = urlencode(
            {
                "state": "all",
                "head": "fixture:component-branch",
                "per_page": 100,
            }
        )
        client = FakeGitHubClient(
            json_values={
                f"repos/apache/tvm/pulls?{query}": [
                    {"number": 1, "html_url": "https://example.invalid/pr/1"}
                ]
            },
            byte_values={},
        )
        with self.assertRaisesRegex(
            auditor.PublicReleaseAuditError,
            "upstream has PRs",
        ):
            auditor._audit_upstream_prs(
                client,
                coordinates={
                    "tvm": {
                        "kind": "annotated_tag_release",
                        "repository": "fixture/component",
                        "release_branch": "component-branch",
                        "upstream_repository": "https://github.com/apache/tvm",
                    }
                },
                authenticated=False,
                keyword_search=False,
            )


if __name__ == "__main__":
    unittest.main()
