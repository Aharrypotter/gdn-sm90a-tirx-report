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

import copy
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


def _canonical_body(value: str) -> bytes:
    return (value.rstrip("\n") + "\n").encode()


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
    def __init__(
        self,
        *,
        tag_name: str = "gdn-sm90a-local-test",
        expected_public_tag: str | None = None,
        include_supersession: bool = False,
        stale_publication_state: bool = False,
    ) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gdn-release-fixture-")
        self.root = Path(self.temporary.name)
        self.tag = tag_name
        self.expected_public_tag = expected_public_tag or tag_name
        self.include_supersession = include_supersession
        self.stale_publication_state = stale_publication_state
        self.r1_body = "# Fixture superseded r1 release\n"
        self.r1_asset_payloads = {
            "fixture-r1-SHA256SUMS": b"fixture-r1-checksums\n",
            "fixture-r1-content.zip": b"fixture-r1-content\n",
            "fixture-r1-evidence.tar": b"fixture-r1-evidence\n",
            "fixture-r1-manifest.json": b'{"fixture":"r1"}\n',
            "fixture-r1-source.tar": b"fixture-r1-source\n",
        }
        self._create()

    def close(self) -> None:
        self.temporary.cleanup()

    def _create(self) -> None:
        _run(self.root, "git", "init", "-q")
        _run(self.root, "git", "config", "user.name", "Release Fixture")
        _run(self.root, "git", "config", "user.email", "fixture@example.invalid")
        (self.root / "LICENSE").write_text("fixture license\n")
        (self.root / "dist/content").mkdir(parents=True)
        publication_content = "# Fixture content\n"
        if self.stale_publication_state:
            publication_content += "A fresh public-tag rerun is pending, not complete.\n"
        (self.root / "dist/content/post.md").write_text(publication_content)
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
        if self.include_supersession:
            (self.root / "releases/gdn-sm90a-r1.md").write_text(self.r1_body)
            _write_json(
                self.root / "releases/gdn-sm90a-r1-supersession.json",
                self._supersession_record(),
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

    def _supersession_record(self) -> dict[str, Any]:
        asset_lock = []
        for index, (name, payload) in enumerate(
            sorted(self.r1_asset_payloads.items()),
            start=1,
        ):
            asset_lock.append(
                {
                    "asset_id": 1000 + index,
                    "browser_download_url": f"https://downloads.invalid/r1/{name}",
                    "content_type": "application/octet-stream",
                    "created_at_utc": "2026-07-28T00:01:00Z",
                    "label": name,
                    "name": name,
                    "sha256": _sha256(payload),
                    "size_bytes": len(payload),
                    "state": "uploaded",
                    "updated_at_utc": "2026-07-28T00:01:01Z",
                    "uploader": "fixture",
                }
            )
        body = _canonical_body(self.r1_body)
        return {
            "schema": "gdn-sm90a.report-release-supersession.v1",
            "status": "SUPERSEDED_FOR_PUBLICATION_GUIDANCE",
            "superseded_release": {
                "repository": "fixture/report",
                "tag": "gdn-sm90a-r1",
                "tag_object": "d" * 40,
                "commit": "e" * 40,
                "tree": "f" * 40,
                "release_id": 10,
                "release_name": "Fixture superseded r1 release",
                "release_url": ("https://github.com/fixture/report/releases/tag/gdn-sm90a-r1"),
                "created_at_utc": "2026-07-28T00:00:00Z",
                "published_at_utc": "2026-07-28T00:02:00Z",
                "updated_at_utc": "2026-07-28T00:02:00Z",
                "retention": "PUBLIC_UNMODIFIED_AUDIT_RECORD",
                "tag_ci_run_id": 20,
                "tag_ci_url": "https://github.com/fixture/report/actions/runs/20",
            },
            "superseding_release": {
                "repository": "fixture/report",
                "tag": self.tag,
                "release_url": f"https://github.com/fixture/report/releases/tag/{self.tag}",
                "effective_condition": "PUBLIC_RELEASE_AND_PUBLIC_AUDIT_PASS",
                "scope": "PUBLICATION_GUIDANCE_ONLY",
            },
            "asset_lock": asset_lock,
            "body_lock": {
                "canonical_sha256": _sha256(body),
                "canonical_size_bytes": len(body),
                "source_path_at_r1": "releases/gdn-sm90a-r1.md",
            },
            "tag_ci_lock": {
                "run_id": 20,
                "url": "https://github.com/fixture/report/actions/runs/20",
                "name": "Verify public report",
                "event": "push",
                "status": "completed",
                "conclusion": "success",
                "head_sha": "e" * 40,
                "head_branch": "gdn-sm90a-r1",
                "created_at_utc": "2026-07-28T00:00:30Z",
                "updated_at_utc": "2026-07-28T00:01:30Z",
                "run_started_at_utc": "2026-07-28T00:00:30Z",
                "workflow_id": 30,
            },
            "mutation_policy": {
                "edit_r1_release_body": False,
                "move_or_delete_r1_tag": False,
                "replace_or_delete_r1_assets": False,
            },
            "validity": {
                "evidence_integrity": "VALID_UNCHANGED",
                "performance_facts": "VALID_UNCHANGED",
                "publication_guidance": "SUPERSEDED",
                "runtime_source": "VALID_UNCHANGED",
                "source_provenance": "VALID_UNCHANGED",
            },
        }

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
        self.api_json_calls: list[tuple[str, bool]] = []
        self.api_list_calls: list[tuple[str, bool]] = []
        self.api_bytes_calls: list[tuple[str, bool]] = []
        self.web_calls: list[tuple[str, bool]] = []

    def api_json(self, path: str, *, authenticated: bool) -> dict[str, Any]:
        self.api_json_calls.append((path, authenticated))
        if path not in self.json_values:
            raise auditor.PublicReleaseAuditError(f"missing fake GitHub object: {path}")
        value = copy.deepcopy(self.json_values[path])
        if not isinstance(value, dict):
            raise AssertionError(f"expected object fixture for {path}")
        return value

    def api_list(self, path: str, *, authenticated: bool) -> list[Any]:
        self.api_list_calls.append((path, authenticated))
        if path not in self.json_values:
            raise auditor.PublicReleaseAuditError(f"missing fake GitHub list: {path}")
        value = copy.deepcopy(self.json_values[path])
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
        del accept
        self.api_bytes_calls.append((path_or_url, authenticated))
        if path_or_url not in self.byte_values:
            raise auditor.PublicReleaseAuditError(f"missing fake GitHub bytes: {path_or_url}")
        return self.byte_values[path_or_url]

    def web_bytes(self, url: str, *, authenticated: bool = False) -> bytes:
        self.web_calls.append((url, authenticated))
        if url not in self.byte_values:
            raise auditor.PublicReleaseAuditError(f"missing fake GitHub page: {url}")
        return self.byte_values[url]


class ReleaseAssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = FixtureRepository()
        cls.assets_a = cls.fixture.build("assets-a")
        cls.assets_b = cls.fixture.build("assets-b")
        cls.r2_fixture = FixtureRepository(
            tag_name="gdn-sm90a-r2",
            include_supersession=True,
        )
        cls.r2_assets = cls.r2_fixture.build("r2-assets")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.r2_fixture.close()
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
        self.assertEqual(result["publication_state_scan"]["status"], "PASS")

    def test_stale_publication_state_is_rejected_from_content_asset(self) -> None:
        fixture = FixtureRepository(stale_publication_state=True)
        try:
            with self.assertRaisesRegex(
                builder.ReleaseAssetError,
                "obsolete pre-rerun guidance",
            ):
                fixture.build("stale-assets")
            self.assertFalse((fixture.root / "stale-assets").exists())
        finally:
            fixture.close()

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

    def _fake_client(
        self,
        *,
        mode: str,
        fixture: FixtureRepository | None = None,
        assets: Path | None = None,
    ) -> FakeGitHubClient:
        fixture = fixture or self.fixture
        assets = assets or self.assets_a
        identity = builder.load_tag_identity(fixture.root, fixture.tag)
        contract = json.loads((fixture.root / "contracts/release-assets.json").read_text())
        notes = (fixture.root / "releases/notes.md").read_text()
        remote_assets = []
        byte_values: dict[str, bytes] = {}
        for index, path in enumerate(sorted(assets.iterdir()), start=1):
            browser_url = f"https://downloads.invalid/{path.name}"
            api_url = f"https://api.invalid/assets/{index}"
            payload = path.read_bytes()
            remote_assets.append(
                {
                    "id": index,
                    "name": path.name,
                    "size": len(payload),
                    "state": "uploaded",
                    "content_type": "application/octet-stream",
                    "label": path.name,
                    "created_at": "2026-07-29T00:00:00Z",
                    "updated_at": "2026-07-29T00:00:01Z",
                    "digest": f"sha256:{_sha256(payload)}",
                    "browser_download_url": browser_url,
                    "url": api_url,
                    "uploader": {"login": "fixture"},
                }
            )
            byte_values[browser_url] = payload
            byte_values[api_url] = payload
        report_release = {
            "id": 1,
            "tag_name": fixture.tag,
            "draft": mode == "draft",
            "prerelease": False,
            "created_at": "2026-07-29T00:00:00Z",
            "updated_at": "2026-07-29T00:00:00Z",
            "published_at": "2026-07-29T00:00:00Z",
            "name": contract["release"]["title"],
            "body": notes,
            "html_url": f"https://github.com/fixture/report/releases/tag/{fixture.tag}",
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
            f"repos/fixture/report/git/ref/tags/{fixture.tag}": {
                "object": {"type": "tag", "sha": identity.tag_object}
            },
            f"repos/fixture/report/git/tags/{identity.tag_object}": {
                "object": {"type": "commit", "sha": identity.commit}
            },
            f"repos/fixture/report/git/commits/{identity.commit}": {"tree": {"sha": identity.tree}},
            f"repos/fixture/report/releases/tags/{fixture.tag}": report_release,
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
        if fixture.include_supersession:
            supersession = fixture._supersession_record()
            superseded = supersession["superseded_release"]
            tag_ci = supersession["tag_ci_lock"]
            r1_assets = []
            for entry in supersession["asset_lock"]:
                name = entry["name"]
                payload = fixture.r1_asset_payloads[name]
                r1_assets.append(
                    {
                        "id": entry["asset_id"],
                        "name": name,
                        "size": entry["size_bytes"],
                        "state": entry["state"],
                        "content_type": entry["content_type"],
                        "label": entry["label"],
                        "created_at": entry["created_at_utc"],
                        "updated_at": entry["updated_at_utc"],
                        "digest": f"sha256:{entry['sha256']}",
                        "browser_download_url": entry["browser_download_url"],
                        "uploader": {"login": entry["uploader"]},
                    }
                )
                byte_values[entry["browser_download_url"]] = payload
            json_values.update(
                {
                    "repos/fixture/report/git/ref/tags/gdn-sm90a-r1": {
                        "object": {
                            "type": "tag",
                            "sha": superseded["tag_object"],
                        }
                    },
                    f"repos/fixture/report/git/tags/{superseded['tag_object']}": {
                        "object": {
                            "type": "commit",
                            "sha": superseded["commit"],
                        }
                    },
                    f"repos/fixture/report/git/commits/{superseded['commit']}": {
                        "tree": {"sha": superseded["tree"]}
                    },
                    "repos/fixture/report/releases/tags/gdn-sm90a-r1": {
                        "id": superseded["release_id"],
                        "tag_name": superseded["tag"],
                        "draft": False,
                        "prerelease": False,
                        "created_at": superseded["created_at_utc"],
                        "updated_at": superseded["updated_at_utc"],
                        "published_at": superseded["published_at_utc"],
                        "name": superseded["release_name"],
                        "body": fixture.r1_body,
                        "html_url": superseded["release_url"],
                        "assets": r1_assets,
                    },
                    f"repos/fixture/report/actions/runs/{tag_ci['run_id']}": {
                        "id": tag_ci["run_id"],
                        "name": tag_ci["name"],
                        "event": tag_ci["event"],
                        "status": tag_ci["status"],
                        "conclusion": tag_ci["conclusion"],
                        "head_sha": tag_ci["head_sha"],
                        "head_branch": tag_ci["head_branch"],
                        "html_url": tag_ci["url"],
                        "created_at": tag_ci["created_at_utc"],
                        "updated_at": tag_ci["updated_at_utc"],
                        "run_started_at": tag_ci["run_started_at_utc"],
                        "workflow_id": tag_ci["workflow_id"],
                    },
                }
            )
            byte_values.update(
                {
                    superseded["release_url"]: b"superseded r1 release",
                    tag_ci["url"]: b"superseded r1 tag CI",
                }
            )
        if mode == "public":
            json_values["repos/fixture/report/releases/latest"] = report_release
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
                f"https://github.com/fixture/report/releases/tag/{fixture.tag}": (
                    b"report release"
                ),
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
        self.assertEqual(
            public["latest_release"],
            {"tag": self.fixture.tag, "release_id": 1, "status": "PASS"},
        )
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

    def _r2_client(self, *, mode: str) -> FakeGitHubClient:
        return self._fake_client(
            mode=mode,
            fixture=self.r2_fixture,
            assets=self.r2_assets,
        )

    def _r1_record(self) -> dict[str, Any]:
        return json.loads(
            (self.r2_fixture.root / "releases/gdn-sm90a-r1-supersession.json").read_text()
        )

    def test_r2_draft_runs_superseded_r1_lock_anonymously(self) -> None:
        client = self._r2_client(mode="draft")
        result = auditor.audit_public_release(
            root=self.r2_fixture.root,
            tag=self.r2_fixture.tag,
            contract_path="contracts/release-assets.json",
            assets=self.r2_assets,
            mode="draft",
            client=client,
            rebuild_count=1,
            run_evidence_verifiers=False,
        )
        lock = result["superseded_release_lock"]
        self.assertEqual(lock["status"], "PASS")
        self.assertEqual(lock["body"]["status"], "PASS")
        self.assertEqual(lock["tag_ci"]["conclusion"], "success")
        self.assertTrue(all(asset["anonymous_download"] for asset in lock["downloaded_assets"]))

        record = self._r1_record()
        superseded = record["superseded_release"]
        tag_ci = record["tag_ci_lock"]
        r1_api_markers = (
            superseded["tag"],
            superseded["tag_object"],
            superseded["commit"],
            f"actions/runs/{tag_ci['run_id']}",
        )
        r1_api_calls = [
            (path, authenticated)
            for path, authenticated in client.api_json_calls
            if any(marker in path for marker in r1_api_markers)
        ]
        self.assertTrue(r1_api_calls)
        self.assertTrue(all(not authenticated for _, authenticated in r1_api_calls))

        r1_web_urls = {
            superseded["release_url"],
            tag_ci["url"],
            *{entry["browser_download_url"] for entry in record["asset_lock"]},
        }
        r1_web_calls = [
            (url, authenticated) for url, authenticated in client.web_calls if url in r1_web_urls
        ]
        self.assertEqual({url for url, _ in r1_web_calls}, r1_web_urls)
        self.assertTrue(all(not authenticated for _, authenticated in r1_web_calls))

    def test_r2_missing_supersession_record_fails_closed(self) -> None:
        fixture = FixtureRepository(tag_name="gdn-sm90a-r2")
        try:
            assets = fixture.build("r2-missing-record-assets")
            for mode in ("draft", "public"):
                with self.subTest(mode=mode):
                    client = self._fake_client(
                        mode=mode,
                        fixture=fixture,
                        assets=assets,
                    )
                    with self.assertRaisesRegex(
                        auditor.PublicReleaseAuditError,
                        r"requires the r1 supersession record",
                    ):
                        auditor.audit_public_release(
                            root=fixture.root,
                            tag=fixture.tag,
                            contract_path="contracts/release-assets.json",
                            assets=assets,
                            mode=mode,
                            client=client,
                            rebuild_count=1,
                            run_evidence_verifiers=False,
                        )
        finally:
            fixture.close()

    def test_r2_draft_rejects_superseded_asset_reupload(self) -> None:
        client = self._r2_client(mode="draft")
        release_path = "repos/fixture/report/releases/tags/gdn-sm90a-r1"
        client.json_values[release_path]["assets"][0]["id"] += 1
        with self.assertRaisesRegex(
            auditor.PublicReleaseAuditError,
            r"locked asset ID drift",
        ):
            auditor.audit_public_release(
                root=self.r2_fixture.root,
                tag=self.r2_fixture.tag,
                contract_path="contracts/release-assets.json",
                assets=self.r2_assets,
                mode="draft",
                client=client,
                rebuild_count=1,
                run_evidence_verifiers=False,
            )

    def test_superseded_body_digest_rejects_coordinated_drift(self) -> None:
        client = self._r2_client(mode="draft")
        release_path = "repos/fixture/report/releases/tags/gdn-sm90a-r1"
        original = self.r2_fixture.r1_body
        drifted = original.replace("Fixture", "Fixturz", 1)
        self.assertEqual(len(drifted), len(original))
        client.json_values[release_path]["body"] = drifted
        with self.assertRaisesRegex(
            auditor.PublicReleaseAuditError,
            r"live release body drift",
        ):
            auditor._audit_superseded_release_lock(
                client,
                record=self._r1_record(),
                expected_body=drifted,
            )

    def test_superseded_tag_ci_is_fail_closed(self) -> None:
        cases = (
            ("status", "in_progress"),
            ("conclusion", "failure"),
            ("head_sha", "a" * 40),
            ("event", "workflow_dispatch"),
        )
        for field, value in cases:
            with self.subTest(field=field):
                client = self._r2_client(mode="draft")
                run_path = "repos/fixture/report/actions/runs/20"
                client.json_values[run_path][field] = value
                with self.assertRaisesRegex(
                    auditor.PublicReleaseAuditError,
                    rf"tag CI {field} drift",
                ):
                    auditor._audit_superseded_release_lock(
                        client,
                        record=self._r1_record(),
                        expected_body=self.r2_fixture.r1_body,
                    )

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
