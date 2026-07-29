#!/usr/bin/env python3
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

"""Verify exact public Git source locks and their published lock document."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepoLock:
    key: str
    repository_slug: str
    commit: str
    tree: str
    tag: str | None = None
    tag_object: str | None = None
    runtime_commit: str | None = None
    runtime_tree: str | None = None


LOCKS = (
    RepoLock(
        key="tvm",
        repository_slug="Aharrypotter/tvm",
        tag="gdn-sm90a-compiler-r0",
        tag_object="18e2172e54aefcba7e11f3e62fa8bfa137b480d4",
        commit="acb1312de80b39340e09b0aaad818ff029e745d6",
        tree="d7989ef4a8621448755da21bd1dec8b5d6c18a1b",
    ),
    RepoLock(
        key="tirx",
        repository_slug="Aharrypotter/tirx-kernels",
        tag="gdn-sm90a-kernel-r0",
        tag_object="f233dcbfc314415b9af496e3fd855554d81d662c",
        commit="12ce3721f7c62c5fbd911103ae373de689e58385",
        tree="cc04daa65ff52014348c8e078721e9afb017467a",
        runtime_commit="90c9c62c84ecc452dd86602f0ea49a625845045c",
        runtime_tree="0453ca537eb846871064338c519a983329618dff",
    ),
    RepoLock(
        key="cutedsl",
        repository_slug="Aharrypotter/cuLA",
        tag="gdn-sm90a-comparator-r1",
        tag_object="0e2c50a4f39b58811e234466682a62f8926998c4",
        commit="88737e9d906cf313995a092624656a89d74dd65e",
        tree="aa01d1a169b72dedd582f86fc9b257e9e2776344",
    ),
    RepoLock(
        key="fla",
        repository_slug="fla-org/flash-linear-attention",
        commit="d1ce07369d581813553f30a750af3b6b5f9af6a9",
        tree="e5ea97e3041c3e4dd0bf6974c2259f7ed104ddc2",
    ),
)

CONFIG_KEY = {
    "tvm": "tvm",
    "tirx": "tirx_kernels",
    "cutedsl": "cutedsl_comparator",
    "fla": "fla",
}


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(repo), *args),
        check=check,
        capture_output=True,
        text=True,
    )


def normalize_github_slug(url: str) -> str | None:
    value = url.strip()
    prefixes = (
        "https://github.com/",
        "http://github.com/",
        "ssh://git@github.com/",
        "git@github.com:",
    )
    for prefix in prefixes:
        if value.startswith(prefix):
            slug = value[len(prefix) :].removesuffix(".git").strip("/")
            return slug
    return None


def configured_values(source_lock: dict[str, Any], lock: RepoLock) -> list[str]:
    errors = []
    repository = source_lock["repositories"][CONFIG_KEY[lock.key]]
    if lock.key == "fla":
        checks = {
            "commit": lock.commit,
            "tree": lock.tree,
        }
        section = repository
    else:
        checks = {
            "tag": lock.tag,
            "tag_object": lock.tag_object,
            "commit": lock.commit,
            "tree": lock.tree,
        }
        section = repository["public_release"]
    for field, expected in checks.items():
        if section.get(field) != expected:
            errors.append(f"{lock.key}: source lock has wrong {field}")
    if lock.runtime_commit is not None:
        runtime = repository["exact_runtime_delta"]
        if runtime.get("commit") != lock.runtime_commit:
            errors.append(f"{lock.key}: source lock has wrong runtime commit")
        if runtime.get("tree") != lock.runtime_tree:
            errors.append(f"{lock.key}: source lock has wrong runtime tree")
    return errors


def verify_repo(repo: Path, lock: RepoLock) -> tuple[list[str], dict[str, Any]]:
    errors = []
    observed: dict[str, Any] = {}
    if not repo.is_dir():
        return [f"{lock.key}: repository directory is missing"], observed
    inside = git(repo, "rev-parse", "--is-inside-work-tree", check=False)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return [f"{lock.key}: path is not a Git worktree"], observed

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    tree = git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()
    observed.update(commit=head, tree=tree)
    if head != lock.commit:
        errors.append(f"{lock.key}: HEAD is {head}, expected {lock.commit}")
    if tree != lock.tree:
        errors.append(f"{lock.key}: tree is {tree}, expected {lock.tree}")

    dirty = git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines()
    if dirty:
        errors.append(f"{lock.key}: worktree is not clean ({len(dirty)} status entries)")

    remote_slugs = set()
    for remote in git(repo, "remote").stdout.splitlines():
        url = git(repo, "remote", "get-url", remote).stdout.strip()
        if slug := normalize_github_slug(url):
            remote_slugs.add(slug)
    if lock.repository_slug not in remote_slugs:
        errors.append(f"{lock.key}: no remote resolves to {lock.repository_slug}")

    if lock.tag is not None:
        ref = f"refs/tags/{lock.tag}"
        tag_result = git(repo, "rev-parse", "--verify", ref, check=False)
        if tag_result.returncode != 0:
            errors.append(f"{lock.key}: tag {lock.tag} is missing")
        else:
            tag_object = tag_result.stdout.strip()
            peeled = git(repo, "rev-parse", f"{ref}^{{}}").stdout.strip()
            observed.update(tag=lock.tag, tag_object=tag_object, peeled_commit=peeled)
            object_type = git(repo, "cat-file", "-t", ref).stdout.strip()
            if object_type != "tag":
                errors.append(f"{lock.key}: {lock.tag} is not an annotated tag")
            if tag_object != lock.tag_object:
                errors.append(f"{lock.key}: tag object is {tag_object}, expected {lock.tag_object}")
            if peeled != lock.commit:
                errors.append(f"{lock.key}: tag peels to {peeled}, expected {lock.commit}")

    if lock.runtime_commit is not None:
        runtime_tree = git(repo, "rev-parse", f"{lock.runtime_commit}^{{tree}}").stdout.strip()
        ancestry = git(
            repo,
            "merge-base",
            "--is-ancestor",
            lock.runtime_commit,
            lock.commit,
            check=False,
        )
        observed.update(runtime_commit=lock.runtime_commit, runtime_tree=runtime_tree)
        if ancestry.returncode != 0:
            errors.append(f"{lock.key}: runtime commit is not an ancestor of the release")
        if runtime_tree != lock.runtime_tree:
            errors.append(
                f"{lock.key}: runtime tree is {runtime_tree}, expected {lock.runtime_tree}"
            )
    return errors, observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--tvm-dir", required=True, type=Path)
    parser.add_argument("--tirx-dir", required=True, type=Path)
    parser.add_argument("--cutedsl-dir", required=True, type=Path)
    parser.add_argument("--fla-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_lock_path = args.source_lock.expanduser().resolve()
    try:
        source_lock = json.loads(source_lock_path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise SystemExit(f"cannot read source lock: {err}") from err
    if source_lock.get("schema") != "gdn-sm90a.public-source-lock.v1":
        raise SystemExit("unexpected public source-lock schema")

    repositories = {
        "tvm": args.tvm_dir.expanduser().resolve(),
        "tirx": args.tirx_dir.expanduser().resolve(),
        "cutedsl": args.cutedsl_dir.expanduser().resolve(),
        "fla": args.fla_dir.expanduser().resolve(),
    }
    errors: list[str] = []
    observed = {}
    for lock in LOCKS:
        try:
            errors.extend(configured_values(source_lock, lock))
            repo_errors, repo_observed = verify_repo(repositories[lock.key], lock)
            errors.extend(repo_errors)
            observed[lock.key] = repo_observed
        except (KeyError, subprocess.CalledProcessError) as err:
            errors.append(f"{lock.key}: verification command or lock field failed: {err}")

    result = {
        "schema": "gdn-sm90a.public-source-verification.v1",
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "observed": observed,
        "working_trees_clean": not any("not clean" in error for error in errors),
        "remote_writes": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
