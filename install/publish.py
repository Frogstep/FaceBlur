"""Publish a new FaceBlur version as a GitHub release, for update.bat on other PCs.

  publish.bat                     build, ask for a note, upload
  publish.bat --notes "..."       with the note given
  publish.bat --dry-run           build the zip and version.json only, upload nothing

The version is today's date (2026.11.20), with .2, .3, ... for more releases the same
day. It is written into faceblur/_version.py inside the zip only. The release goes to the
public repo named in install/update.py (RELEASES_REPO), which holds only the release zips:
this source repo stays private. Needs the GitHub CLI (gh), signed in.

The shipped files must be committed first, so every release matches a commit; the commit
gets a local tag v<version>. The package locks must be made from the current
install/requirements.in (make_dist.py --lock), or other PCs would not get new packages.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dist  # noqa: E402
from update import RELEASES_REPO, parse_version  # noqa: E402

ROOT = make_dist.ROOT
SHIPPED = ["faceblur", "install"] + make_dist.SCRIPTS + ["README.md", "LICENSE"]


def run(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=check)


def gh() -> str:
    exe = shutil.which("gh")
    if not exe:
        sys.exit("The GitHub CLI is needed: https://cli.github.com/, then 'gh auth login'.")
    if subprocess.run([exe, "auth", "status"], capture_output=True).returncode:
        sys.exit("The GitHub CLI is not signed in: run 'gh auth login'.")
    return exe


def check_committed() -> None:
    dirty = run("git", "status", "--porcelain", "--untracked-files=all", "--", *SHIPPED).stdout.strip()
    if dirty:
        sys.exit("Commit these first, so the release matches a commit:\n" + dirty)


def ensure_repo(exe: str) -> None:
    if subprocess.run([exe, "repo", "view", RELEASES_REPO], capture_output=True).returncode == 0:
        return
    print(f"The releases repo github.com/{RELEASES_REPO} does not exist.")
    if input("Create it now as a PUBLIC repo (it will hold only the release zips)? [y/N] ").strip().lower() != "y":
        sys.exit("Not published.")
    subprocess.run([exe, "repo", "create", RELEASES_REPO, "--public", "--add-readme",
                    "--description", "FaceBlur releases (installed by update.bat)"], check=True)


def next_version(exe: str | None) -> str:
    today = datetime.date.today()
    base = f"{today.year}.{today.month}.{today.day}"
    tags = set(run("git", "tag", "--list", f"v{base}*").stdout.split())
    if exe:
        out = subprocess.run([exe, "release", "list", "--repo", RELEASES_REPO, "--limit", "100",
                              "--json", "tagName", "--jq", ".[].tagName"],
                             capture_output=True, text=True).stdout
        tags |= set(out.split())
    day = (today.year, today.month, today.day)
    taken = [v for v in (parse_version(t[1:]) for t in tags if t.startswith("v")) if v[:3] == day]
    if not taken:
        return base
    n = max((v[3] if len(v) > 3 else 1) for v in taken)
    return f"{base}.{n + 1}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--notes", help="what changed (asked for when missing)")
    ap.add_argument("--dry-run", action="store_true", help="build only, upload nothing")
    a = ap.parse_args()

    if not make_dist.locks_current():
        msg = ("install/requirements.in changed since the package locks were made, so other PCs would "
               "not get the new packages.\nRun make_dist.bat --lock, commit install/requirements-*.txt and "
               "install/locks.sha256, then publish again.")
        if not a.dry_run:
            sys.exit(msg)
        print(f"Warning: {msg}\n")
    exe = None if a.dry_run else gh()
    if not a.dry_run:
        check_committed()
        ensure_repo(exe)
    version = next_version(exe)
    print(f"Version {version}")

    zip_path = make_dist.build(version)
    notes = a.notes if a.notes is not None else ("" if a.dry_run else input("What changed (one line, may be empty)? ").strip())
    info = {
        "version": version,
        "zip": zip_path.name,
        "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "size": zip_path.stat().st_size,
        "notes": notes,
        "commit": run("git", "rev-parse", "--short", "HEAD").stdout.strip(),
        "date": datetime.date.today().isoformat(),
    }
    version_json = zip_path.with_name("version.json")
    version_json.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"{version_json.relative_to(ROOT)} written")
    if a.dry_run:
        print("Dry run: nothing uploaded.")
        return 0

    if input(f"Publish FaceBlur {version} to github.com/{RELEASES_REPO}? [y/N] ").strip().lower() != "y":
        print("Not published.")
        return 1
    subprocess.run([exe, "release", "create", f"v{version}", str(zip_path), str(version_json),
                    "--repo", RELEASES_REPO, "--title", f"FaceBlur {version}",
                    "--notes", notes or f"FaceBlur {version}", "--latest"], check=True)
    run("git", "tag", f"v{version}", check=False)
    print(f"\nPublished FaceBlur {version}. On the other PC, run update.bat (or ./update.sh).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
