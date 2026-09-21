"""Inspect built wheel and sdist archives before publication."""

from __future__ import annotations

import stat
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from release_audit import (
    ALLOWED_SUFFIX_PATHS,
    FORBIDDEN_PARTS,
    FORBIDDEN_SUFFIXES,
    MAX_FILE_BYTES,
    ROOT,
    content_findings,
)

SOURCE_PACKAGE = ROOT / "src" / "imessage_mlx"
SOURCE_MEMBERS = {
    path.relative_to(SOURCE_PACKAGE).as_posix(): path.read_bytes()
    for path in SOURCE_PACKAGE.rglob("*")
    if path.is_file() and path.suffix in {".py", ".yaml"}
}


def unsafe_member_name(name: str) -> str | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return "unsafe archive member path"
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & FORBIDDEN_PARTS:
        return "forbidden archive path"
    allowed_suffix = any(
        len(path.parts) >= len(allowed.parts) and path.parts[-len(allowed.parts) :] == allowed.parts
        for allowed in ALLOWED_SUFFIX_PATHS
    )
    standard_metadata = path.name in {"entry_points.txt", "top_level.txt"} and any(
        part.endswith(".dist-info") for part in path.parts
    )
    if path.suffix.lower() in FORBIDDEN_SUFFIXES and not (allowed_suffix or standard_metadata):
        return "forbidden archive artifact type"
    return None


def inspect_content(label: str, data: bytes, findings: list[str]) -> None:
    findings.extend(content_findings(PurePosixPath(label), data))


def audit_wheel(path: Path) -> list[str]:
    findings: list[str] = []
    seen_sources: set[str] = set()
    saw_license = False
    saw_notice = False
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                reason = unsafe_member_name(info.filename)
                if reason:
                    findings.append(f"{reason}: {path.name}:{info.filename}")
                    continue
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    findings.append(f"symbolic link in archive: {path.name}:{info.filename}")
                    continue
                if info.file_size > MAX_FILE_BYTES:
                    findings.append(f"unexpected large member: {path.name}:{info.filename}")
                    continue
                data = archive.read(info)
                inspect_content(f"{path.name}:{info.filename}", data, findings)
                if info.filename.endswith(".dist-info/licenses/LICENSE"):
                    saw_license = True
                if info.filename.endswith(".dist-info/licenses/NOTICE.md"):
                    saw_notice = True
                prefix = "imessage_mlx/"
                if info.filename.startswith(prefix):
                    source_name = info.filename[len(prefix) :]
                    if source_name not in SOURCE_MEMBERS:
                        findings.append(f"unexpected wheel package member: {info.filename}")
                    else:
                        seen_sources.add(source_name)
                        if data != SOURCE_MEMBERS[source_name]:
                            findings.append(f"wheel source mismatch: {info.filename}")
    except (OSError, zipfile.BadZipFile) as error:
        return [f"could not inspect wheel {path.name}: {error}"]
    missing = sorted(set(SOURCE_MEMBERS) - seen_sources)
    findings.extend(f"wheel missing source member: imessage_mlx/{name}" for name in missing)
    if not saw_license:
        findings.append("wheel is missing LICENSE")
    if not saw_notice:
        findings.append("wheel is missing NOTICE.md")
    return findings


def audit_sdist(path: Path) -> list[str]:
    findings: list[str] = []
    seen_sources: set[str] = set()
    saw_license = False
    saw_notice = False
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                reason = unsafe_member_name(member.name)
                if reason:
                    findings.append(f"{reason}: {path.name}:{member.name}")
                    continue
                if member.issym() or member.islnk() or member.isdev():
                    findings.append(f"unsafe special member: {path.name}:{member.name}")
                    continue
                if member.isdir():
                    continue
                if not member.isfile():
                    findings.append(f"unexpected member type: {path.name}:{member.name}")
                    continue
                if member.size > MAX_FILE_BYTES:
                    findings.append(f"unexpected large member: {path.name}:{member.name}")
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    findings.append(f"could not read archive member: {path.name}:{member.name}")
                    continue
                data = extracted.read()
                inspect_content(f"{path.name}:{member.name}", data, findings)
                parts = PurePosixPath(member.name).parts
                if len(parts) == 2 and parts[1] == "LICENSE":
                    saw_license = True
                if len(parts) == 2 and parts[1] == "NOTICE.md":
                    saw_notice = True
                marker = ("src", "imessage_mlx")
                for index in range(len(parts) - 1):
                    if tuple(parts[index : index + 2]) == marker:
                        source_name = PurePosixPath(*parts[index + 2 :]).as_posix()
                        if source_name in SOURCE_MEMBERS:
                            seen_sources.add(source_name)
                            if data != SOURCE_MEMBERS[source_name]:
                                findings.append(f"sdist source mismatch: {member.name}")
                        break
    except (OSError, tarfile.TarError) as error:
        return [f"could not inspect sdist {path.name}: {error}"]
    missing = sorted(set(SOURCE_MEMBERS) - seen_sources)
    findings.extend(f"sdist missing source member: {name}" for name in missing)
    if not saw_license:
        findings.append("sdist is missing LICENSE")
    if not saw_notice:
        findings.append("sdist is missing NOTICE.md")
    return findings


def main(arguments: list[str]) -> int:
    archives = [Path(argument) for argument in arguments]
    if not archives:
        print("usage: audit_archives.py <wheel-or-sdist> [...]")
        return 2
    findings: list[str] = []
    for archive in archives:
        if archive.suffix == ".whl":
            findings.extend(audit_wheel(archive))
        elif archive.name.endswith(".tar.gz"):
            findings.extend(audit_sdist(archive))
        else:
            findings.append(f"unsupported archive type: {archive}")
    if findings:
        print("archive audit failed:")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        return 1
    print(f"archive audit passed: {len(archives)} archives checked")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
