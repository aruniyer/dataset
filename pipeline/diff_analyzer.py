"""Diff parsing and comment-to-change mapping.

Parses unified diffs into structured hunks and extracts the relevant
before/after code context for each review comment.
"""

import re
from dataclasses import dataclass, field


@dataclass
class HunkLine:
    """A single line in a diff hunk."""
    type: str  # "add", "remove", "context"
    content: str
    old_lineno: int | None = None
    new_lineno: int | None = None


@dataclass
class Hunk:
    """A single hunk from a unified diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    lines: list[HunkLine] = field(default_factory=list)


@dataclass
class FileDiff:
    """Diff for a single file."""
    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_rename: bool = False


@dataclass
class CommentContext:
    """All context needed to generate a test for a review comment."""
    file_path: str
    before_code: str          # full file at head_commit
    after_code: str           # full file at merged_commit
    diff_hunk: str            # from the review comment
    before_patch_lines: str   # relevant lines from patch_to_review
    after_patch_lines: str    # relevant lines from merged_patch
    comment_text: str


def parse_unified_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff string into structured FileDiff objects.

    Handles standard git unified diff format with --- / +++ headers
    and @@ hunk markers.
    """
    if not diff_text:
        return []

    file_diffs = []
    current_file = None
    current_hunk = None

    lines = diff_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # New file diff header
        if line.startswith("diff --git"):
            # Parse a/path b/path from the diff line
            match = re.match(r"diff --git a/(.*) b/(.*)", line)
            if match:
                current_file = FileDiff(
                    old_path=match.group(1),
                    new_path=match.group(2),
                )
                file_diffs.append(current_file)
                current_hunk = None
            i += 1
            continue

        # Detect new/deleted files
        if current_file and line.startswith("new file mode"):
            current_file.is_new = True
            i += 1
            continue
        if current_file and line.startswith("deleted file mode"):
            current_file.is_deleted = True
            i += 1
            continue
        if current_file and line.startswith("rename from"):
            current_file.is_rename = True
            i += 1
            continue

        # --- / +++ lines (update paths if needed)
        if line.startswith("--- "):
            if current_file and line.startswith("--- a/"):
                current_file.old_path = line[6:]
            i += 1
            continue
        if line.startswith("+++ "):
            if current_file and line.startswith("+++ b/"):
                current_file.new_path = line[6:]
            i += 1
            continue

        # Hunk header
        hunk_match = re.match(
            r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line
        )
        if hunk_match and current_file:
            current_hunk = Hunk(
                old_start=int(hunk_match.group(1)),
                old_count=int(hunk_match.group(2) or "1"),
                new_start=int(hunk_match.group(3)),
                new_count=int(hunk_match.group(4) or "1"),
                header=line,
            )
            current_file.hunks.append(current_hunk)
            i += 1
            continue

        # Hunk content lines
        if current_hunk is not None:
            if line.startswith("+"):
                old_lineno = None
                new_lineno = (
                    current_hunk.new_start
                    + sum(
                        1
                        for l in current_hunk.lines
                        if l.type in ("add", "context")
                    )
                )
                current_hunk.lines.append(
                    HunkLine("add", line[1:], old_lineno, new_lineno)
                )
            elif line.startswith("-"):
                old_lineno = (
                    current_hunk.old_start
                    + sum(
                        1
                        for l in current_hunk.lines
                        if l.type in ("remove", "context")
                    )
                )
                current_hunk.lines.append(
                    HunkLine("remove", line[1:], old_lineno, None)
                )
            elif line.startswith(" "):
                old_lineno = (
                    current_hunk.old_start
                    + sum(
                        1
                        for l in current_hunk.lines
                        if l.type in ("remove", "context")
                    )
                )
                new_lineno = (
                    current_hunk.new_start
                    + sum(
                        1
                        for l in current_hunk.lines
                        if l.type in ("add", "context")
                    )
                )
                current_hunk.lines.append(
                    HunkLine("context", line[1:], old_lineno, new_lineno)
                )
            elif line.startswith("\\"):
                # "\ No newline at end of file" — skip
                pass
            else:
                # Not part of hunk content (e.g. blank line between diffs)
                current_hunk = None

        i += 1

    return file_diffs


def _extract_file_hunks(patch_text: str, filepath: str) -> list[Hunk]:
    """Extract hunks for a specific file from a patch."""
    file_diffs = parse_unified_diff(patch_text)
    for fd in file_diffs:
        if fd.new_path == filepath or fd.old_path == filepath:
            return fd.hunks
    return []


def _hunk_lines_as_text(hunks: list[Hunk], side: str = "new") -> str:
    """Render hunk lines as text for a given side ('old' or 'new').

    For 'old': include remove + context lines.
    For 'new': include add + context lines.
    """
    result = []
    for hunk in hunks:
        result.append(hunk.header)
        for hl in hunk.lines:
            if side == "old" and hl.type in ("remove", "context"):
                prefix = "-" if hl.type == "remove" else " "
                result.append(f"{prefix}{hl.content}")
            elif side == "new" and hl.type in ("add", "context"):
                prefix = "+" if hl.type == "add" else " "
                result.append(f"{prefix}{hl.content}")
    return "\n".join(result)


def extract_comment_context(
    comment: dict,
    patch_to_review: str,
    merged_patch: str,
    get_file_fn,
    head_commit: str,
    merged_commit: str,
) -> CommentContext:
    """Extract full context for a review comment.

    Args:
        comment: A review comment dict with keys: path, line, text, diff_hunk, etc.
        patch_to_review: The full patch of the commit under review.
        merged_patch: The full merged patch.
        get_file_fn: Callable(commit, filepath) -> str that retrieves file contents.
        head_commit: The commit hash of the code under review.
        merged_commit: The commit hash after addressing the review.

    Returns:
        CommentContext with all necessary information for test generation.
    """
    filepath = comment["path"]

    # Get full file contents at both commits
    before_code = get_file_fn(head_commit, filepath)
    after_code = get_file_fn(merged_commit, filepath)

    # Extract the relevant hunks from both patches for this file
    before_hunks = _extract_file_hunks(patch_to_review, filepath)
    after_hunks = _extract_file_hunks(merged_patch, filepath)

    before_patch_lines = _hunk_lines_as_text(before_hunks, side="new")
    after_patch_lines = _hunk_lines_as_text(after_hunks, side="new")

    return CommentContext(
        file_path=filepath,
        before_code=before_code,
        after_code=after_code,
        diff_hunk=comment.get("diff_hunk", ""),
        before_patch_lines=before_patch_lines,
        after_patch_lines=after_patch_lines,
        comment_text=comment.get("text", ""),
    )


def compute_per_comment_diff(
    patch_to_review: str, merged_patch: str
) -> dict[str, list[Hunk]]:
    """Compute the difference between two patches to identify what changed.

    This helps identify what specific changes were made in response to
    review comments by comparing the patch under review with the final
    merged patch.

    Returns:
        Dict mapping filepath to list of hunks that differ between the two patches.
    """
    before_diffs = {fd.new_path: fd for fd in parse_unified_diff(patch_to_review)}
    after_diffs = {fd.new_path: fd for fd in parse_unified_diff(merged_patch)}

    all_files = set(before_diffs.keys()) | set(after_diffs.keys())
    result = {}

    for filepath in all_files:
        before_fd = before_diffs.get(filepath)
        after_fd = after_diffs.get(filepath)

        if before_fd is None and after_fd is not None:
            # File only in merged patch (new changes for this file)
            result[filepath] = after_fd.hunks
        elif before_fd is not None and after_fd is None:
            # File removed in merged patch
            result[filepath] = before_fd.hunks
        elif before_fd is not None and after_fd is not None:
            # Both have changes for this file — check if they differ
            before_text = _hunk_lines_as_text(before_fd.hunks, "new")
            after_text = _hunk_lines_as_text(after_fd.hunks, "new")
            if before_text != after_text:
                result[filepath] = after_fd.hunks

    return result
