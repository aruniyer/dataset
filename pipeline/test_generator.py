"""LLM-powered test generation.

Generates executable tests that fail on the "before" code (pre-review)
and pass on the "after" code (post-review) for each review comment.
Supports multiple languages — Python, JavaScript, TypeScript, Go, Java,
and others — by adapting prompts, validation, and test file extensions.
"""

import ast
import logging
import os
import re
from dataclasses import dataclass

from .diff_analyzer import CommentContext
from .llm_client import DEFAULT_MODEL, LLMError, LLMUsage, chat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language detection & configuration
# ---------------------------------------------------------------------------

# Map file extensions to language identifiers
_EXT_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python", ".pyi": "python", ".pyx": "python", ".pxd": "python",
    # JavaScript
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript", ".tsx": "typescript",
    # Svelte (tested with JS tooling)
    ".svelte": "javascript",
    # Go
    ".go": "go",
    # Java
    ".java": "java",
    # Ruby
    ".rb": "ruby",
    # Rust
    ".rs": "rust",
    # C / C++
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    # Scala
    ".scala": "scala",
}


@dataclass(frozen=True)
class LanguageConfig:
    """Per-language settings for test generation and execution."""
    name: str
    test_file_ext: str            # e.g. ".py", ".test.js"
    test_framework: str           # human-readable framework name
    run_command: str              # shell command to execute a test file
    code_block_tag: str           # markdown fence tag (```python, ```js, …)


LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        name="Python",
        test_file_ext=".py",
        test_framework="pytest",
        run_command="python -m pytest",
        code_block_tag="python",
    ),
    "javascript": LanguageConfig(
        name="JavaScript",
        test_file_ext=".test.js",
        test_framework="Jest",
        run_command="npx jest",
        code_block_tag="javascript",
    ),
    "typescript": LanguageConfig(
        name="TypeScript",
        test_file_ext=".test.ts",
        test_framework="Jest",
        run_command="npx jest",
        code_block_tag="typescript",
    ),
    "go": LanguageConfig(
        name="Go",
        test_file_ext="_test.go",
        test_framework="go test",
        run_command="go test",
        code_block_tag="go",
    ),
    "java": LanguageConfig(
        name="Java",
        test_file_ext=".java",
        test_framework="JUnit 5",
        run_command="mvn test or gradle test",
        code_block_tag="java",
    ),
    "ruby": LanguageConfig(
        name="Ruby",
        test_file_ext="_test.rb",
        test_framework="Minitest",
        run_command="ruby -Ilib:test",
        code_block_tag="ruby",
    ),
    "rust": LanguageConfig(
        name="Rust",
        test_file_ext=".rs",
        test_framework="cargo test",
        run_command="cargo test",
        code_block_tag="rust",
    ),
    "c": LanguageConfig(
        name="C",
        test_file_ext=".c",
        test_framework="assert.h",
        run_command="gcc -o test_bin && ./test_bin",
        code_block_tag="c",
    ),
    "cpp": LanguageConfig(
        name="C++",
        test_file_ext=".cpp",
        test_framework="assert / Catch2",
        run_command="g++ -o test_bin && ./test_bin",
        code_block_tag="cpp",
    ),
    "scala": LanguageConfig(
        name="Scala",
        test_file_ext=".scala",
        test_framework="ScalaTest",
        run_command="sbt test",
        code_block_tag="scala",
    ),
}

DEFAULT_LANGUAGE = "python"


def detect_language(file_path: str) -> str:
    """Detect the programming language from a file path.

    Returns a language key (e.g. 'python', 'javascript') or None for
    non-source files (config, docs, etc.).
    """
    _, ext = os.path.splitext(file_path)
    return _EXT_TO_LANGUAGE.get(ext.lower())


def get_language_config(language: str) -> LanguageConfig:
    """Return the LanguageConfig for a language key, defaulting to Python."""
    return LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS[DEFAULT_LANGUAGE])


def get_test_file_ext(language: str) -> str:
    """Return the test file extension for a language."""
    return get_language_config(language).test_file_ext


# ---------------------------------------------------------------------------
# Comment type classification (language-agnostic)
# ---------------------------------------------------------------------------

_FUNCTIONAL_KEYWORDS = [
    "bug", "error", "fix", "wrong", "incorrect", "broken", "crash",
    "fail", "issue", "handle", "check", "validate", "return", "raise",
    "exception", "missing", "add", "should", "must", "need",
    "behavior", "behaviour", "logic", "condition", "edge case",
]
_STRUCTURAL_KEYWORDS = [
    "copy", "redundant", "unnecessary", "remove", "simplify",
    "refactor", "duplicate", "unused", "dead code", "clean up",
    "don't need", "no need", "not needed", "extra", "overhead",
    "performance", "efficient", "optimize", "instead of",
]
_STYLE_KEYWORDS = [
    "naming", "name", "rename", "convention", "style", "format",
    "pep8", "pep 8", "camelcase", "snake_case", "consistent",
    "import", "ordering", "sort", "alphabetical",
]
_DOC_KEYWORDS = [
    "docstring", "doc", "comment", "documentation", "describe",
    "explain", "type hint", "annotation", "typing", "readme",
]


def classify_comment(comment_text: str, diff_hunk: str = "") -> str:
    """Classify a review comment into a category.

    Categories:
        - 'functional': behavior changes, bug fixes, logic changes
        - 'structural': code structure changes (remove redundancy, simplify)
        - 'style': naming, formatting, conventions
        - 'documentation': docstrings, comments, type hints

    Uses keyword heuristics for fast classification.
    """
    text = (comment_text + " " + diff_hunk).lower()

    scores = {
        "documentation": sum(1 for kw in _DOC_KEYWORDS if kw in text),
        "style": sum(1 for kw in _STYLE_KEYWORDS if kw in text),
        "structural": sum(1 for kw in _STRUCTURAL_KEYWORDS if kw in text),
        "functional": sum(1 for kw in _FUNCTIONAL_KEYWORDS if kw in text),
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "functional"
    return best


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_prompt(
    context: CommentContext, repo: str, comment_type: str, language: str,
) -> str:
    """Build a language-aware test generation prompt."""
    cfg = get_language_config(language)

    # --- type-specific instruction fragment (language-neutral phrasing) ---
    type_instructions = {
        "functional": (
            "Generate a test that verifies the behavior change requested "
            "by this review comment. The test should FAIL on the 'before' "
            "code and PASS on the 'after' code. Import/require the module "
            "under test and check runtime behavior (call functions, check "
            "return values, etc.)."
        ),
        "structural": (
            "Generate a test that verifies the structural change requested "
            "by this review comment. STRONGLY PREFER a functional test that "
            "imports/requires the code and checks behavior — e.g. call the "
            "function and verify it produces the correct result, or check "
            "that redundant/unnecessary operations are no longer observable. "
            "Only fall back to source inspection (reading the file as a "
            "string) if there is truly no observable behavioral difference. "
            "The test should FAIL on the 'before' code and PASS on the "
            "'after' code."
        ),
        "style": (
            "Generate a test that verifies the style change requested by "
            "this review comment. STRONGLY PREFER a functional test that "
            "imports/requires the module and checks attributes, function "
            "names, or the public API surface via runtime introspection. "
            "Only fall back to reading the source file as a string if "
            "there is truly no way to observe the change at runtime. "
            "The test should FAIL on the 'before' code and PASS on the "
            "'after' code."
        ),
        "documentation": (
            "Generate a test that verifies the documentation change "
            "requested by this review comment. STRONGLY PREFER a "
            "functional test that inspects doc attributes or signatures "
            "at runtime. Only fall back to reading the source file as a "
            "string if the change cannot be observed via runtime "
            "introspection. The test should FAIL on the 'before' code "
            "and PASS on the 'after' code."
        ),
    }
    instruction = type_instructions.get(comment_type, type_instructions["functional"])

    # --- language-specific requirements block ---
    if language == "python":
        module_path = context.file_path.replace("/", ".").replace(".py", "")
        file_info = f"`{context.file_path}` (importable as `{module_path}`)"
        requirements = """\
1. Write a single, self-contained pytest test file.
2. The test MUST fail when run against the "before" version of the code and MUST pass when run against the "after" version.
3. Use clear assertion messages that explain what the test checks.
4. STRONGLY PREFER functional tests that import and execute the actual code (call functions, instantiate classes, check return values, side effects, or raised exceptions). Only fall back to source-level inspection (reading files as strings) when the change has absolutely no observable runtime effect.
5. Do NOT use `ast.parse()` or `ast.get_source_segment()` to verify changes — these produce fragile tests. If source inspection is truly necessary, read the file **by absolute path**: `source = open("/workspace/{file_path}").read()` and use simple string checks. NEVER resolve paths relative to `__file__`.
6. Keep imports minimal — only use stdlib and the repo itself.
7. Do NOT use unittest.mock to patch behavior — test the actual code.
8. The test file should be runnable with `python -m pytest <test_file>`.
9. CRITICAL: The test must actually RUN without errors (no crashes, no import errors, no TypeErrors, no AttributeErrors) on BOTH code versions. Only the assertions should differ. A test that crashes is NOT a valid test failure — design your test so it executes cleanly and uses assert statements to check conditions.
10. Keep tests simple and focused. One clear assertion about the key change is better than multiple fragile checks.
11. CRITICAL: The test MUST reliably FAIL on the "before" version. Do NOT include early returns, skip conditions, or fallback paths that could cause the test to silently pass. Every test path must reach an assertion.
12. If you cannot import a module (e.g. due to missing C extensions or complex setup), fall back to source file inspection using the absolute path `/workspace/{file_path}`.""".format(file_path=context.file_path)
    elif language in ("javascript", "typescript"):
        file_info = f"`{context.file_path}`"
        lang_label = cfg.name
        requirements = f"""\
1. Write a single, self-contained {cfg.test_framework} test file in {lang_label}.
2. The test MUST fail when run against the "before" version of the code and MUST pass when run against the "after" version.
3. Use clear assertion/expect messages that explain what the test checks.
4. Use `require()` or `import` to load the module under test. Use absolute paths from `/workspace/` (e.g. `require('/workspace/{context.file_path}')`).
5. STRONGLY PREFER functional tests that call functions and check return values, thrown errors, or side effects. Only fall back to reading the source file as a string when the change has absolutely no observable runtime effect.
6. If reading source files, use absolute paths: `fs.readFileSync('/workspace/{context.file_path}', 'utf8')`.
7. Do NOT mock/stub the module under test — test the actual code.
8. The test file should be runnable with `npx jest <test_file>`.
9. CRITICAL: The test must actually RUN without errors on BOTH code versions. Only the assertions should differ.
10. Keep tests simple and focused. One clear `expect()` about the key change is better than multiple fragile checks.
11. CRITICAL: The test MUST reliably FAIL on the "before" version. Do NOT include early returns or fallback paths that could cause the test to silently pass."""
    elif language == "go":
        file_info = f"`{context.file_path}`"
        requirements = """\
1. Write a single, self-contained Go test file (package *_test).
2. The test MUST fail when run against the "before" version of the code and MUST pass when run against the "after" version.
3. Use `testing.T` and clear `t.Errorf`/`t.Fatalf` messages that explain what the test checks.
4. Import the package under test. STRONGLY PREFER functional tests that call functions and check return values.
5. Only fall back to reading the source file as a string when the change has absolutely no observable runtime effect.
6. The test file should be runnable with `go test`.
7. CRITICAL: The test must compile and RUN without errors on BOTH code versions. Only the assertions should differ.
8. Keep tests simple and focused."""
    elif language == "java":
        file_info = f"`{context.file_path}`"
        requirements = """\
1. Write a single, self-contained JUnit 5 test class.
2. The test MUST fail when run against the "before" version of the code and MUST pass when run against the "after" version.
3. Use `assertEquals`, `assertTrue`, `assertThrows`, etc. with clear messages.
4. Import the class under test. STRONGLY PREFER functional tests that instantiate objects and call methods.
5. The test should be runnable with `mvn test` or `gradle test`.
6. CRITICAL: The test must compile and RUN without errors on BOTH code versions. Only the assertions should differ.
7. Keep tests simple and focused."""
    else:
        file_info = f"`{context.file_path}`"
        requirements = f"""\
1. Write a single, self-contained test file in {cfg.name} using {cfg.test_framework}.
2. The test MUST fail when run against the "before" version of the code and MUST pass when run against the "after" version.
3. Use clear assertion messages that explain what the test checks.
4. STRONGLY PREFER functional tests that import/require the code and check runtime behavior.
5. Only fall back to reading the source file as a string when the change has absolutely no observable runtime effect.
6. CRITICAL: The test must RUN without errors on BOTH code versions. Only the assertions should differ.
7. Keep tests simple and focused."""

    tag = cfg.code_block_tag

    prompt = f"""You are generating a test for a code review comment on the repository `{repo}`.

## Execution Environment
- The test runs inside a Docker container with the repo source code at `/workspace/` (which is the working directory).
- The repo package is installed. Standard library and repo dependencies are available.
- The test file is placed at `/workspace/<test_filename>` (repo root level).
- When referencing source files, ALWAYS use absolute paths starting with `/workspace/` (e.g. `/workspace/{context.file_path}`). NEVER use `__file__` or `os.path.dirname(__file__)` to locate source files.
- pytest runs with `-c /dev/null` (repo pytest config is ignored — no custom plugins or conftest).

## Review Comment
{context.comment_text}

## Comment Type
{comment_type}

## Language
{cfg.name} (test framework: {cfg.test_framework})

## File Under Review
{file_info}

## Diff Hunk (from the review)
```
{context.diff_hunk}
```

## Code BEFORE the review was addressed (should FAIL the test)
```{tag}
{context.before_patch_lines}
```

## Code AFTER the review was addressed (should PASS the test)
```{tag}
{context.after_patch_lines}
```

## Full file BEFORE (at head_commit)
```{tag}
{context.before_code}
```

## Full file AFTER (at merged_commit)
```{tag}
{context.after_code}
```

## Instructions
{instruction}

## Requirements
{requirements}

## Output
Return ONLY the test code, no explanations.
"""
    return prompt


# ---------------------------------------------------------------------------
# Code extraction & validation
# ---------------------------------------------------------------------------

def _extract_code_from_response(response_text: str, language: str) -> str:
    """Extract code from the LLM response.

    Tries language-specific fenced blocks first, then generic blocks,
    then falls back to treating the entire response as code.
    """
    cfg = get_language_config(language)
    tag = cfg.code_block_tag

    # Try language-specific fenced block
    pattern = rf"```{re.escape(tag)}\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Try generic fenced block
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    return response_text.strip()


def validate_test_syntax(test_code: str, language: str = "python") -> bool:
    """Validate generated test code syntax.

    For Python, uses ast.parse(). For other languages, performs basic
    structural checks (non-empty, contains test-related keywords).
    """
    if not test_code or not test_code.strip():
        return False

    if language == "python":
        try:
            ast.parse(test_code)
            return True
        except SyntaxError as e:
            logger.warning("Generated test has syntax error: %s", e)
            return False

    if language in ("javascript", "typescript"):
        # Must contain at least one test/it/describe block
        if re.search(r'\b(test|it|describe)\s*\(', test_code):
            return True
        logger.warning("Generated JS/TS test has no test/it/describe block")
        return False

    if language == "go":
        if re.search(r'func\s+Test\w+\s*\(', test_code):
            return True
        logger.warning("Generated Go test has no Test function")
        return False

    if language == "java":
        if re.search(r'@Test', test_code):
            return True
        logger.warning("Generated Java test has no @Test annotation")
        return False

    if language == "ruby":
        if re.search(r'\b(def\s+test_|describe|it)\b', test_code):
            return True
        logger.warning("Generated Ruby test has no test method")
        return False

    if language == "rust":
        if re.search(r'#\[test\]', test_code):
            return True
        logger.warning("Generated Rust test has no #[test] attribute")
        return False

    # For unrecognized languages, accept non-empty code
    return True


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------

def generate_test(
    context: CommentContext,
    repo: str,
    comment_type: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
    language: str = DEFAULT_LANGUAGE,
    environment_notes: str = "",
) -> tuple[str | None, LLMUsage]:
    """Generate a test for a review comment using an LLM.

    Args:
        context: The CommentContext with all before/after code.
        repo: Repository name (e.g. 'tobymao/sqlglot').
        comment_type: One of 'functional', 'structural', 'style', 'documentation'.
        model: LLM model to use.
        max_retries: Number of retries on syntax validation failure.
        language: Target language for the test (e.g. 'python', 'javascript').
        environment_notes: Optional notes about the execution environment
            (e.g. import availability) to include in the prompt.

    Returns:
        Tuple of (valid test code or None, accumulated LLMUsage).
    """
    cfg = get_language_config(language)
    prompt = _build_prompt(context, repo, comment_type, language)
    if environment_notes:
        prompt += f"\n\n## Environment Pre-flight Results\n{environment_notes}\n"
    total_usage = LLMUsage()

    for attempt in range(1 + max_retries):
        try:
            logger.info(
                "Generating %s test (attempt %d/%d, model=%s)",
                cfg.name, attempt + 1, 1 + max_retries, model,
            )
            response = chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=4096,
            )
            total_usage = total_usage + response.usage
            test_code = _extract_code_from_response(response.text, language)

            if validate_test_syntax(test_code, language):
                return test_code, total_usage

            logger.warning("Syntax validation failed, retrying...")
            prompt += (
                "\n\nNOTE: Your previous response had a syntax error or "
                "was missing required test structure. Please ensure valid "
                f"{cfg.name} syntax and include proper test functions."
            )

        except LLMError as e:
            logger.error("LLM API error: %s", e)
            if attempt == max_retries:
                return None, total_usage

    return None, total_usage


# ---------------------------------------------------------------------------
# Feedback & regeneration
# ---------------------------------------------------------------------------

def _classify_error(output: str) -> str | None:
    """Classify the type of error in test output.

    Returns a category string or None if the output looks like a clean assertion failure.
    """
    if not output:
        return None

    # Import / module errors — suggest source inspection fallback
    if any(s in output for s in [
        "ModuleNotFoundError", "ImportError", "No module named",
        "Cannot find module", "cannot resolve",
    ]):
        return "import_error"

    # File not found — path resolution issue
    if any(s in output for s in [
        "FileNotFoundError", "No such file or directory",
        "ENOENT", "Could not find",
    ]):
        return "path_error"

    # Conftest / plugin interference
    if any(s in output for s in [
        "unrecognized arguments", "Error importing plugin",
        "PytestConfigWarning", "no conftest",
    ]):
        return "config_error"

    # Build / extension errors
    if any(s in output for s in [
        "has not been built correctly",
        "Extension modules", "No module named",
        "sklearn.__check_build",
    ]):
        return "build_error"

    # Runtime crashes (not assertion failures)
    if any(s in output for s in [
        "TypeError", "AttributeError", "NameError",
        "RuntimeError", "SyntaxError",
        "ReferenceError", "NullPointerException",
    ]):
        return "runtime_error"

    return None


def _build_feedback_message(
    before_passed: bool,
    before_output: str,
    after_passed: bool,
    after_output: str,
) -> str:
    """Build a feedback message diagnosing why the test didn't achieve fail-then-pass.

    Classifies the error type and provides targeted advice, including strategy
    switching hints (e.g., suggesting source inspection when imports fail).
    """
    # Classify errors on both versions
    before_error = _classify_error(before_output)
    after_error = _classify_error(after_output)

    # Strategy switching hint based on error type
    def _strategy_hint(error_type: str | None, output: str) -> str:
        if error_type == "import_error":
            return (
                "\n\nSTRATEGY CHANGE: The module cannot be imported (likely due to "
                "missing C extensions or complex dependencies). Switch to a SOURCE "
                "INSPECTION test instead: read the source file as a string using "
                "`open('/workspace/<filepath>').read()` and check for the specific "
                "code pattern that changed. Do NOT try to import the module."
            )
        if error_type == "path_error":
            return (
                "\n\nPATH FIX: The file path is wrong. Use absolute paths starting "
                "with `/workspace/` (the repo root). NEVER use `__file__` or "
                "`os.path.dirname(__file__)` to resolve paths. Example: "
                "`open('/workspace/path/to/file.py').read()`"
            )
        if error_type == "config_error":
            return (
                "\n\nENVIRONMENT NOTE: pytest config interference. The test should "
                "be fully self-contained. Do not rely on any repo-level conftest.py "
                "or pytest plugins."
            )
        if error_type == "build_error":
            return (
                "\n\nSTRATEGY CHANGE: The package has unbuilt C extensions and "
                "cannot be imported. Switch to a SOURCE INSPECTION test: read the "
                "source file with `open('/workspace/<filepath>').read()` and check "
                "for the specific code change. Do NOT import the package."
            )
        if error_type == "runtime_error":
            return (
                "\n\nFIX: Your test has a runtime error (not an assertion failure). "
                "Make sure the test code itself is correct and handles the module's "
                "API properly. If the import works but the API call fails, check the "
                "function signatures and arguments against the actual code."
            )
        return ""

    # Both versions have the same error → likely an environment/approach issue
    if before_error and after_error:
        hint = _strategy_hint(after_error, after_output)
        return (
            "Your test crashed with a runtime error on BOTH versions "
            f"(error type: {after_error}). "
            "Here's the output from the 'after' version:\n\n"
            f"```\n{after_output[-3000:]}\n```"
            f"{hint}\n\n"
            "Fix the error so the test runs cleanly on both versions. "
            "Only the assertions should differ between versions."
        )

    if before_passed and after_passed:
        return (
            "The test PASSES on BOTH the 'before' and 'after' versions, "
            "so it doesn't detect the change. Your test needs a stronger "
            "assertion that specifically checks for the behavior/code that "
            "changed in the diff.\n\n"
            "IMPORTANT: Do NOT use early returns, skip conditions, or "
            "fallback paths that could cause the test to silently pass. "
            "Every execution path must reach an assertion.\n\n"
            "Look carefully at the diff — what SPECIFIC thing changed? "
            "Write an assertion that checks for exactly that change."
        )

    if before_passed and not after_passed:
        hint = _strategy_hint(after_error, after_output) if after_error else ""
        return (
            "The test logic is INVERTED — it passes on the 'before' version "
            "and fails on the 'after' version. We need the opposite: FAIL before, "
            "PASS after. Here's the 'after' failure output:\n\n"
            f"```\n{after_output[-3000:]}\n```"
            f"{hint}\n\n"
            "Flip your assertion so it checks for the NEW correct behavior "
            "(the behavior present in the 'after' code)."
        )

    if not before_passed and not after_passed:
        hint = _strategy_hint(after_error, after_output) if after_error else ""
        return (
            "The test correctly FAILS on the 'before' version — good! "
            "But it also FAILS on the 'after' version (it should PASS). "
            "Here's the 'after' version output:\n\n"
            f"```\n{after_output[-3000:]}\n```"
            f"{hint}\n\n"
            "Fix the test so it passes on the 'after' version while still "
            "failing on the 'before' version. The 'after' code is the "
            "CORRECT version — your assertions should match what the "
            "'after' code actually does."
        )

    return (
        f"Test result: before={'PASS' if before_passed else 'FAIL'}, "
        f"after={'PASS' if after_passed else 'FAIL'}. "
        "We need before=FAIL, after=PASS. Please fix the test."
    )


def regenerate_test(
    context: CommentContext,
    repo: str,
    comment_type: str,
    previous_test: str,
    before_passed: bool,
    before_output: str,
    after_passed: bool,
    after_output: str,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    environment_notes: str = "",
) -> tuple[str | None, LLMUsage]:
    """Regenerate a test using multi-turn conversation with execution feedback.

    Sends the original prompt, the previous test attempt, and a feedback
    message diagnosing the failure mode so the LLM can fix the test.

    Returns:
        Tuple of (valid test code or None, LLMUsage for this call).
    """
    cfg = get_language_config(language)
    prompt = _build_prompt(context, repo, comment_type, language)
    if environment_notes:
        prompt += f"\n\n## Environment Pre-flight Results\n{environment_notes}\n"
    feedback = _build_feedback_message(
        before_passed, before_output, after_passed, after_output,
    )

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"```{cfg.code_block_tag}\n{previous_test}\n```"},
        {"role": "user", "content": feedback},
    ]

    try:
        logger.info("Regenerating %s test with execution feedback (model=%s)", cfg.name, model)
        response = chat(
            messages=messages,
            model=model,
            max_tokens=4096,
        )
        test_code = _extract_code_from_response(response.text, language)

        if validate_test_syntax(test_code, language):
            return test_code, response.usage

        logger.warning("Regenerated test has syntax errors")
        return None, response.usage

    except LLMError as e:
        logger.error("LLM API error during regeneration: %s", e)
        return None, LLMUsage()
