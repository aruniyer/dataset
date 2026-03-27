"""LLM-based comment quality classification.

Classifies review comments as HIGH or LOW quality.
Low-quality comments (subjective preferences, non-actionable responses,
etc.) are identified so they can be filtered out before test generation.
"""

import json
import logging
import re

from .llm_client import DEFAULT_MODEL, LLMError, LLMUsage, chat

logger = logging.getLogger(__name__)

LOW_QUALITY_CATEGORIES = {
    "subjective_style_preference": (
        "Naming, formatting, or stylistic preference where alternatives are technically valid."
    ),
    "non_substantive_nitpick": (
        "Minor cosmetic issue (grammar/wording/formatting) with no technical impact."
    ),
    "discussion_without_actionable_outcome": (
        "Opinion exchange or discussion that does not identify a concrete verifiable issue."
    ),
    "non_actionable_response": (
        "Acknowledgment, approval, or meta-comment without requesting a technical change."
    ),
    "tangential_out_of_scope": (
        "Suggestion unrelated to the code under review or outside this change set."
    ),
    "incomplete_cryptic": (
        "Too vague/fragmentary to determine a concrete technical issue."
    ),
    "verbatim_code_solution": (
        "Provides a near-direct code patch or copy-paste solution, making it possible to pass by transcription rather than demonstrating code-improvement ability."
    ),
}


def _build_filter_prompt(comments: list[dict], instance_id: str) -> str:
    """Build the classification prompt for all comments in one instance."""
    comment_blocks = []
    for i, comment in enumerate(comments):
        text = comment.get("text", "")
        diff_hunk = comment.get("diff_hunk", "")
        path = comment.get("path", "")
        block = f"""### Comment {i}
**File:** `{path}`
**Diff hunk:**
```
{diff_hunk}
```
**Review text:**
{text}"""
        comment_blocks.append(block)

    comments_text = "\n\n".join(comment_blocks)
    low_categories_text = "\n".join(
        f"- {k}: {v}" for k, v in LOW_QUALITY_CATEGORIES.items()
    )

    return f"""You are classifying code review comments for benchmark construction.

Your task is to determine whether each comment is:

- HIGH: identifies an objectively verifiable technical issue or improvement
- LOW: reflects subjective preference, style, or non-actionable discussion

If a comment is HIGH, you must further classify it as:
- "defect"
- "code_quality"

# DEFINITION OF HIGH (OBJECTIVE)

A comment is HIGH if it identifies a property of the code that can be evaluated using technical criteria — independent of personal taste.

If two competent engineers applied the same standards, they would agree the issue exists.

A comment can be HIGH even if it is short or phrased as a question, as long as it clearly implies a concrete technical issue or improvement.

Classify HIGH comments as:

1) defect

Use "defect" if the comment identifies or strongly implies:

- Incorrect behavior
- Logical errors
- Edge-case bugs
- Violations of language/runtime rules
- Type/nullability issues
- Missing or incorrect error handling
- Security vulnerabilities
- Resource leaks or incorrect API usage
- Broken invariants
- Any issue that could cause incorrect results or runtime failure

Heuristic:
If the issue implies incorrectness or potential runtime failure → "defect"

2) code_quality

Use "code_quality" if the comment recommends a technically justified improvement that does NOT imply current incorrect behavior, such as:

- Refactoring to reduce duplication or complexity
- Improving maintainability or clarity in a way that reduces future bug risk
- Improving robustness or extensibility
- Performance improvements with clear technical reasoning
- Improving structure, testability, diagnosability
- Enforcing documented repository conventions
- Removing dead code or unnecessary parameters
- Improving internal API consistency
- Correcting misleading documentation about concrete runtime behavior or user-facing usage constraints
    - Classify documentation/docstring comments as HIGH only when they describe concrete, verifiable usage/runtime behavior (e.g., required arguments, side effects, failure modes, deprecations that break usage, constraints needed to run/use code correctly).
    - If a docstring suggestion is only wording/clarity/tone/completeness and does not materially change how the code is used or executed, classify as LOW (usually `non_substantive_nitpick`)

Heuristic:
If it improves robustness, maintainability, structure, or technical soundness — but does not imply incorrect behavior → "code_quality"

# DEFINITION OF LOW (SUBJECTIVE)

A comment is LOW if it does not identify an objectively verifiable technical issue.

This includes:

- Naming or formatting preferences
- Cosmetic changes (whitespace, wording, grammar)
- Alternative structures where both options are equally valid
- Personal stylistic opinions
- Non-actionable discussion
- Acknowledgments or approvals
- Suggestions unrelated to the code under review
- Comments too vague to evaluate
- Documentation/docstring edits that are only wording/clarity and do not change concrete usage/runtime guidance
- Comments that include a near-verbatim implementation/patch snippet that can be copied directly

Special benchmark rule:
- If the comment gives a direct copy-paste code solution (for example a full replacement block or concrete patch to transcribe), classify as LOW using `verbatim_code_solution`, even when the change itself is technically valid.

Use this LOW category mapping:
{low_categories_text}

Key principle:
If reasonable engineers could disagree purely based on taste or preference → LOW

# MULTI-TURN THREADS

If any part of a comment thread identifies an objectively verifiable technical issue → classify as HIGH.

# TASK

For each comment below, classify it as HIGH or LOW.

Comments:
{comments_text}

# OUTPUT FORMAT

Return a JSON array with exactly {len(comments)} objects, one per comment, in order:

[
  {{
    "comment_index": 0,
    "quality": "HIGH" or "LOW",
    "category": "defect" | "code_quality" | "<one LOW category key>",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of why the issue is objectively verifiable or subjective"
  }}
]
"""


def _parse_filter_response(response_text: str, expected_count: int) -> list[dict]:
    """Extract and validate JSON from LLM response."""
    # Try to extract from ```json ... ``` blocks
    pattern = r"```(?:json)?\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        text = max(matches, key=len).strip()
    else:
        text = response_text.strip()

    results = json.loads(text)

    if not isinstance(results, list):
        raise ValueError(f"Expected JSON array, got {type(results).__name__}")

    if len(results) != expected_count:
        raise ValueError(
            f"Expected {expected_count} results, got {len(results)}"
        )

    valid_categories = set(LOW_QUALITY_CATEGORIES.keys()) | {"defect", "code_quality"}
    valid_qualities = {"HIGH", "LOW"}

    for i, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"Result {i} is not a dict")

        quality = item.get("quality", "").upper()
        if quality not in valid_qualities:
            raise ValueError(f"Result {i} has invalid quality: {item.get('quality')}")
        item["quality"] = quality

        category = item.get("category", "")
        if category not in valid_categories:
            raise ValueError(f"Result {i} has invalid category: {category}")

        confidence = item.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            item["confidence"] = 0.5
        else:
            item["confidence"] = float(confidence)

        item.setdefault("reasoning", "")
        item["comment_index"] = i

    return results


def classify_comments(
    instance: dict,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
) -> dict:
    """Classify all comments in an instance as HIGH or LOW quality.

    Args:
        instance: Dataset instance dict with reference_review_comments.
        model: LLM model to use.
        max_retries: Number of retries on API/parse errors.

    Returns:
        Dict with instance_id, repo, comment classifications, and counts.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    comments = instance["reference_review_comments"]

    prompt = _build_filter_prompt(comments, instance_id)
    total_usage = LLMUsage()

    for attempt in range(1 + max_retries):
        try:
            logger.info(
                "Classifying %d comments for %s (attempt %d/%d, model=%s)",
                len(comments), instance_id, attempt + 1, 1 + max_retries, model,
            )
            response = chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=4096,
            )
            total_usage = total_usage + response.usage
            results = _parse_filter_response(response.text, len(comments))

            # Enrich results with comment metadata
            for i, result in enumerate(results):
                result["path"] = comments[i].get("path", "")
                result["text"] = comments[i].get("text", "")

            num_high = sum(1 for r in results if r["quality"] == "HIGH")
            num_low = sum(1 for r in results if r["quality"] == "LOW")

            return {
                "instance_id": instance_id,
                "repo": repo,
                "model": model,
                "usage": total_usage.to_dict(),
                "num_comments": len(comments),
                "num_high_quality": num_high,
                "num_low_quality": num_low,
                "comments": results,
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Parse error on attempt %d: %s", attempt + 1, e)
            if attempt == max_retries:
                logger.error("Failed to parse response after %d attempts", 1 + max_retries)
                break

        except LLMError as e:
            logger.error("LLM API error: %s", e)
            if attempt == max_retries:
                break

    # Return fallback: mark all as HIGH (safe default)
    logger.warning("Falling back to all-HIGH classification for %s", instance_id)
    fallback_comments = []
    for i, comment in enumerate(comments):
        fallback_comments.append({
            "comment_index": i,
            "quality": "HIGH",
            "category": "defect",
            "confidence": 0.0,
            "reasoning": "Classification failed — defaulting to HIGH",
            "path": comment.get("path", ""),
            "text": comment.get("text", ""),
        })

    return {
        "instance_id": instance_id,
        "repo": repo,
        "model": model,
        "usage": total_usage.to_dict(),
        "num_comments": len(comments),
        "num_high_quality": len(comments),
        "num_low_quality": 0,
        "comments": fallback_comments,
    }


def format_instance_results(result: dict, verbose: bool = False) -> str:
    """Format classification results as a readable terminal table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f" INSTANCE: {result['instance_id']} ({result['repo']})")
    lines.append("=" * 80)
    lines.append(
        f"  Comments: {result['num_comments']} total | "
        f"{result['num_high_quality']} high quality | "
        f"{result['num_low_quality']} low quality"
    )
    lines.append("")

    # Table header
    lines.append(
        f"  {'#':<3} {'Quality':<8} {'Confidence':<11} "
        f"{'Category':<30} {'File':<25} {'Comment'}"
    )
    lines.append(
        f"  {'--':<3} {'-------':<8} {'----------':<11} "
        f"{'------------------------------':<30} {'-------------------------':<25} "
        f"{'-------------------------'}"
    )

    for comment in result["comments"]:
        idx = comment["comment_index"]
        quality = comment["quality"]
        confidence = comment["confidence"]
        category = comment["category"]
        path = comment.get("path", "")
        text = comment.get("text", "").replace("\n", " ").strip()

        # Truncate long values for table display
        if len(path) > 24:
            path = "..." + path[-(24 - 3):]
        if len(text) > 40:
            text = text[:37] + "..."

        lines.append(
            f"  {idx:<3} {quality:<8} {confidence:<11.2f} "
            f"{category:<30} {path:<25} {text}"
        )

        if verbose and comment.get("reasoning"):
            reasoning = comment["reasoning"]
            lines.append(f"       Reasoning: {reasoning}")

    lines.append("")
    return "\n".join(lines)


def compute_summary(all_results: list[dict]) -> dict:
    """Compute aggregate statistics across all classified instances."""
    total_comments = 0
    total_high = 0
    total_low = 0
    category_counts: dict[str, int] = {}
    total_usage = LLMUsage()

    for result in all_results:
        total_comments += result["num_comments"]
        total_high += result["num_high_quality"]
        total_low += result["num_low_quality"]

        for comment in result["comments"]:
            cat = comment["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        if "usage" in result:
            u = result["usage"]
            total_usage = total_usage + LLMUsage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
                cost_usd=u.get("cost_usd", 0.0),
            )

    # Use the model from the first result that has one
    model = ""
    for r in all_results:
        if r.get("model"):
            model = r["model"]
            break

    summary: dict = {
        "total_instances": len(all_results),
        "total_comments": total_comments,
        "total_high_quality": total_high,
        "total_low_quality": total_low,
        "high_quality_rate": total_high / total_comments if total_comments else 0.0,
        "category_breakdown": dict(sorted(category_counts.items(), key=lambda x: -x[1])),
        "instance_results": [
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "num_comments": r["num_comments"],
                "num_high_quality": r["num_high_quality"],
                "num_low_quality": r["num_low_quality"],
            }
            for r in all_results
        ],
    }
    if model:
        summary["model"] = model
    summary["usage"] = total_usage.to_dict()
    return summary
