"""Tests for extract_patch, normalize_code, and count_diff_lines functions in patch_env.py"""

from tinker_cookbook.recipes.rlvr.patch_env import (
    count_diff_lines,
    extract_last_markdown_block,
    normalize_code,
)


def test_extract_patch_multiline():
    """Standard multiline code block"""
    text = "```diff\n*** Begin Patch\n...\n*** End Patch\n```"
    assert "Begin Patch" in extract_last_markdown_block(text)


def test_extract_patch_single_line():
    """Everything on one line"""
    text = "```diff *** Begin Patch *** End Patch```"
    assert "Begin Patch" in extract_last_markdown_block(text)


def test_extract_patch_with_preamble():
    """Text before the code block"""
    text = "Here's the patch:\n```diff\n*** Begin Patch\n*** End Patch\n```"
    assert extract_last_markdown_block(text).startswith("*** Begin Patch")


def test_extract_patch_no_code_block():
    """Raw patch without markdown"""
    text = "*** Begin Patch\n...\n*** End Patch"
    assert extract_last_markdown_block(text) == text.strip()


def test_extract_patch_no_diff_tag():
    """Code block without 'diff' language tag"""
    text = "```\n*** Begin Patch\n*** End Patch\n```"
    assert "Begin Patch" in extract_last_markdown_block(text)


def test_extract_patch_multiple_blocks_returns_last():
    """Multiple code blocks - should return the LAST one"""
    text = """Here's a code snippet:

```python
def foo():
    pass
```
The spec says to use triple backticks formatting for answer e.g. ```diff...```
but the only snippet extracted by the postprocessor is the last one

```python
def baz():
  pass
```
and nothing else :)
"""
    result = extract_last_markdown_block(text)
    assert "baz" in result
    assert "foo" not in result


# Tests for normalize_code (used by PatchRelaxedMatchEnv)


def test_normalize_code_preserves_indentation():
    """Normalize should preserve indentation (semantic in Python)"""
    code1 = "def foo():\n    pass"
    code2 = "def foo():\npass"
    # Different indentation = different code
    assert normalize_code(code1) != normalize_code(code2)


def test_normalize_code_preserves_empty_lines():
    """Empty lines are preserved (can be semantic)"""
    code1 = "def foo():\n\n\npass"
    code2 = "def foo():\npass"
    # Different number of empty lines = different (as sets, one has "" the other doesn't)
    assert normalize_code(code1) != normalize_code(code2)


def test_normalize_code_order_independent():
    """Line order doesn't matter (set comparison)"""
    code1 = "line1\nline2\nline3"
    code2 = "line3\nline1\nline2"
    assert normalize_code(code1) == normalize_code(code2)


def test_normalize_code_detects_differences():
    """Different lines should not match"""
    code1 = "def foo():\npass"
    code2 = "def bar():\npass"
    assert normalize_code(code1) != normalize_code(code2)


# Tests for count_diff_lines (used by PatchMinimalDiffEnv)


def test_count_diff_lines_basic():
    """Count added and removed lines"""
    patch = "+added line\n-removed line\n context line"
    assert count_diff_lines(patch) == 2


def test_count_diff_lines_excludes_headers():
    """Should exclude --- and +++ file headers"""
    patch = "--- a/file.py\n+++ b/file.py\n+added line"
    assert count_diff_lines(patch) == 1


def test_count_diff_lines_multiple_hunks():
    """Count across multiple hunks"""
    patch = """\
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 context
+added1
-removed1
@@ -10,3 +11,4 @@
 more context
+added2
"""
    assert count_diff_lines(patch) == 3


def test_count_diff_lines_empty():
    """Empty patch has zero changes"""
    assert count_diff_lines("") == 0
    assert count_diff_lines("context only") == 0

