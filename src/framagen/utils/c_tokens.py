"""Lightweight C tokenization shared by the agent guard and the benchmark adapters.

This is not a C parser. It is precise enough to (a) compare the executable
tokens of two files while ignoring ACSL/comments, (b) locate top-level
declarations and definitions of SV-COMP harness functions, and (c) classify
loop structure for dataset statistics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

TOKEN = re.compile(
    r"""(?P<literal>(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|(?:u|U|L)?'(?:\\.|[^'\\])*')
       |(?P<comment>/\*.*?\*/|//[^\n]*)
       |(?P<space>\s+)
       |(?P<word>[A-Za-z_]\w*|\d[\w.]*)
       |(?P<symbol>\#\#|<<=|>>=|\.\.\.|\+\+|--|->|&&|\|\||<=|>=|==|!=|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\S)""",
    re.VERBOSE | re.DOTALL,
)
ACSL_BLOCK = re.compile(r"/\*@.*?\*/", re.DOTALL)
ACSL_LINE = re.compile(r"(?m)^[ \t]*//@[^\n]*(?:\n|$)")
_ACSL_ANY = re.compile(r"/\*@.*?\*/|//@[^\n]*", re.DOTALL)
_PREPROCESSOR_LINE = re.compile(r"(?m)^[ \t]*#[^\n]*$")


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    start: int
    end: int


def iter_tokens(text: str) -> Iterator[Token]:
    """Yield significant tokens (no whitespace, no comments, no ACSL)."""
    for match in TOKEN.finditer(text):
        if match.lastgroup in {"space", "comment"}:
            continue
        yield Token(match.lastgroup or "", match.group(), match.start(), match.end())


def strip_acsl(text: str) -> str:
    return ACSL_LINE.sub("\n", ACSL_BLOCK.sub("", text))


def acsl_annotations(text: str) -> list[str]:
    """Return every ACSL annotation (block or line form) in source order."""
    return [m.group() for m in _ACSL_ANY.finditer(text)]


def executable_tokens(text: str) -> list[str]:
    return [t.text for t in iter_tokens(strip_acsl(text))]


def same_executable_tokens(original: str, candidate: str) -> bool:
    return executable_tokens(original) == executable_tokens(candidate)


def preprocessor_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PREPROCESSOR_LINE.finditer(text)]


@dataclass(frozen=True)
class TopLevelItem:
    """A top-level declaration or function definition."""

    name: str
    start: int  # first character of the item (after preceding whitespace)
    end: int  # one past the closing ';' or '}'
    is_definition: bool
    return_type: str  # tokens before the name, joined by spaces


def top_level_functions(text: str) -> list[TopLevelItem]:
    """Find top-level function prototypes and definitions.

    An item starts after the previous top-level ';', '}' or preprocessor line,
    and ends at the next top-level ';' (prototype) or the brace closing the
    body (definition). Only items whose first top-level parenthesis follows an
    identifier are reported.
    """
    tokens = [t for t in iter_tokens(text) if not _in_spans(t.start, preprocessor_spans(text))]
    pp_ends = [end for _, end in preprocessor_spans(text)]
    items: list[TopLevelItem] = []
    i = 0
    boundary = 0
    while i < len(tokens):
        # Advance the boundary past preprocessor lines that precede token i.
        for pp_end in pp_ends:
            if boundary < pp_end <= tokens[i].start:
                boundary = pp_end
        first = i
        name: str | None = None
        name_index = -1
        paren = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.text == "(":
                if paren == 0 and name is None and i > first and tokens[i - 1].kind == "word":
                    name = tokens[i - 1].text
                    name_index = i - 1
                paren += 1
            elif tok.text == ")":
                paren -= 1
            elif paren == 0 and tok.text == ";":
                break
            elif paren == 0 and tok.text == "{":
                break
            elif paren == 0 and tok.text == "=":
                name = None  # variable initializer, not a function
                name_index = -1
                # skip to the end of the declaration
                depth = 0
                while i < len(tokens):
                    if tokens[i].text in "({[":
                        depth += 1
                    elif tokens[i].text in ")}]":
                        depth -= 1
                    elif depth == 0 and tokens[i].text == ";":
                        break
                    i += 1
                break
            i += 1
        if i >= len(tokens):
            break
        start = tokens[first].start
        if tokens[i].text == "{":
            depth = 0
            while i < len(tokens):
                if tokens[i].text == "{":
                    depth += 1
                elif tokens[i].text == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            end = tokens[min(i, len(tokens) - 1)].end
            is_definition = name is not None
            # `struct s { ... } x;` style: consume the trailing declarator.
            if name is None:
                while i + 1 < len(tokens) and tokens[i + 1].text != ";":
                    i += 1
                if i + 1 < len(tokens):
                    i += 1
                end = tokens[i].end
        else:
            end = tokens[i].end
            is_definition = False
        if name is not None and name_index >= first:
            return_type = " ".join(
                t.text
                for t in tokens[first:name_index]
                if t.text not in {"extern", "static", "inline", "__inline"}
            )
            items.append(TopLevelItem(name, start, end, is_definition, return_type))
        boundary = end
        i += 1
    return items


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


LOOP_KEYWORDS = {"while", "for", "do"}
BRANCH_KEYWORDS = {"if", "switch", "?", "case"}


def _skip_parens(tokens: list[Token], i: int) -> int:
    """Given tokens[i] == '(', return the index just past its matching ')'."""
    depth = 0
    while i < len(tokens):
        if tokens[i].text == "(":
            depth += 1
        elif tokens[i].text == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _statement_end(tokens: list[Token], i: int) -> int:
    """Return the index just past the statement starting at tokens[i]."""
    if i >= len(tokens):
        return i
    head = tokens[i].text
    if head == "{":
        depth = 0
        while i < len(tokens):
            if tokens[i].text == "{":
                depth += 1
            elif tokens[i].text == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return i
    if head in {"while", "for", "switch"}:
        return _statement_end(tokens, _skip_parens(tokens, i + 1))
    if head == "if":
        j = _statement_end(tokens, _skip_parens(tokens, i + 1))
        if j < len(tokens) and tokens[j].text == "else":
            j = _statement_end(tokens, j + 1)
        return j
    if head == "do":
        j = _statement_end(tokens, i + 1)  # body
        # while ( cond ) ;
        if j < len(tokens) and tokens[j].text == "while":
            j = _skip_parens(tokens, j + 1)
        if j < len(tokens) and tokens[j].text == ";":
            j += 1
        return j
    # labeled statement `ident :` (not `?:`)
    if tokens[i].kind == "word" and i + 1 < len(tokens) and tokens[i + 1].text == ":":
        return _statement_end(tokens, i + 2)
    depth = 0
    while i < len(tokens):
        if tokens[i].text in "([{":
            depth += 1
        elif tokens[i].text in ")]}":
            depth -= 1
        elif depth == 0 and tokens[i].text == ";":
            return i + 1
        i += 1
    return i


@dataclass(frozen=True)
class LoopShape:
    loops: int
    max_depth: int
    branching_loops: int  # loops whose body (excluding nested loops' bodies) branches
    has_branches: bool  # any branch anywhere
    has_pointers: bool  # dereference/address-of/array parameter heuristics


def loop_shape(text: str, *, skip_functions: set[str] | None = None) -> LoopShape:
    """Classify loops by nesting and in-body branching, using token structure."""
    skip = skip_functions or set()
    full = [t for t in iter_tokens(strip_acsl(text))]
    pp = preprocessor_spans(strip_acsl(text))
    tokens = [t for t in full if not _in_spans(t.start, pp)]
    # Drop bodies of skipped (harness) functions.
    kept: list[Token] = []
    i = 0
    while i < len(tokens):
        if (
            tokens[i].kind == "word"
            and tokens[i].text in skip
            and i + 1 < len(tokens)
            and tokens[i + 1].text == "("
        ):
            j = _skip_parens(tokens, i + 1)
            if j < len(tokens) and tokens[j].text == "{":
                i = _statement_end(tokens, j)
                continue
        kept.append(tokens[i])
        i += 1
    tokens = kept

    loops = 0
    branching = 0
    max_depth = 0

    def body_range(i: int) -> tuple[int, int]:
        head = tokens[i].text
        if head == "do":
            return i + 1, _statement_end(tokens, i + 1)
        start = _skip_parens(tokens, i + 1)
        return start, _statement_end(tokens, start)

    def visit(lo: int, hi: int, depth: int) -> None:
        nonlocal loops, branching, max_depth
        i = lo
        while i < hi:
            tok = tokens[i]
            if tok.kind == "word" and tok.text in LOOP_KEYWORDS:
                # `while` closing a do-while is consumed by body_range of `do`.
                loops += 1
                max_depth = max(max_depth, depth + 1)
                b_lo, b_hi = body_range(i)
                if _branches_outside_nested(b_lo, b_hi):
                    branching += 1
                visit(b_lo, b_hi, depth + 1)
                i = b_hi
                if tok.text == "do":
                    # skip trailing `while (...) ;`
                    if i < len(tokens) and tokens[i].text == "while":
                        i = _skip_parens(tokens, i + 1)
                        if i < len(tokens) and tokens[i].text == ";":
                            i += 1
                continue
            i += 1

    def _branches_outside_nested(lo: int, hi: int) -> bool:
        i = lo
        while i < hi:
            tok = tokens[i]
            if tok.kind == "word" and tok.text in LOOP_KEYWORDS:
                # condition of a nested loop still counts as part of this body
                b_lo, b_hi = body_range(i)
                if tok.text != "do" and _has_branch(i + 1, b_lo):
                    return True
                i = b_hi
                continue
            if tok.text in BRANCH_KEYWORDS or tok.text in {"break", "continue", "goto"}:
                return True
            i += 1
        return False

    def _has_branch(lo: int, hi: int) -> bool:
        return any(tokens[k].text == "?" for k in range(lo, hi))

    visit(0, len(tokens), 0)
    has_branches = any(t.text in BRANCH_KEYWORDS for t in tokens)
    has_pointers = _pointer_heuristic(tokens)
    return LoopShape(loops, max_depth, branching, has_branches, has_pointers)


def _pointer_heuristic(tokens: list[Token]) -> bool:
    for k, tok in enumerate(tokens):
        if tok.text == "->":
            return True
        if tok.text == "*" and k > 0:
            prev = tokens[k - 1]
            # declarator `int *p` / `(int *)` / unary deref after operator
            if prev.kind == "word" and prev.text in {
                "int", "char", "unsigned", "long", "short", "void", "float", "double",
                "const", "_Bool", "size_t",
            }:
                return True
            if prev.text in {"(", ",", "=", ";", "{", "}", "return", "+", "-", "==", "!="}:
                return True
        if tok.text == "[" and k + 1 < len(tokens) and tokens[k + 1].text == "]":
            return True
    return False
