"""Convert an SV-COMP ``unreach-call`` task into a Frama-C/WP proof task.

SV-COMP encodes the property "reach_error() is never called" with harness
functions that Frama-C does not know. This adapter removes their definitions
and declarations and re-declares them, at the top of the file, with ACSL
contracts that turn the reachability property into WP proof obligations:

- ``reach_error``          ``requires \\false``: every call site must be
                           proved unreachable.
- ``__VERIFIER_assert(c)`` ``requires c != 0``: every call site is a goal.
- ``assume_abort_if_not(c)`` / ``__VERIFIER_assume(c)``: ``ensures c != 0``
                           (an assumption on the normal-return path).
- ``abort``/``exit``       ``ensures \\false`` (no normal return).
- ``__VERIFIER_nondet_T``  ``assigns \\nothing``; result is any value of T.

The remaining program text is kept byte-for-byte, so an LLM only has to add
loop annotations (``loop invariant``, ``loop assigns``) for WP to discharge
the call-site goals. This mirrors the encoding used by Frama-C-SV.

The harness block is fenced by ``HARNESS_BEGIN``/``HARNESS_END`` markers; the
agent guard rejects candidates that alter it (see :func:`harness_block`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from framagen.utils.c_tokens import top_level_functions

HARNESS_BEGIN = "/* >>> sv2acsl harness: generated, do not edit >>> */"
HARNESS_END = "/* <<< sv2acsl harness <<< */"

_NONDET_TYPES = {
    "bool": "_Bool",
    "char": "char",
    "uchar": "unsigned char",
    "short": "short",
    "ushort": "unsigned short",
    "int": "int",
    "uint": "unsigned int",
    "long": "long",
    "ulong": "unsigned long",
    "longlong": "long long",
    "ulonglong": "unsigned long long",
    "float": "float",
    "double": "double",
    "size_t": "unsigned int",
    "u32": "unsigned int",
    "pointer": "void *",
}

_HARNESS_SPECS = {
    "reach_error": (
        "/*@ requires \\false;\n    assigns \\nothing; */\n"
        "void reach_error(void);"
    ),
    "__VERIFIER_assert": (
        "/*@ requires cond != 0;\n    assigns \\nothing; */\n"
        "void __VERIFIER_assert(int cond);"
    ),
    "assume_abort_if_not": (
        "/*@ assigns \\nothing;\n    ensures cond != 0; */\n"
        "void assume_abort_if_not(int cond);"
    ),
    "__VERIFIER_assume": (
        "/*@ assigns \\nothing;\n    ensures cond != 0; */\n"
        "void __VERIFIER_assume(int cond);"
    ),
}
_NORETURN_SPECS = {
    "abort": (
        "/*@ assigns \\nothing;\n    ensures \\false; */\n"
        "void abort(void) __attribute__((__noreturn__));"
    ),
    "exit": (
        "/*@ assigns \\nothing;\n    ensures \\false; */\n"
        "void exit(int status) __attribute__((__noreturn__));"
    ),
}
_REMOVE = set(_HARNESS_SPECS) | set(_NORETURN_SPECS)
_DROP_IF_UNUSED = {"__assert_fail", "__assert_perror_fail", "__assert"}
_NONDET = re.compile(r"\b__VERIFIER_nondet_(\w+)\b")
_UNSUPPORTED = re.compile(
    r"\b(__VERIFIER_atomic_\w*|__VERIFIER_error|pthread_create|malloc|calloc|realloc|free)\b"
)


class ConversionError(ValueError):
    pass


@dataclass
class Conversion:
    text: str
    removed: list[str] = field(default_factory=list)
    nondet: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def convert(source: str) -> Conversion:
    if "/*@" in source or "//@" in source:
        raise ConversionError("Task already contains ACSL annotations")
    used = set(re.findall(r"\b[A-Za-z_]\w*\b", source))
    if "reach_error" not in used and "__VERIFIER_assert" not in used:
        raise ConversionError("No reach_error/__VERIFIER_assert property site found")

    items = top_level_functions(source)
    spans: list[tuple[int, int, str]] = []
    nondet_types: dict[str, str] = {}
    for item in items:
        nondet = _NONDET.fullmatch(item.name)
        if item.name in _REMOVE or nondet:
            spans.append((item.start, item.end, item.name))
            if nondet and item.return_type:
                nondet_types[item.name] = item.return_type
    # Remove from the end so offsets stay valid.
    body = source
    removed: list[str] = []
    for start, end, name in sorted(spans, reverse=True):
        body = body[:start] + body[end:]
        removed.append(name)
    removed.reverse()

    # glibc assert prototypes are only used by the removed reach_error body;
    # drop them when nothing else refers to them, to keep the task readable.
    words = re.findall(r"\b[A-Za-z_]\w*\b", _strip_comments(body))
    for item in sorted(top_level_functions(body), key=lambda it: -it.start):
        if item.name in _DROP_IF_UNUSED and not item.is_definition and words.count(item.name) == 1:
            body = body[: item.start] + body[item.end :]
            removed.append(item.name)

    # What is still referenced after removing harness definitions?
    remaining = set(re.findall(r"\b[A-Za-z_]\w*\b", _strip_comments(body)))
    lines = [HARNESS_BEGIN]
    for name, spec in _HARNESS_SPECS.items():
        if name in remaining:
            lines.append(spec)
    for name, spec in _NORETURN_SPECS.items():
        if name in remaining:
            lines.append(spec)
    warnings: list[str] = []
    nondet_names = sorted({m.group(0) for m in _NONDET.finditer(body)})
    for name in nondet_names:
        suffix = name[len("__VERIFIER_nondet_"):]
        ctype = nondet_types.get(name) or _NONDET_TYPES.get(suffix)
        if ctype is None:
            raise ConversionError(f"Unknown nondet function type: {name}")
        lines.append(f"/*@ assigns \\nothing; */\nextern {ctype} {name}(void);")
    lines.append(HARNESS_END)

    if "reach_error" not in remaining and "__VERIFIER_assert" not in remaining:
        raise ConversionError("Property sites disappeared during conversion")
    unsupported = sorted(set(_UNSUPPORTED.findall(body)))
    if unsupported:
        warnings.append("Constructs outside the WP fragment: " + ", ".join(unsupported))
    if re.search(r"\bfloat\b|\bdouble\b", _strip_comments(body)):
        warnings.append("Floating-point arithmetic (WP float model is limited)")

    text = "\n".join(lines) + "\n\n" + _tidy(body).lstrip("\n")
    return Conversion(text=text, removed=removed, nondet=nondet_names, warnings=warnings)


def harness_block(text: str) -> str | None:
    """Return the fenced harness block, or None if the fences are missing."""
    start = text.find(HARNESS_BEGIN)
    end = text.find(HARNESS_END)
    if start < 0 or end < 0 or end < start:
        return None
    return text[start : end + len(HARNESS_END)]


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.DOTALL)


def _tidy(text: str) -> str:
    # Removing definitions leaves runs of blank lines; collapse them.
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
