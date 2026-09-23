"""Threshold calibration samples for the imprint-embed duplicate gate.

Three labeled groups of 20 pairs each. The whole point of calibration is that
lexical Jaccard cannot separate these groups (measured: true duplicate 0.383 vs
antonym conflict 0.700), so the cosine threshold must be chosen from data, not
guessed — a wrong threshold either lets duplicates through or blocks legitimate
distinct rules.

Groups
  duplicate  - same policy, different wording (MUST be caught)
  conflict   - opposite/mutually exclusive policy (MUST NOT be caught as duplicate)
  unrelated  - different topic (MUST NOT be caught)
"""

# fmt: off
DUPLICATE: list[tuple[str, str]] = [
    # The acceptance pair from the handover: Jaccard is only ~0.38.
    ("Go exported identifiers must use PascalCase", "Exported things use Pascal Case naming"),
    ("Wrap errors with fmt.Errorf and %w", "Always wrap errors using %w"),
    ("Use table-driven tests in Go", "Go tests should be table driven"),
    ("Prefer snake_case for Python functions", "Python function names use snake case"),
    ("Never commit secrets to the repo", "Do not commit secrets into the repository"),
    ("Run gofmt before committing", "Always run gofmt before you commit"),
    ("Keep functions under 50 lines", "Functions should stay shorter than 50 lines"),
    ("Use context as the first parameter", "context.Context must be the first argument"),
    ("Write comments in English", "Code comments should be written in English"),
    ("Avoid global mutable state", "Do not use global mutable variables"),
    ("Handle every error explicitly", "Every error must be handled explicitly"),
    ("Prefer composition over inheritance", "Use composition instead of inheritance"),
    ("Name interfaces by behavior", "Interfaces should be named after their behavior"),
    ("Keep imports grouped stdlib first", "Imports must be grouped with stdlib first"),
    ("Use tabs for indentation in Go", "Go files are indented with tabs"),
    ("Add unit tests for new packages", "New packages need unit tests"),
    ("Return early instead of nesting", "Prefer early returns to nested blocks"),
    ("Log structured fields not strings", "Use structured logging with fields"),
    ("Pin dependency versions", "Dependencies must be version pinned"),
    ("Document exported symbols", "Every exported symbol needs documentation"),
]

CONFLICT: list[tuple[str, str]] = [
    ("Use tabs for indentation", "Use spaces for indentation"),
    ("Always wrap errors with %w", "Return bare errors without wrapping"),
    ("Prefer tabs over spaces in Go", "Prefer spaces over tabs in Go"),
    ("Use snake_case for Python", "Use camelCase for Python"),
    ("Write comments in English", "Write comments in Chinese"),
    ("Avoid global mutable state", "Use a global config singleton"),
    ("Keep functions short", "Long functions are fine when linear"),
    ("Use table-driven tests", "Use assertion libraries instead of table tests"),
    ("Commit directly to main", "Never commit directly to main"),
    ("Pin dependency versions", "Use floating dependency ranges"),
    ("Handle every error explicitly", "Ignore errors from best-effort calls"),
    ("Prefer composition over inheritance", "Prefer deep inheritance hierarchies"),
    ("Log structured fields not strings", "Log plain formatted strings"),
    ("Return early instead of nesting", "Use a single exit point per function"),
    ("Vendor dependencies into the repo", "Do not vendor dependencies"),
    ("Use context as the first parameter", "Pass context only when needed"),
    ("Name interfaces by behavior", "Name interfaces with an I prefix"),
    ("Add unit tests for new packages", "Skip unit tests for trivial code"),
    ("Document exported symbols", "Self-documenting code needs no comments"),
    ("Use tabs for Go indentation", "Use two-space indentation everywhere"),
]

UNRELATED: list[tuple[str, str]] = [
    ("Go exported identifiers must use PascalCase", "Deploy to staging before release"),
    ("Wrap errors with %w", "Use dark mode in the editor"),
    ("Use table-driven tests in Go", "Take a coffee break every 90 minutes"),
    ("Prefer snake_case for Python functions", "Kubernetes pods restart on failure"),
    ("Never commit secrets to the repo", "Write the changelog in markdown"),
    ("Run gofmt before committing", "Schedule team meetings on Tuesday"),
    ("Keep functions under 50 lines", "Use Redis for session storage"),
    ("Use context as the first parameter", "Board games on Friday night"),
    ("Write comments in English", "Rotate TLS certificates quarterly"),
    ("Avoid global mutable state", "Back up photos to an external drive"),
    ("Handle every error explicitly", "Order lunch before noon"),
    ("Prefer composition over inheritance", "The train departs at 7am"),
    ("Name interfaces by behavior", "Water the plants twice a week"),
    ("Keep imports grouped stdlib first", "Upgrade postgres to version 16"),
    ("Use tabs for indentation in Go", "Buy a standing desk"),
    ("Add unit tests for new packages", "Renew the domain registration"),
    ("Return early instead of nesting", "Archive old invoices monthly"),
    ("Log structured fields not strings", "Paint the hallway white"),
    ("Pin dependency versions", "Walk ten thousand steps daily"),
    ("Document exported symbols", "Use a mechanical keyboard"),
]
# fmt: on

GROUPS: dict[str, list[tuple[str, str]]] = {
    "duplicate": DUPLICATE,
    "conflict": CONFLICT,
    "unrelated": UNRELATED,
}

# The single pair that decides whether the feature works at all.
ACCEPTANCE_PAIR = DUPLICATE[0]
