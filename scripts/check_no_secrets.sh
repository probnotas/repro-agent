#!/usr/bin/env bash
# Abort if an OpenRouter key appears anywhere in the working tree outside .env.
# Run this before every push:  ./scripts/check_no_secrets.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Built in two pieces so this file does not itself contain a matchable literal.
PREFIX='sk-or'
PATTERN="${PREFIX}-v1-[A-Za-z0-9]{16,}"

hits=$(grep -rInE "$PATTERN" . \
    --exclude-dir=.git \
    --exclude-dir=.cache \
    --exclude-dir=runs \
    --exclude-dir=__pycache__ \
    --exclude-dir='*.egg-info' \
    --exclude-dir=.venv \
    --exclude-dir=venv \
    --exclude=.env \
    --exclude="$(basename "$0")" \
    || true)

if [ -n "$hits" ]; then
    echo "ABORT: a real-looking OpenRouter key was found outside .env:" >&2
    # Print locations only -- never echo the key material itself.
    echo "$hits" | cut -d: -f1,2 | sort -u >&2
    exit 1
fi

# Belt and braces: refuse to let .env reach the index.
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
    echo "ABORT: .env is tracked by git. Run: git rm --cached .env" >&2
    exit 1
fi

echo "OK: no key material outside .env, and .env is not tracked."
