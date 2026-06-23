#!/usr/bin/env bash
# Compile every .c under the repo and run it. Usage: ./build_and_test.sh [subdir]
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$ROOT}"
mkdir -p "$ROOT/build"
pass=0; fail=0
while IFS= read -r -d '' f; do
    name="$(basename "${f%.c}")"
    if gcc "$f" -O2 -o "$ROOT/build/$name" -lm 2>"$ROOT/build/_err"; then
        out="$("$ROOT/build/$name" 2>&1)"
        echo "OK   $name"
        echo "$out" | sed 's/^/        /'
        pass=$((pass+1))
    else
        echo "FAIL $name"
        sed 's/^/        /' "$ROOT/build/_err"
        fail=$((fail+1))
    fi
done < <(find "$TARGET" -name '*.c' -print0 | sort -z)
echo "---------------------------------------------"
echo "PASS=$pass  FAIL=$fail"
