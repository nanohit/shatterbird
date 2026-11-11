#!/bin/sh
# Publishes the probe page to the gh-pages branch (index.html + det.js) without touching the working tree.
# Usage: sh gpu-probe/publish.sh   -> https://nanohit.github.io/shatterbird/
set -e
cd "$(dirname "$0")"
h=$(git hash-object -w det.html); j=$(git hash-object -w det.js); n=$(printf '' | git hash-object -w --stdin)
tree=$(printf '100644 blob %s\t.nojekyll\n100644 blob %s\tdet.js\n100644 blob %s\tindex.html\n' "$n" "$j" "$h" | git mktree)
parent=$(git rev-parse -q --verify refs/remotes/origin/gh-pages || true)
c=$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "Probe page $(date -u +%Y-%m-%dT%H:%MZ)")
git push origin "$c:refs/heads/gh-pages"
