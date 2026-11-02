# Provenance

The public repository's history was rewritten once before it opened, to remove three working notes that held local paths (the Deviations in [`../PREREG.md`](../PREREG.md) record this). That changed every commit hash, so these two files tie the record back to the old hashes:

- `runs.json`: the private repo's GitHub Actions run list at the time, exported with `gh run list --limit 200 --json databaseId,name,headSha,createdAt,conclusion,event`. Timestamps come from GitHub's servers, and `headSha` is the old hash.
- `commit-map.txt`: old hash and new hash for every commit, written by `git filter-repo`.

The old repository stays private and its Actions logs stay there, so runs.json is a record and not something you can click through. Any commit hash in `review/final-review.md`, in a verdict log or in a stored CI log is an old hash; look it up in the map.
