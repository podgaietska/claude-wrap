# Releasing

A release is a git tag, `vX.Y.Z`, on a commit on `main`. Pushing the tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml), which:

1. runs the full CI suite on the tagged commit,
2. checks the tag is on `main`, matches `version` in `pyproject.toml`, and has a section in `CHANGELOG.md`,
3. builds the package and smoke-tests a clean install,
4. publishes it to [PyPI](https://pypi.org/p/claude-wrap) after you approve the `pypi` deployment,
5. creates the [GitHub Release](https://github.com/podgaietska/claude-wrap/releases) with the changelog section as its notes and the built files attached.

If any step fails, nothing after it runs. A failure before step 4 leaves nothing published.

## Choosing the version

`MAJOR.MINOR.PATCH`. While the version is `0.x`:

| Change | Bump | Example |
| --- | --- | --- |
| Bug fix, new model or price | PATCH | 0.3.1 → 0.3.2 |
| New feature, or support for a new Claude Code version | MINOR | 0.3.1 → 0.4.0 |
| Breaking change (config keys, CLI flags, data locations) | MINOR, marked **Breaking** in the changelog | 0.3.1 → 0.4.0 |

After 1.0, breaking changes bump MAJOR. To try a release out first, use a pre-release version such as `0.4.0rc1`: `pip install` skips it unless asked for by version or with `--pre`.

## Steps

1. **Prepare a release PR.** From an up-to-date `main`, branch `release/vX.Y.Z` and:
   - set `version = "X.Y.Z"` in `pyproject.toml`,
   - in `CHANGELOG.md`, rename `## Unreleased` to `## X.Y.Z - YYYY-MM-DD` (today's date) and add a new empty `## Unreleased` above it,
   - check the notes: `python scripts/release_notes.py vX.Y.Z`,
   - commit `chore(release): vX.Y.Z`, open the PR, and merge it once CI passes.

2. **Tag the merged commit and push the tag.**

   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

3. **Watch the Release workflow** in the Actions tab. When it reaches `publish-pypi` it waits for you: click **Review deployments**, tick `pypi` and approve.

4. **Check the result:** the version on https://pypi.org/p/claude-wrap, the release on GitHub, and a fresh install:

   ```bash
   pipx install --force claude-wrap==X.Y.Z && wrap --version
   ```

## When something goes wrong

- **Failed before publishing** (tests, tag check, build): nothing was published. Delete the tag, fix on `main` through a PR, and tag again:

  ```bash
  git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z
  ```

- **Published but broken:** a version on PyPI can never be replaced or reused, even after deleting it. Fix it and release the next PATCH. If the broken version is harmful, *yank* it on PyPI (project → Manage → Release → Options → Yank): it stays installable by exact version but `pip` stops choosing it.

- **Published, but the GitHub Release step failed:** re-run the failed job from the workflow run page.

## Dry run on TestPyPI

[TestPyPI](https://test.pypi.org) is a separate sandbox copy of PyPI. To rehearse the build and upload without a tag, run the workflow by hand from `main`:

```bash
gh workflow run release.yml
```

It runs CI, builds, and publishes the current version to TestPyPI (skipping it if that version is already there). Install it from there with:

```bash
pipx install --force --pip-args="--extra-index-url https://pypi.org/simple/" --index-url https://test.pypi.org/simple/ claude-wrap
```
