#!/usr/bin/env sh
# `npm ci`, retried — for CI runners on a flaky network.
#
# Some packages download their own binaries from a postinstall script, and those
# downloads are NOT covered by npm's fetch-retry logic. The Pages deploy of 2026-07-26
# died with `connect ETIMEDOUT 150.171.110.146:443` inside
# onnxruntime-node/script/install while every other step was healthy. (That binary is a
# Node runtime the browser bundle never loads, but it still has to install.)
#
# Retries the whole install, then FAILS LOUDLY. It never swallows the error and never
# proceeds with a half-populated node_modules — a partial install is wiped between
# attempts, because npm will otherwise happily build on top of one.
#
# Usage (from a directory with a package-lock.json):
#   sh path/to/npm-ci-retry.sh [extra npm ci args...]
# Env: NPM_CI_ATTEMPTS (default 3), NPM_CI_RETRY_DELAY seconds (default 15).
set -eu

ATTEMPTS="${NPM_CI_ATTEMPTS:-3}"
DELAY="${NPM_CI_RETRY_DELAY:-15}"

attempt=1
while [ "$attempt" -le "$ATTEMPTS" ]; do
  if npm ci "$@"; then
    exit 0
  fi
  if [ "$attempt" -lt "$ATTEMPTS" ]; then
    echo "npm-ci-retry: attempt $attempt/$ATTEMPTS failed; retrying in ${DELAY}s" >&2
    rm -rf node_modules
    sleep "$DELAY"
  fi
  attempt=$((attempt + 1))
done

echo "npm-ci-retry: npm ci failed after $ATTEMPTS attempts" >&2
exit 1
