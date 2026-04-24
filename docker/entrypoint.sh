#!/usr/bin/env bash
set -e

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec sleep infinity
