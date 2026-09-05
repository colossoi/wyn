#!/usr/bin/env bash
set -euo pipefail

cargo test -p wyn --test package_manager_functional -- --nocapture
