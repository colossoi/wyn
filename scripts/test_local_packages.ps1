$ErrorActionPreference = "Stop"

cargo test -p wyn --test package_manager_functional -- --nocapture
