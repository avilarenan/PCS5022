#!/usr/bin/env bash
set -euo pipefail

utility-peft prepare-data --config correlation_smoke
utility-peft run-correlation-benchmark --config correlation_smoke

