#!/usr/bin/env bash
set -euo pipefail

utility-peft prepare-data --config correlation --download
utility-peft train-source-head --config correlation --download
utility-peft run-correlation-benchmark --config correlation
