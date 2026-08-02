#!/usr/bin/env bash
set -euo pipefail

utility-peft prepare-data --config correlation_pilot --download
utility-peft train-source-head --config correlation_pilot --download
utility-peft run-correlation-benchmark --config correlation_pilot

