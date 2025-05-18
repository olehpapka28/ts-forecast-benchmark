#!/usr/bin/env julia
# src/jl/run.jl
#
# Міні-тест: читаємо pipeline.yml і друкуємо підтвердження,
# що контейнер Julia працює.

using CSV
using DataFrames
using YAML                      # читання YAML-специфікації
using StateSpaceModels          # ETS, SARIMA тощо
using ARFIMA                    # Auto-ARIMA/SARIMA
using BenchmarkTools            # профілювання швидкості

# ────────────────────────────────────────────────────────────
# 1. Зчитуємо єдину специфікацію пайплайна
spec = YAML.load_file("pipeline.yml")

# 2. (Поки що) просто показуємо структуру YAML
println("🔹 Pipeline keys: ", keys(spec))

# 3. Smoke-повідомлення
println("✅ Hello from Julia container!  OK.")
