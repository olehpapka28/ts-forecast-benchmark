# docker/julia/Dockerfile
FROM julia:1.11
ENV TZ=UTC

RUN julia -e 'using Pkg; \
    Pkg.add(["CSV", "DataFrames", "StateSpaceModels", "ARFIMA", "YAML", "BenchmarkTools", "TimerOutputs"]); \
    Pkg.precompile()'

WORKDIR /work
