# docker/r/Dockerfile
FROM rocker/verse:4.4.0
ENV TZ=UTC
RUN install2.r --error forecast bench tsibble fable urca
WORKDIR /work
