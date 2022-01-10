# # build with
# docker build -t tint .
# # run with
# docker run -i tint -f json < infile.txt > outfile.json
FROM maven:3.8-openjdk-8 AS builder

WORKDIR /app

ARG VERSION=develop

RUN apt-get update && apt-get install -y git-lfs

RUN mkdir tint-develop && \
    cd tint-develop && \
    git clone https://github.com/fbk/utils && \
    cd utils && \
    git checkout $VERSION && \
    mvn clean install && \
    cd .. && \
    git clone https://github.com/dhfbk/tint && \
    cd tint && \
    git checkout $VERSION && \
    mvn clean package -Pcomplete

# RUN cd /app/tint-develop/tint/tint-runner/target && \
#     tar -xzf tint-*.tar.gz
#
# WORKDIR /app/tint-develop/tint/tint-runner/target/tint
#
# ENTRYPOINT ["/app/tint-develop/tint/tint-runner/target/tint/tint-server.sh"]

FROM openjdk:8u282-jre

WORKDIR /app

COPY --from=builder /app/tint-develop/tint/tint-runner/target/tint-*.tar.gz .

RUN tar -xzf /app/tint-*.tar.gz && \
    rm /app/tint-*.tar.gz

WORKDIR /app/tint

ENTRYPOINT ["./tint-server.sh"]

EXPOSE 8012

CMD ["-p", "8012"]