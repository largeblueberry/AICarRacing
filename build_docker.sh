#!/usr/bin/env bash
# AICarRacing 도커 이미지 빌드 → 스모크 테스트 → save+gzip → e-class용 분할.
# 사용: bash build_docker.sh        (기본 linux/amd64, 90MB 분할)
#       PLATFORM=linux/arm64 bash build_docker.sh   (네이티브 빠른 빌드, Apple Silicon 전용)
set -euo pipefail

IMG="aicarracing-teamb:latest"
PLATFORM="${PLATFORM:-linux/amd64}"
OUT="aicarracing_teamb_image.tar.gz"
PARTSIZE="${PARTSIZE:-90m}"

echo "==[1/4] build ($PLATFORM) =="
docker buildx build --platform "$PLATFORM" -t "$IMG" --load .

echo "==[2/4] smoke test (eval 1 episode in-container) =="
docker run --rm "$IMG" python -m scripts.evaluate_agent_2action \
    --model ./models/ppo_2action4/best_model.pth --episodes 1 --max-steps 40

echo "==[3/4] docker save | gzip -> $OUT =="
docker save "$IMG" | gzip > "$OUT"

echo "==[4/4] split -> ${OUT}.part_* (${PARTSIZE}) =="
rm -f "${OUT}.part_"*
split -b "$PARTSIZE" "$OUT" "${OUT}.part_"
echo "--- 결과 ---"
ls -lh "$OUT" "${OUT}.part_"* 2>/dev/null
echo
echo "복원(채점자): cat ${OUT}.part_* > ${OUT} && gunzip -c ${OUT} | docker load"
echo "실행 예시:   docker run --rm $IMG python -m scripts.evaluate_agent_2action --model ./models/ppo_2action4/best_model.pth --episodes 100 --seed 42"
