# AICarRacing (팀 B) — 재현용 이미지
# 빌드: docker buildx build --platform linux/amd64 -t aicarracing-teamb:latest --load .
# 평가: docker run --rm aicarracing-teamb:latest \
#         python -m scripts.evaluate_agent_2action --model ./models/ppo_2action4/best_model.pth --episodes 100 --seed 42
FROM python:3.11-slim-bookworm

# 헤드리스 렌더(SDL dummy) + matplotlib Agg + 무버퍼 로그
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    MPLBACKEND=Agg

# 시스템 의존성: box2d 빌드(swig/gcc), opencv/pygame 런타임(libGL/glib/SDL), ffmpeg(녹화)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        swig \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch는 CPU 휠로 먼저 설치(용량/이식성). 이후 requirements는 이미 충족된 torch를 건너뜀.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"

# 2) 나머지 의존성 (레이어 캐시를 위해 코드보다 먼저)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3) 프로젝트 코드 + 평가용 체크포인트 (대용량은 .dockerignore로 제외됨)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md requirements.txt ./
COPY BestSavedAgents/evaluated641.pth ./BestSavedAgents/
COPY models/ppo_2action4/best_model.pth ./models/ppo_2action4/
COPY models/obs_small/best_model.pth ./models/obs_small/

# 임포트/환경 등록 sanity 체크 (빌드 시점)
RUN python -c "import torch, gymnasium, cv2, pygame, Box2D; import src.car_racing_obstacles; print('deps OK | torch', torch.__version__, '| cv2', cv2.__version__)"

# 기본 실행: 베이스 모델 빠른 평가 (인자로 오버라이드 가능)
CMD ["python", "-m", "scripts.evaluate_agent_2action", \
     "--model", "./models/ppo_2action4/best_model.pth", "--episodes", "5", "--seed", "42"]
