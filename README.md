# Wyoming Whisper GPU – DGX Spark / GB10

Wyoming-kompatibler Whisper.cpp Server mit CUDA-Beschleunigung für den **NVIDIA DGX Spark (Grace Blackwell GB10, ARM64)**.

Getestet mit: Home Assistant + Wyoming Protocol

## Voraussetzungen

- NVIDIA DGX Spark (GB10, ARM64)
- Docker mit `nvidia-container-toolkit`
- NGC-Account für `nvcr.io` (kostenlos): [ngc.nvidia.com](https://ngc.nvidia.com)
- Externes Docker-Netzwerk `ai_network`:
  ```bash
  docker network create ai_network
  ```

## Setup

```bash
# 1. NGC Login
docker login nvcr.io

# 2. Repo klonen
git clone https://github.com/Alex10000121/dgx_whisper_gpu_wyoming.git
cd dgx_whisper_gpu_wyoming

# 3. Bauen & starten
docker compose up -d --build

# 4. Logs beobachten
docker compose logs -f
```

Beim ersten Start wird das Modell automatisch heruntergeladen und unter `/data/` gecacht.

## Konfiguration

In `docker-compose.yaml` die Umgebungsvariablen anpassen:

| Variable | Standard | Optionen |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3` |
| `WHISPER_LANG` | `de` | `de`, `en`, `fr`, `auto`, ... |
| `WHISPER_BEAM_SIZE` | `5` | `1`–`10` |

## Home Assistant

Einbinden als Wyoming-Integration:
- Host: IP des DGX Spark
- Port: `10300`
