"""LeRobot runtime qualified for dataset admission and export (ADR 0003)."""

QUALIFIED_LEROBOT_VERSION = "0.6.1"
# LeRobot 0.6.1 declares transformers>=5.4,<5.6 and its pi0/pi0.5 code relies
# on 5.4+ embedding scaling; on 5.3 the image and language embeddings are
# 1/sqrt(hidden) too small (Studio ADR 0003, correction 2026-10-12).
QUALIFIED_TRANSFORMERS_VERSION = "5.5.4"
