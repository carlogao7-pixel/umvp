# -*- coding: utf-8 -*-
"""视频嵌入模块（WEMMClient + EmbedStore + VideoEmbedStage）。PIPE0004 向量档索引核心。"""

from .wemm_client import (  # noqa: F401
    WEMM_MRL_DIMS, WemmHttpError, WEMMClient, check_dimension, resolve_wemm_cfg,
)
from .embed_store import ClipHit, EmbedStore, FrameHit  # noqa: F401
from .video_embed_stage import VideoEmbedStage  # noqa: F401
