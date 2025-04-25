
# fork of ComfyUI Qwen2.5VL and Qwen2.5 wrapper for mlx-lm / mlx-lm (or any openai api server)
*forked from: https://github.com/alexcong/ComfyUI_QwenVL*

- modified `mlx-lm` server that leverages `mlx-vlm` library for qwen2.5-vl support: https://github.com/fblissjr/mlx-lm/tree/qwen-server/
    - specifically: https://github.com/fblissjr/mlx-lm/blob/qwen-server/mlx_lm/vlm_server.py

haphazardly put together and not well tested, but the mlx-lm server appears to be working pretty well for basic uses and multi-image support.