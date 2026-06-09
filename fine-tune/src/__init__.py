def __init__(
    self,
    model_name_or_path: str,
    tgt_lang: str,
    device: str = "cuda",
    lora_adapter: str = None,
    **init_kwargs,
):