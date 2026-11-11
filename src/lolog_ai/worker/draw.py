"""콘티 → 4컷 PNG (2호 젯슨, FLUX.2 klein 4B nf4).

8GB 통합 메모리에 텍스트 인코더(nf4 약 2.8GB)와 이미지 모델(nf4 약 2.3GB)을 같이 둘 수 없어서(#6 측정:
이미지 모델만으로도 최대 RAM 7.2GB) 만화 하나마다 순서대로 올렸다 내린다.
  1) 텍스트 인코더 → 컷 4개 프롬프트 인코딩 → 내림
  2) 이미지 모델 + VAE → 컷 4장 → 얼굴 검출 → 말풍선 → 세로 4칸 → 내림

양자화는 처음 한 번만 한다. bf16 원본을 읽어 nf4로 바꾸는 데 각각 약 97초 걸렸으므로(#6), 처음 바꾼 결과를
FLUX_NF4_DIR에 저장해 두고 다음부터는 거기서 바로 읽는다.

환경 변수
  FLUX_MODEL     기본 black-forest-labs/FLUX.2-klein-4B (HF 캐시에 받아 둠, HF_HUB_OFFLINE=1 권장)
  FLUX_NF4_DIR   기본 /opt/lolog/models/flux2-klein-4b-nf4
  FLUX_LORA      화풍 LoRA(.safetensors) 경로. 없으면 원본 그대로

참조 그림: 콘티 화자에 기본 화자·프로필 캐릭터가 붙어 있으면 그 화자가 나오는 컷에 그림을 FLUX.2 klein의
참조 이미지(image)로 넣는다. 2호 벤치(2026-11-10): 참조 3장까지 메모리 부족 없음, 컷 23초 → 384px 참조 1장
26초·2장 30초·3장 37초. 384와 512는 눈으로 차이가 없어 384로 줄여 넣는다.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

MODEL = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")
NF4_DIR = Path(os.environ.get("FLUX_NF4_DIR", "/opt/lolog/models/flux2-klein-4b-nf4"))
LORA = os.environ.get("FLUX_LORA") or None
PANEL = 768
REF_SIZE = 384
STEPS = 4
FONT_SIZE = 36


def _free() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bnb(lib):
    import torch
    return lib.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)


def load_text_encoder():
    import torch
    import transformers
    from transformers import Qwen3ForCausalLM

    saved = NF4_DIR / "text_encoder"
    if saved.exists():
        return Qwen3ForCausalLM.from_pretrained(saved, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    log.info("텍스트 인코더를 nf4로 처음 바꾸는 중(약 100초), %s에 저장", saved)
    te = Qwen3ForCausalLM.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                                          device_map="cuda", quantization_config=_bnb(transformers))
    te.save_pretrained(saved)
    return te.eval()


def load_transformer():
    import diffusers
    import torch
    from diffusers import Flux2Transformer2DModel

    saved = NF4_DIR / "transformer"
    if saved.exists():
        return Flux2Transformer2DModel.from_pretrained(saved, torch_dtype=torch.bfloat16)
    log.info("이미지 모델을 nf4로 처음 바꾸는 중(약 100초), %s에 저장", saved)
    model = Flux2Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                    quantization_config=_bnb(diffusers))
    model.save_pretrained(saved)
    return model


def encode(prompts: list[str]):
    import torch
    from diffusers import Flux2KleinPipeline

    te = load_text_encoder()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, text_encoder=te, transformer=None, vae=None,
                                              torch_dtype=torch.bfloat16)
    with torch.inference_mode():
        embeds = [pipe.encode_prompt(p, device="cuda")[0] for p in prompts]
    pipe = te = None
    _free()
    return embeds


def _image_pipe():
    import torch
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(MODEL, transformer=load_transformer(), text_encoder=None,
                                              tokenizer=None, torch_dtype=torch.bfloat16)
    pipe.vae.to("cuda")
    if LORA:
        pipe.load_lora_weights(LORA)
    return pipe


def _render(pipe, embeds, seed: int, refs: list | None = None):
    import torch

    with torch.inference_mode():
        latents = pipe(image=refs or None, prompt_embeds=embeds, width=PANEL, height=PANEL,
                       guidance_scale=1.0, num_inference_steps=STEPS, output_type="latent",
                       generator=torch.Generator("cuda").manual_seed(seed)).images
        img = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
        return pipe.image_processor.postprocess(img, output_type="pil")[0]


def _ref(img):
    from PIL import Image
    return img.convert("RGB").resize((REF_SIZE, REF_SIZE), Image.LANCZOS)


def draw(storyboard: dict, seed: int, refs: dict | None = None) -> bytes:
    """콘티 → PNG 바이트. 파일을 남기지 않는다(deployment.md 7장 임시 파일).

    refs: 화자 키 → 참조 그림(PIL). 없는 화자는 외모 글만으로 그린다.
    """
    from ..bubble import draw_bubbles, place_bubbles
    from ..faces import detect_faces, match_speakers
    from ..render import compose, prompt, ref_keys

    refs = {k: _ref(v) for k, v in (refs or {}).items()}
    t0 = time.perf_counter()
    embeds = encode([prompt(storyboard, p, set(refs)) for p in storyboard["panels"]])
    t1 = time.perf_counter()
    pipe = None
    try:
        pipe = _image_pipe()
        t2 = time.perf_counter()
        panels = []
        for i, panel in enumerate(storyboard["panels"]):
            img = _render(pipe, embeds[i], seed + i, [refs[k] for k in ref_keys(panel, set(refs))])
            if panel["dialogue"]:
                faces = detect_faces(img)
                pos = {c["key"]: c["position"] for c in panel["cast"]}
                placed = place_bubbles(img, panel["dialogue"], pos, font_size=FONT_SIZE,
                                       faces=faces, speaker_faces=match_speakers(faces, pos))
                img = draw_bubbles(img, panel["dialogue"], placed, font_size=FONT_SIZE)
            panels.append(img)
    finally:
        embeds = pipe = None
        _free()
    buf = io.BytesIO()
    compose(panels).save(buf, format="PNG", optimize=True)
    log.info("그림: 인코딩 %.0f초, 모델 올림 %.0f초, 그리기 %.0f초, 참조 %d명, %d바이트",
             t1 - t0, t2 - t1, time.perf_counter() - t2, len(refs), buf.tell())
    return buf.getvalue()


def draw_avatar(photo, appearance: str, seed: int) -> bytes:
    """본인 사진 → 기본 화자와 같은 화풍·구도의 768 프로필 캐릭터 PNG. 사진은 메모리에서만 쓴다."""
    from PIL import ImageOps

    from ..render import portrait_prompt

    photo = ImageOps.exif_transpose(photo).convert("RGB")
    photo.thumbnail((512, 512))  # 벤치에서 잰 최대 참조 크기(512)를 넘지 않게
    embeds = encode([portrait_prompt(appearance, from_photo=True)])
    pipe = None
    try:
        pipe = _image_pipe()
        img = _render(pipe, embeds[0], seed, [photo])
    finally:
        embeds = pipe = None
        _free()
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def draw_scenes(sb: dict, seed: int, refs: dict | None = None) -> list:
    """영상 콘티(video_storyboard) → 장면 그림(PIL, 768) 목록. 말풍선 없이 그대로 쓴다(#33).
    모델을 올리고 내리는 순서는 draw()와 같다."""
    from ..render import ref_keys, video_prompt

    refs = {k: _ref(v) for k, v in (refs or {}).items()}
    t0 = time.perf_counter()
    embeds = encode([video_prompt(sb, s, set(refs)) for s in sb["scenes"]])
    pipe = None
    try:
        pipe = _image_pipe()
        out = [_render(pipe, embeds[i], seed + i, [refs[k] for k in ref_keys(s, set(refs))])
               for i, s in enumerate(sb["scenes"])]
    finally:
        embeds = pipe = None
        _free()
    log.info("영상 장면 %d장 %.0f초, 참조 %d명", len(out), time.perf_counter() - t0, len(refs))
    return out
