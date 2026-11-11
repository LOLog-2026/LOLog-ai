"""만화 만들기 (spec/functional-spec.md FR-COMIC, spec/storyboard.md). 2호 젯슨.

source 받기(→ STORYBOARD) → 콘티(1호 젯슨 LLM) → 검사, 틀리면 한 번 다시 묻기 → storyboard 보내기(→ DRAWING)
→ 고른 그림(기본 화자·프로필 캐릭터)이 있는 화자는 그 그림을 받아 참조로 → FLUX 컷 4장 → 말풍선 → 세로 4칸 PNG
→ image 보내기(→ DONE)

kind가 "avatar"인 메시지는 본인 사진 → 프로필 캐릭터(characters.md 3장). 사진은 메모리에서만 쓰고 남기지 않는다.

재시도하면 처음부터 다시 한다(source를 다시 받고 콘티도 다시 만듦). 백엔드는 이미 받은 storyboard가 다시 와도
같은 작업이면 받아 준다는 전제다.
"""

from __future__ import annotations

import io

from PIL import Image

from .api import InternalApi
from .draw import draw, draw_avatar
from .llm import Llm
from .storyboard_gen import StoryboardInvalid, fixed_looks, make_storyboard

__all__ = ["StoryboardInvalid", "avatar", "comic"]


def _image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def comic(msg: dict, api: InternalApi, llm: Llm) -> None:
    source = api.get_comic_source(msg["comicId"])
    storyboard = make_storyboard(source, llm)
    api.post_storyboard(msg["comicId"], storyboard)
    # 서명 주소는 10분짜리라 콘티를 만든 뒤 바로 받는다
    refs = {k: _image(api.fetch(c["imageUrl"])) for k, c in fixed_looks(source).items()}
    api.post_image(msg["comicId"], draw(storyboard, seed=int(msg["comicId"]), refs=refs))


# 사진만 받기로 해서(characters.md 3장) 나이대·성별·안경을 알 수 없다. 지금 쓰는 모델(콘티 LLM, FLUX 텍스트 인코더)은
# 사진을 못 보므로 외모 글은 중립 문장으로 둔다. 그림은 이 캐릭터 그림을 참조로 넣어 그리므로 얼굴·머리·안경은
# 그림에서 따라간다. 사진으로 외모 글을 만드는 방법은 따로 정한다(characters.md 5장 "외모 글").
PROFILE_APPEARANCE = "the same person as in the reference image"


def avatar(msg: dict, api: InternalApi) -> None:
    """프로필 캐릭터: 본인 사진 → 768 초상화. 시드는 jobId라 같은 사진으로 다시 만들어도 다른 그림이 나온다."""
    uid = int(msg["userId"])
    photo = _image(api.get_character_photo(uid))
    look = PROFILE_APPEARANCE
    api.post_character_avatar(uid, draw_avatar(photo, look, seed=int(msg["jobId"])), look)
