"""만화 만들기 (spec/functional-spec.md FR-COMIC, spec/storyboard.md). 2호 젯슨.

source 받기(→ STORYBOARD) → 콘티(1호 젯슨 LLM) → 검사, 틀리면 한 번 다시 묻기 → storyboard 보내기(→ DRAWING)
→ 고른 그림(기본 화자·친구 아바타)이 있는 화자는 그 그림을 받아 참조로 → FLUX 컷 4장 → 말풍선 → 세로 4칸 PNG
→ image 보내기(→ DONE)

kind가 "avatar"인 메시지는 친구 사진 → 아바타(FR-FRIEND). 사진은 메모리에서만 쓰고 남기지 않는다.

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


AGE = {("CHILD", "F"): "a girl about eight years old", ("CHILD", "M"): "a boy about eight years old",
       ("YOUNG", "F"): "a young woman in her twenties", ("YOUNG", "M"): "a young man in his twenties",
       ("MIDDLE", "F"): "a middle-aged woman in her fifties", ("MIDDLE", "M"): "a middle-aged man in his fifties"}


def friend_appearance(msg: dict) -> str:
    """친구 외모 글. 사진은 LLM이 못 보므로 사용자가 고른 나이대·성별·안경으로 만든다."""
    return AGE[(msg["ageGroup"], msg["gender"])] + (", wearing glasses" if msg.get("glasses") else ", no glasses")


def avatar(msg: dict, api: InternalApi) -> None:
    fid = int(msg["friendId"])
    look = friend_appearance(msg)
    photo = _image(api.get_friend_photo(fid))
    api.post_friend_avatar(fid, draw_avatar(photo, look, seed=fid), look)
