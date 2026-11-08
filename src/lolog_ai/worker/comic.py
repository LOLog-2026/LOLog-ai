"""만화 만들기 (spec/functional-spec.md FR-COMIC, spec/storyboard.md). 2호 젯슨.

source 받기(→ STORYBOARD) → 콘티(1호 젯슨 LLM) → 검사, 틀리면 한 번 다시 묻기 → storyboard 보내기(→ DRAWING)
→ FLUX 컷 4장 → 말풍선 → 세로 4칸 PNG → image 보내기(→ DONE)

재시도하면 처음부터 다시 한다(source를 다시 받고 콘티도 다시 만듦). 백엔드는 이미 받은 storyboard가 다시 와도
같은 작업이면 받아 준다는 전제다.
"""

from __future__ import annotations

from .api import InternalApi
from .draw import draw
from .llm import Llm
from .storyboard_gen import StoryboardInvalid, make_storyboard

__all__ = ["StoryboardInvalid", "comic"]


def comic(msg: dict, api: InternalApi, llm: Llm) -> None:
    source = api.get_comic_source(msg["comicId"])
    storyboard = make_storyboard(source, llm)
    api.post_storyboard(msg["comicId"], storyboard)
    api.post_image(msg["comicId"], draw(storyboard, seed=int(msg["comicId"])))
