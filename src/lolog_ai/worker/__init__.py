"""Jetson 추론 워커.

Redis Streams(lolog:detect, lolog:comic, lolog:video)에서 작업을 꺼내 처리하고 결과를 메인 백엔드 /internal로 보낸다.
명세: LOLog-docs spec/api-spec.md "큐 메시지", spec/api-dto.md 4~5장, spec/deployment.md 7장.

    python -m lolog_ai.worker --role detect   # 1호 젯슨 (LLM)
    python -m lolog_ai.worker --role comic    # 2호 젯슨 (FLUX). 만화, 프로필 캐릭터, 영상 그림·합치기
    python -m lolog_ai.worker --role video    # 3호 젯슨 (TTS). 영상 콘티·목소리
"""
