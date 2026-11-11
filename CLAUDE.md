# LOLog 프로젝트 규칙

LOLog: 일상 대화에서 웃긴 이야기를 찾아 카드로 모으고 4컷만화로 그려 주는 서비스.
조직: https://github.com/LOLog-2026 (팀 3명, 2026-11-02 ~ 2026-11-27)

## Git 작업 규칙 (GitFlow)

- `main`, `develop`에 직접 커밋·push 하지 않는다. 작업은 반드시 브랜치에서 한다.
- 브랜치는 `develop`에서 만든다. 이름은 `타입/이슈번호-영문-설명`, `#`은 넣지 않는다.
  - `feature/23-login` (새 기능), `fix/12-auth-error` (버그), `hotfix/31-upload-crash` (운영 긴급 수정)
- 이슈가 없으면 먼저 이슈를 만들거나 사용자에게 이슈 번호를 묻는다.
- PR은 `develop`을 대상으로 만든다. Claude는 **PR 생성까지만** 하고 머지는 사람이 한다.
- 병합 방식은 Merge. PR 승인은 필요 없지만 테스트는 통과해야 한다.

## 커밋 메시지 (Conventional Commits)

```
type(scope): subject

body

Fixes #이슈번호
```

- type: `feat` `fix` `docs` `style` `refactor` `test` `chore`
- 제목: 50자 이내 명령문, 끝에 마침표 없음. 한국어로 쓴다. 예: `feat(user): 프로필 이미지 업로드 기능 추가`
- scope: 선택. 쓸 때는 영어 소문자 한 단어(`auth`, `record`), 애매하면 생략
- 본문: 변경 이유와 상세 설명, 제목과 한 줄 띄움
- footer: `Fixes #이슈번호`

## 이슈

- 제목: `[Task]`, `[Bug]`, `[Feature]` 머리말 + 50자 이내. 예: `[Bug] 로그인 시 500 에러 발생`
- 이슈 타입(Task / Bug / Feature)을 지정한다. 기본 레이블 `bug`, `enhancement`는 쓰지 않는다.
- 레이블
  - 도메인: `front` 화면 디자인 / `app` 안드로이드 앱 / `backend` 서버(인증 포함) / `ai` 모델·전처리·추론 / `infra` 서버·배포·Jetson 환경
  - 우선순위: `P0` 시연에 꼭 필요 / `P1` 있으면 좋음 / `P2` 여유 있으면
  - 기타: `documentation` `good first issue` `help wanted` `question` `duplicate` `invalid` `wontfix`
- 만들기 전에 같은 이슈가 있는지 검색한다.

## PR

- 제목: 커밋 타입 머리말(첫 글자 대문자) + 이슈 번호. 예: `[Feat] 데이터 검색 필터 기능 추가 (#45)`
  - 머리말: `[Feat]` `[Fix]` `[Docs]` `[Refactor]` `[Test]` `[Chore]` `[Style]`
- 본문 템플릿:

```markdown
## 📌 관련 이슈
Closes #

## ✨ 변경 사항
-

## 🔁 추가 수정 사항
- (리뷰 결과에 따라 PR을 올린 뒤 커밋을 더 했다면 무엇을 고쳤는지 적는다. 없으면 비워 둔다.)

## 💬 리뷰 포인트
-
```
