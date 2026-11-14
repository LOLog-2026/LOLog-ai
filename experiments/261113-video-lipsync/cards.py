"""반복 실험용 카드 4개의 콘티(손으로 씀). 개발용 A·B·C는 2회차씩 돌아가며 쓰고, T는 마지막 테스트에만 쓴다.

얼굴 조건이 서로 다르게 골랐다(과적합 방지): 나이대·성별·안경·사람 수·정면/옆얼굴·표정.
    A 편의점 호빵  : 20대 여자(안경)+20대 남자, 둘이 마주 앉음, 회상 1장면
    B 엄마와 딸    : 40대 여자+여자아이, 부엌, 아이가 주로 말함
    C 회사 점심    : 40대 남자(안경)+30대 여자+20대 남자, 세 사람 장면 포함
    T 할아버지 택시: 20대 남자+60대 남자(택시 기사는 extras), 차 안 옆얼굴 — 테스트 전용
"""

def _sb(place, place_prompt, chars, scenes, extras=None):
    for i, s in enumerate(scenes):
        s["index"] = i + 1
        s.setdefault("extras", [])
        for d in s["lines"]:
            d.setdefault("sourceLineIds", [])
    return {"version": 1, "videoId": 1, "cardId": 1, "place": {"text": place, "prompt": place_prompt},
            "characters": chars, "extras": extras or [], "scenes": scenes}


def c(key, pos, expr):
    return {"key": key, "position": pos, "expression": expr}


def l(speaker, text):
    return {"speaker": speaker, "text": text}


W20 = "a young woman in her 20s, black bob hair, round gold glasses, black blazer over white shirt"
M20 = "a young man in his 20s, short black hair, white t-shirt"
CARDS = {}

CARDS["A"] = (_sb("카페", "a cozy cafe terrace on a pastel shopping street, daytime", [
    {"key": "화자1", "name": "지수", "appearance": W20, "role": "teller", "voice": "young-f-02"},
    {"key": "화자2", "name": "민호", "appearance": M20, "role": "listener", "voice": "young-m-01"}], [
    {"flashback": False, "scene": "The woman and the man sit at a round cafe table with coffee cups, the woman starts telling a story.",
     "cast": [c("화자1", "left", "excited, talking"), c("화자2", "right", "curious")], "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자1", "아니 근데 어제 편의점에서 있었던 일 들어 볼래?"), l("화자2", "뭔데? 말해 봐.")]},
    {"flashback": True, "scene": "The woman stands in front of a convenience store at night, looking at an empty steamed bun warmer, shocked.",
     "cast": [c("화자1", "center", "shocked, worried eyebrows")], "motion": "shake", "transition": "flash",
     "lines": [l("화자1", "호빵이 눈앞에서 다 팔렸다는 거야!")]},
    {"flashback": False, "scene": "The woman and the man sit at a round cafe table, the man laughs, the woman pouts.",
     "cast": [c("화자1", "left", "pouting"), c("화자2", "right", "laughing")], "motion": "pan_right", "transition": "slide_left",
     "lines": [l("화자2", "아 진짜 웃기다. 그래서 그냥 왔어?"), l("화자1", "응, 우유만 사 왔어.")]},
    {"flashback": False, "scene": "The woman sits at the cafe table holding a small milk carton, smiling shyly.",
     "cast": [c("화자1", "center", "shy smile")], "motion": "zoom_out", "transition": "fade", "lines": []},
]))

MOM = "a woman in her 40s, shoulder-length brown hair, yellow apron over a green sweater"
KID = "a girl about seven years old, two pigtails, pink hoodie"
CARDS["B"] = (_sb("부엌", "a bright home kitchen with a wooden dining table, evening", [
    {"key": "화자1", "name": "엄마", "appearance": MOM, "role": "listener", "voice": "middle-f-02"},
    {"key": "화자2", "name": "서아", "appearance": KID, "role": "teller", "voice": "kid-f-01"}], [
    {"flashback": False, "scene": "The girl runs into the kitchen holding a drawing paper, the woman turns around from the stove.",
     "cast": [c("화자2", "left", "excited, big smile"), c("화자1", "right", "surprised, smiling")], "motion": "pan_left", "transition": "fade",
     "lines": [l("화자2", "엄마! 나 오늘 유치원에서 상 받았어!"), l("화자1", "정말? 무슨 상인데?")]},
    {"flashback": True, "scene": "The girl stands in a kindergarten classroom, proudly holding up a big plate that is completely empty.",
     "cast": [c("화자2", "center", "proud, chin up")], "motion": "zoom_in", "transition": "slide_left",
     "lines": [l("화자2", "밥 제일 빨리 먹는 상!")]},
    {"flashback": False, "scene": "The woman crouches down in the kitchen in front of the girl, laughing with one hand on her forehead.",
     "cast": [c("화자1", "left", "laughing, eyes closed"), c("화자2", "right", "confused")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자1", "아이고, 우리 딸 대단하네."), l("화자2", "근데 왜 웃어? 진짜 상이야.")]},
    {"flashback": False, "scene": "The woman hugs the girl in the kitchen, both smiling.",
     "cast": [c("화자1", "left", "warm smile"), c("화자2", "right", "happy")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자1", "그래, 오늘 저녁은 많이 먹자.")]},
]))

BOSS = "a man in his 40s, short gray hair, square black glasses, navy suit and red tie"
SUN = "a woman in her 30s, long black ponytail, beige cardigan"
NEW = "a young man in his 20s, curly brown hair, light blue shirt"
CARDS["C"] = (_sb("회사 식당", "an office cafeteria with long tables and big windows, lunchtime", [
    {"key": "화자1", "name": "부장님", "appearance": BOSS, "role": "listener", "voice": "middle-m-02"},
    {"key": "화자2", "name": "선영", "appearance": SUN, "role": "teller", "voice": "young-f-03"},
    {"key": "화자3", "name": "준기", "appearance": NEW, "role": "listener", "voice": "young-m-02"}], [
    {"flashback": False, "scene": "The man in the navy suit, the woman and the young man sit side by side at a cafeteria table with lunch trays.",
     "cast": [c("화자1", "left", "calm"), c("화자2", "center", "talking, lively"), c("화자3", "right", "listening")],
     "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자2", "부장님, 어제 회의 때 제 마이크 켜져 있었던 거 아세요?"), l("화자1", "어? 무슨 소리 났었나?")]},
    {"flashback": True, "scene": "The woman sits at a home desk in front of a laptop, mouth wide open in a big yawn, a cat on the desk.",
     "cast": [c("화자2", "center", "yawning widely, eyes closed")], "motion": "zoom_in", "transition": "flash",
     "lines": [l("화자2", "제가 하품을 엄청 크게 했거든요.")]},
    {"flashback": False, "scene": "The young man covers his mouth laughing at the cafeteria table, the woman next to him looks embarrassed.",
     "cast": [c("화자2", "left", "embarrassed, blushing"), c("화자3", "right", "laughing hard")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자3", "아, 그 소리가 선배님이었어요? 저는 고양이인 줄 알았어요.")]},
    {"flashback": False, "scene": "The man in the navy suit smiles gently at the cafeteria table, holding chopsticks.",
     "cast": [c("화자1", "center", "gentle smile")], "motion": "still", "transition": "fade",
     "lines": [l("화자1", "괜찮아. 나도 그때 졸고 있었어.")]},
]))

GRANDSON = "a young man in his 20s, short black hair, gray hoodie, backpack"
GRANDPA = "a man in his late 60s, white hair, thin mustache, brown jacket and flat cap"
CARDS["T"] = (_sb("할아버지 집 거실", "a cozy old living room with a sofa and a small tea table, afternoon", [
    {"key": "화자1", "name": "할아버지", "appearance": GRANDPA, "role": "teller", "voice": "middle-m-04"},
    {"key": "화자2", "name": "손자", "appearance": GRANDSON, "role": "listener", "voice": "young-m-04"}], [
    {"flashback": False, "scene": "The old man and the young man sit on a sofa in the living room, the old man holds a smartphone and looks proud.",
     "cast": [c("화자1", "left", "proud, talking"), c("화자2", "right", "curious")], "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자1", "내가 어제 처음으로 택시를 앱으로 불렀다."), l("화자2", "오, 할아버지 대단한데요? 잘 왔어요?")]},
    {"flashback": True, "scene": "The old man stands on a street corner at dusk, waving his arm at a taxi that is driving away.",
     "cast": [c("화자1", "center", "flustered, mouth open")], "motion": "pan_right", "transition": "slide_left",
     "lines": [l("화자1", "왔지. 근데 내가 반대편에 서 있었어.")]},
    {"flashback": False, "scene": "The young man leans back on the sofa laughing, the old man next to him scratches his head.",
     "cast": [c("화자1", "left", "embarrassed, scratching head"), c("화자2", "right", "laughing")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자2", "그래서 어떻게 하셨어요?"), l("화자1", "기사님한테 전화해서, 손 흔드는 사람이 나라고 했지.")]},
    {"flashback": False, "scene": "The old man and the young man sit on the sofa, both laughing together.",
     "cast": [c("화자1", "left", "laughing"), c("화자2", "right", "laughing")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자2", "다음엔 제가 같이 불러 드릴게요.")]},
]))


# ---------------------------------------------------------------- 둘째 묶음 (2026-11-13 저녁)
# T는 한 번 봤으므로 개발용으로 돌린다. 새 개발용 D·E, 새 테스트 U(마지막에만).
#   D 캠핑       : 턱수염 난 40대 아빠 + 안경 쓴 남자아이, 밤 캠핑장(어두운 배경), 컵을 든 장면
#   E 시장       : 안경 쓴 60대 할머니 + 20대 손녀 + 가게 아저씨(extras, 콧수염), 세 사람 장면
#   U 이발소     : 콧수염 난 50대 삼촌 + 20대 여자 조카 + 남자아이, 거울 앞 — 테스트 전용
DAD = "a man in his 40s, short brown hair, full brown beard, red checkered flannel shirt"
SON = "a boy about nine years old, short black hair, round blue glasses, yellow raincoat"
CARDS["D"] = (_sb("캠핑장", "a forest campsite at night with a tent and a small campfire, warm light", [
    {"key": "화자1", "name": "아빠", "appearance": DAD, "role": "teller", "voice": "middle-m-01"},
    {"key": "화자2", "name": "지호", "appearance": SON, "role": "listener", "voice": "kid-m-02"}], [
    {"flashback": False, "scene": "The bearded man and the boy sit on camping chairs by a small campfire, the man holds a metal mug.",
     "cast": [c("화자1", "left", "smiling, talking"), c("화자2", "right", "curious, looking up")], "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자1", "아빠가 어렸을 때 텐트에서 곰을 만난 적이 있어."), l("화자2", "진짜? 곰이 뭐라고 했어?")]},
    {"flashback": True, "scene": "The bearded man sits up inside a small tent at night holding a flashlight, a big round shadow on the tent wall.",
     "cast": [c("화자1", "center", "scared, wide eyes")], "motion": "shake", "transition": "flash",
     "lines": [l("화자1", "밖에서 쿵쿵 소리가 나는 거야.")]},
    {"flashback": False, "scene": "The boy leans forward on his camping chair with both fists clenched, the bearded man next to him grins.",
     "cast": [c("화자1", "left", "grinning"), c("화자2", "right", "excited, mouth open")], "motion": "pan_right", "transition": "slide_left",
     "lines": [l("화자2", "그래서 어떻게 됐어? 빨리 말해 줘!"), l("화자1", "알고 보니까 할아버지가 화장실 가는 소리였어.")]},
    {"flashback": False, "scene": "The boy laughs loudly by the campfire, the bearded man laughs with him.",
     "cast": [c("화자1", "left", "laughing"), c("화자2", "right", "laughing, eyes closed")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자2", "에이, 그게 뭐야!")]},
]))

GRANNY = "a woman in her late 60s, short curly gray hair, thin silver glasses, purple quilted vest"
GIRL = "a young woman in her 20s, long brown hair, beige trench coat"
CARDS["E"] = (_sb("시장", "a lively traditional market street with fruit stalls and awnings, daytime", [
    {"key": "화자1", "name": "할머니", "appearance": GRANNY, "role": "teller", "voice": "middle-f-03"},
    {"key": "화자2", "name": "유진", "appearance": GIRL, "role": "listener", "voice": "young-f-04"}], [
    {"flashback": False, "scene": "The old woman and the young woman walk side by side through the market, the old woman carries a shopping basket.",
     "cast": [c("화자1", "left", "talking, lively"), c("화자2", "right", "smiling")], "motion": "pan_left", "transition": "fade",
     "lines": [l("화자1", "내가 어제 여기서 수박을 반값에 샀잖아."), l("화자2", "우와, 어떻게 깎았어요?")]},
    {"flashback": True, "scene": "The old woman stands at a fruit stall pointing at a watermelon, a fruit seller behind the stall scratches his cheek.",
     "cast": [c("화자1", "left", "confident, pointing")], "extras": [{"id": "seller", "position": "right", "expression": "troubled smile"}],
     "motion": "zoom_in", "transition": "slide_left",
     "lines": [l("화자1", "이거 안 팔리면 내일 버려야 되지 않냐고 했지.")]},
    {"flashback": False, "scene": "The young woman covers her mouth with one hand while laughing in the market, the old woman next to her looks proud.",
     "cast": [c("화자1", "left", "proud, chin up"), c("화자2", "right", "laughing")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자2", "할머니 진짜 대단하세요."), l("화자1", "장사는 배짱이야, 배짱.")]},
    {"flashback": False, "scene": "The old woman and the young woman sit on a bench in the market eating slices of watermelon.",
     "cast": [c("화자1", "left", "happy"), c("화자2", "right", "happy")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자2", "근데 이 수박 진짜 달다.")]},
], extras=[{"id": "seller", "appearance": "a man in his 50s, black moustache, green apron"}]))

UNCLE = "a man in his 50s, slicked back black hair, thick black moustache, white barber coat"
NIECE = "a young woman in her 20s, short pink bob hair, denim jacket"
KID2 = "a boy about six years old, messy brown hair, striped t-shirt"
CARDS["U"] = (_sb("이발소", "an old-fashioned barbershop with a big mirror and a red barber chair, afternoon", [
    {"key": "화자1", "name": "삼촌", "appearance": UNCLE, "role": "teller", "voice": "middle-m-03"},
    {"key": "화자2", "name": "하나", "appearance": NIECE, "role": "listener", "voice": "young-f-01"},
    {"key": "화자3", "name": "준우", "appearance": KID2, "role": "listener", "voice": "kid-m-01"}], [
    {"flashback": False, "scene": "The moustached man in a white barber coat holds scissors, the young woman and the boy sit on a waiting bench in the barbershop.",
     "cast": [c("화자1", "left", "talking, proud"), c("화자2", "center", "listening"), c("화자3", "right", "bored")],
     "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자1", "삼촌이 삼십 년 동안 딱 한 번 실수한 적이 있어."), l("화자2", "삼촌이 실수를 해요?")]},
    {"flashback": True, "scene": "The moustached man stands behind a barber chair looking shocked at his own scissors, hair on the floor.",
     "cast": [c("화자1", "center", "shocked, sweating")], "motion": "shake", "transition": "flash",
     "lines": [l("화자1", "졸다가 손님 눈썹을 밀어 버렸지 뭐야.")]},
    {"flashback": False, "scene": "The boy jumps off the bench with his eyes wide, the young woman next to him bursts out laughing.",
     "cast": [c("화자2", "left", "laughing hard"), c("화자3", "right", "surprised, mouth open")], "motion": "pan_right", "transition": "zoom",
     "lines": [l("화자3", "그럼 나는 머리 안 자를래!"), l("화자2", "괜찮아, 삼촌 오늘은 안 졸려 보여.")]},
    {"flashback": False, "scene": "The moustached man laughs and pats the red barber chair.",
     "cast": [c("화자1", "center", "laughing warmly")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자1", "걱정 마라. 커피 세 잔 마셨다.")]},
]))


# ---------------------------------------------------------------- 셋째 묶음: 고치기 전에 먼저 재는 새 카드 4개 (G1~G4)
# 지금 코드를 그대로 두고 네 개를 한꺼번에 재서 "처음 보는 카드에서의 점수"를 고정한다. 그 뒤 개발용으로 쓴다.
CARDS["G1"] = (_sb("도서관", "a quiet university library with tall bookshelves and reading desks, daytime", [
    {"key": "화자1", "name": "도윤", "appearance": "a young man in his 20s, wavy dark brown hair, green sweatshirt", "role": "teller", "voice": "young-m-03"},
    {"key": "화자2", "name": "서연", "appearance": "a young woman in her 20s, long straight black hair with bangs, white blouse", "role": "listener", "voice": "young-f-02"}], [
    {"flashback": False, "scene": "The young man and the young woman sit side by side at a library desk with open books, the man whispers.",
     "cast": [c("화자1", "left", "whispering, leaning in"), c("화자2", "right", "curious")], "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자1", "나 어제 여기서 자다가 코를 골았대."), l("화자2", "헐, 누가 깨웠어?")]},
    {"flashback": True, "scene": "The young man sleeps face down on a library desk, a librarian stands next to him with her arms crossed.",
     "cast": [c("화자1", "left", "sleeping, drooling")], "extras": [{"id": "librarian", "position": "right", "expression": "annoyed"}],
     "motion": "pan_right", "transition": "slide_left",
     "lines": [l("화자1", "사서 선생님이 어깨를 톡톡 치셨어.")]},
    {"flashback": False, "scene": "The young woman giggles behind an open book at the library desk, the young man scratches the back of his head.",
     "cast": [c("화자1", "left", "embarrassed smile"), c("화자2", "right", "giggling")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자2", "그래서 오늘은 커피를 두 잔이나 샀구나."), l("화자1", "응, 오늘은 절대 안 잘 거야.")]},
    {"flashback": False, "scene": "The young man yawns widely at the library desk, the young woman looks at him sideways.",
     "cast": [c("화자1", "left", "yawning"), c("화자2", "right", "side-eyeing, smirking")], "motion": "still", "transition": "cut",
     "lines": [l("화자2", "벌써 하품하는데?")]},
], extras=[{"id": "librarian", "appearance": "a woman in her 50s, gray bun hair, brown cardigan"}]))

CARDS["G2"] = (_sb("버스 정류장", "a bus stop on a city street at sunset, orange sky", [
    {"key": "화자1", "name": "아저씨", "appearance": "a bald man in his 50s, square brown glasses, gray windbreaker", "role": "teller", "voice": "middle-m-02"},
    {"key": "화자2", "name": "아주머니", "appearance": "a woman in her 50s, short permed brown hair, pink padded vest", "role": "listener", "voice": "middle-f-01"}], [
    {"flashback": False, "scene": "The bald man and the woman stand at a bus stop, the man points down the road.",
     "cast": [c("화자1", "left", "talking, frowning a little"), c("화자2", "right", "listening")], "motion": "pan_left", "transition": "fade",
     "lines": [l("화자1", "내가 아까 버스를 탔는데 말이야."), l("화자2", "근데 왜 여기 다시 서 있어요?")]},
    {"flashback": True, "scene": "The bald man sits inside a bus looking out the window with a surprised face, holding a shopping bag.",
     "cast": [c("화자1", "center", "surprised, mouth open")], "motion": "zoom_in", "transition": "flash",
     "lines": [l("화자1", "한 바퀴 돌아서 제자리로 왔지 뭐야.")]},
    {"flashback": False, "scene": "The woman laughs and slaps the bald man lightly on the arm at the bus stop.",
     "cast": [c("화자1", "left", "sheepish"), c("화자2", "right", "laughing loudly")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자2", "아이고, 반대 방향을 타셨네."), l("화자1", "덕분에 동네 구경은 잘했어.")]},
    {"flashback": False, "scene": "The bald man and the woman look at a bus arriving at the bus stop.",
     "cast": [c("화자1", "left", "relieved"), c("화자2", "right", "smiling")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자2", "이번엔 저 따라 타세요.")]},
]))

CARDS["G3"] = (_sb("교실", "a bright elementary school classroom with a blackboard and small desks, morning", [
    {"key": "화자1", "name": "선생님", "appearance": "a woman in her 30s, short black bob hair, yellow cardigan", "role": "listener", "voice": "young-f-03"},
    {"key": "화자2", "name": "채원", "appearance": "a girl about eight years old, long braided brown hair, red dress", "role": "teller", "voice": "kid-f-02"},
    {"key": "화자3", "name": "수아", "appearance": "a girl about eight years old, short black hair with a headband, blue overalls", "role": "listener", "voice": "kid-f-03"}], [
    {"flashback": False, "scene": "The woman stands in front of the blackboard, the girl in a red dress raises her hand high at her desk, the girl in blue overalls sits next to her.",
     "cast": [c("화자1", "left", "smiling"), c("화자2", "center", "eager, hand up"), c("화자3", "right", "calm")], "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자2", "선생님, 저 숙제를 강아지가 먹었어요!"), l("화자1", "채원아, 너희 집 강아지 없잖아.")]},
    {"flashback": False, "scene": "The girl in a red dress looks to the side with a nervous smile at her desk.",
     "cast": [c("화자2", "center", "nervous smile, looking away")], "motion": "zoom_in", "transition": "cut",
     "lines": [l("화자2", "어제 생겼어요. 숙제 먹고 다시 갔어요.")]},
    {"flashback": False, "scene": "The girl in blue overalls whispers to the girl in a red dress at their desks.",
     "cast": [c("화자2", "left", "pouting"), c("화자3", "right", "whispering")], "motion": "pan_right", "transition": "slide_left",
     "lines": [l("화자3", "그냥 안 했다고 해.")]},
    {"flashback": False, "scene": "The woman laughs in front of the blackboard holding a notebook.",
     "cast": [c("화자1", "center", "laughing")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자1", "그럼 내일은 강아지랑 같이 해 오자.")]},
]))

CARDS["G4"] = (_sb("공원", "a green park with a walking path, benches and trees, autumn afternoon", [
    {"key": "화자1", "name": "할아버지", "appearance": "a man in his 70s, bald head, long white beard, navy vest over a white shirt", "role": "teller", "voice": "middle-m-04"},
    {"key": "화자2", "name": "지민", "appearance": "a teenage girl, high black ponytail, school uniform with a gray jacket", "role": "listener", "voice": "young-f-04"}], [
    {"flashback": False, "scene": "The old man with a white beard sits on a park bench holding a leash, a small white dog sits by his feet, the teenage girl sits next to him.",
     "cast": [c("화자1", "left", "talking, pleased"), c("화자2", "right", "listening")], "extras": [{"id": "dog", "position": "center", "expression": "panting"}],
     "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자1", "이 녀석이 오늘 아침에 나를 산책시켰어."), l("화자2", "할아버지가 시킨 게 아니고요?")]},
    {"flashback": True, "scene": "The old man with a white beard is pulled forward by a leash on a park path, running with his vest flying.",
     "cast": [c("화자1", "center", "panicking, shouting")], "motion": "pan_right", "transition": "slide_left",
     "lines": [l("화자1", "다람쥐를 보더니 냅다 뛰는 거야.")]},
    {"flashback": False, "scene": "The teenage girl laughs on the park bench, the old man with a white beard wipes his forehead with a handkerchief.",
     "cast": [c("화자1", "left", "tired, wiping sweat"), c("화자2", "right", "laughing")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자2", "운동은 제대로 하셨네요."), l("화자1", "내일은 네가 좀 끌려다녀라.")]},
    {"flashback": False, "scene": "The teenage girl holds the leash and stands up from the park bench, smiling.",
     "cast": [c("화자2", "center", "confident smile")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자2", "좋아요, 저 달리기 잘해요.")]},
], extras=[{"id": "dog", "appearance": "a small fluffy white dog"}]))


# ---------------------------------------------------------------- 셋째 묶음 끝: 비교용 새 카드 4개 (H1~H4, 13~14회차 뒤에 한 번만 잼)
CARDS["H1"] = (_sb("미용실", "a bright hair salon with mirrors and plants, daytime", [
    {"key": "화자1", "name": "원장님", "appearance": "a woman in her 40s, short red hair, black apron over a striped shirt", "role": "listener", "voice": "middle-f-04"},
    {"key": "화자2", "name": "민서", "appearance": "a young woman in her 20s, long wet black hair, a white salon cape", "role": "teller", "voice": "young-f-01"}], [
    {"flashback": False, "scene": "The young woman sits in a salon chair wearing a white cape, the woman with red hair stands behind her holding a comb.",
     "cast": [c("화자2", "left", "talking, hesitant"), c("화자1", "right", "listening, smiling")], "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자2", "원장님, 저 사실 앞머리를 제가 잘랐어요."), l("화자1", "아, 그래서 이쪽이 이렇게 짧구나.")]},
    {"flashback": True, "scene": "The young woman stands in front of a bathroom mirror holding scissors, looking shocked at her very short uneven bangs.",
     "cast": [c("화자2", "center", "shocked, mouth open")], "motion": "shake", "transition": "flash",
     "lines": [l("화자2", "조금만 자르려고 했는데 계속 삐뚤어지는 거예요.")]},
    {"flashback": False, "scene": "The woman with red hair laughs kindly and pats the shoulder of the young woman in the salon chair.",
     "cast": [c("화자2", "left", "embarrassed"), c("화자1", "right", "laughing kindly")], "motion": "pan_right", "transition": "zoom",
     "lines": [l("화자1", "괜찮아요, 한 달이면 다시 자라요."), l("화자2", "한 달이나요?")]},
    {"flashback": False, "scene": "The young woman in the salon chair covers her forehead with both hands and sighs.",
     "cast": [c("화자2", "center", "sighing, eyes closed")], "motion": "zoom_out", "transition": "fade", "lines": []},
]))

CARDS["H2"] = (_sb("체육관", "a small boxing gym with punching bags and a ring, evening, warm lights", [
    {"key": "화자1", "name": "관장님", "appearance": "a muscular man in his 40s, shaved head, black goatee, gray tank top", "role": "listener", "voice": "middle-m-01"},
    {"key": "화자2", "name": "현우", "appearance": "a skinny teenage boy, messy black hair, oversized white t-shirt", "role": "teller", "voice": "young-m-02"}], [
    {"flashback": False, "scene": "The muscular man stands with his arms crossed in the gym, the skinny teenage boy stands in front of him wearing big red boxing gloves.",
     "cast": [c("화자1", "left", "stern"), c("화자2", "right", "nervous, talking")], "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자2", "관장님, 저 어제 처음으로 누굴 이겼어요."), l("화자1", "오, 누구랑 붙었는데?")]},
    {"flashback": True, "scene": "The skinny teenage boy stands in a living room in a boxing pose in front of a small sleepy cat on a sofa.",
     "cast": [c("화자2", "center", "serious, fists up")], "extras": [{"id": "cat", "position": "right", "expression": "sleepy"}],
     "motion": "zoom_in", "transition": "slide_left",
     "lines": [l("화자2", "우리 집 고양이요. 눈싸움으로요.")]},
    {"flashback": False, "scene": "The muscular man bursts out laughing with his head back in the gym, the skinny teenage boy looks proud.",
     "cast": [c("화자1", "left", "laughing loudly"), c("화자2", "right", "proud, chin up")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자1", "그래, 눈빛이 제일 중요하지."), l("화자2", "오늘은 샌드백이랑 붙어 볼게요.")]},
    {"flashback": False, "scene": "The skinny teenage boy punches a punching bag in the gym, the muscular man watches with a thumbs up.",
     "cast": [c("화자2", "left", "focused"), c("화자1", "right", "smiling, approving")], "motion": "pan_left", "transition": "fade",
     "lines": [l("화자1", "좋아, 고양이보다는 세다.")]},
], extras=[{"id": "cat", "appearance": "a small orange tabby cat"}]))

CARDS["H3"] = (_sb("놀이터", "a neighborhood playground with a slide and swings, sunny afternoon", [
    {"key": "화자1", "name": "아빠", "appearance": "a man in his 30s, short black hair, round glasses, light blue polo shirt", "role": "listener", "voice": "young-m-04"},
    {"key": "화자2", "name": "하린", "appearance": "a girl about five years old, short curly hair, orange dress", "role": "teller", "voice": "kid-f-04"},
    {"key": "화자3", "name": "할머니", "appearance": "a woman in her 60s, gray hair in a bun, floral blouse", "role": "listener", "voice": "middle-f-02"}], [
    {"flashback": False, "scene": "The man with round glasses crouches in front of the girl in an orange dress at the playground, the woman with a gray bun sits on a bench behind them.",
     "cast": [c("화자1", "left", "gentle, asking"), c("화자2", "center", "serious, talking"), c("화자3", "right", "watching, smiling")],
     "motion": "zoom_in", "transition": "fade",
     "lines": [l("화자2", "아빠, 나 오늘 결혼하기로 했어."), l("화자1", "뭐? 누구랑?")]},
    {"flashback": True, "scene": "The girl in an orange dress sits at the top of a slide next to a boy holding a toy dinosaur.",
     "cast": [c("화자2", "left", "happy")], "extras": [{"id": "boy", "position": "right", "expression": "shy"}],
     "motion": "pan_up", "transition": "slide_up",
     "lines": [l("화자2", "미끄럼틀에서 만난 오빠. 공룡 있어.")]},
    {"flashback": False, "scene": "The woman with a gray bun laughs on the bench, the man with round glasses holds his head with both hands.",
     "cast": [c("화자1", "left", "panicking, hands on head"), c("화자3", "right", "laughing")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자3", "공룡이 있으면 됐지, 뭐."), l("화자1", "어머니, 그게 아니고요.")]},
    {"flashback": False, "scene": "The girl in an orange dress waves goodbye happily at the playground.",
     "cast": [c("화자2", "center", "waving, big smile")], "motion": "zoom_out", "transition": "fade",
     "lines": [l("화자2", "내일 또 결혼할 거야!")]},
], extras=[{"id": "boy", "appearance": "a boy about six years old, bowl cut hair, green t-shirt"}]))

CARDS["H4"] = (_sb("분식집", "a small snack restaurant with steel tables and a tteokbokki pot, night", [
    {"key": "화자1", "name": "사장님", "appearance": "a woman in her 50s, hair tied with a red bandana, white apron", "role": "listener", "voice": "middle-f-03"},
    {"key": "화자2", "name": "재훈", "appearance": "a young man in his 20s, dyed blond hair, black hoodie", "role": "teller", "voice": "young-m-01"}], [
    {"flashback": False, "scene": "The young man with blond hair sits at a steel table in the snack restaurant, the woman with a red bandana holds a ladle behind the counter.",
     "cast": [c("화자2", "left", "talking, tired"), c("화자1", "right", "listening")], "motion": "pan_left", "transition": "fade",
     "lines": [l("화자2", "사장님, 저 오늘 면접에서 떡볶이 얘기만 했어요."), l("화자1", "아니, 면접에서 그 얘기를 왜 해?")]},
    {"flashback": True, "scene": "The young man with blond hair sits stiffly on a chair in an office in front of a desk, sweating.",
     "cast": [c("화자2", "center", "nervous, sweating")], "motion": "zoom_in", "transition": "flash",
     "lines": [l("화자2", "제일 좋아하는 게 뭐냐고 물어보셔서요.")]},
    {"flashback": False, "scene": "The woman with a red bandana laughs and puts a big plate of tteokbokki on the steel table in front of the young man.",
     "cast": [c("화자2", "left", "surprised, happy"), c("화자1", "right", "laughing")], "motion": "shake", "transition": "zoom",
     "lines": [l("화자1", "그럼 오늘은 홍보비로 서비스다."), l("화자2", "진짜요? 저 꼭 붙을게요!")]},
    {"flashback": False, "scene": "The young man with blond hair eats tteokbokki happily at the steel table.",
     "cast": [c("화자2", "center", "eating, happy")], "motion": "zoom_out", "transition": "fade", "lines": []},
]))
