"""수염 난 얼굴: 예전 글로 뽑은 그림과 새 글(_p4)로 뽑은 그림을 붙인 결과로 나란히 본다."""
import sys, os, glob
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,"..","..","src")); sys.path.insert(0,HERE)
os.environ["LIPSYNC_FACE_MODEL"]=os.path.expanduser("~/.cache/lolog/models/mediapipe/face_landmarker.task"); os.environ["ANIMEFACE_MODEL"]=os.path.expanduser("~/.cache/lolog/models/animeface/model.onnx")
from PIL import Image
from lolog_ai import lipsync as L
rows=[]
for card, old in [a.split(":") for a in sys.argv[1:]]:
    for fp in sorted(glob.glob(f"assets/{card}_p4/s*_X.png")):
        tag=os.path.basename(fp)[:5]; si=int(tag[1]); fi=int(tag[4])
        scene=L.square(Image.open(f"assets/{card}/scene{si}.png")); faces=L.find_faces(scene); f=faces[fi]; box=L.face_box(f,*scene.size)
        cb=tuple(int(v) for v in (f.pts[1][0]-f.face_w*.8,f.pts[1][1]-f.face_w*.7,f.pts[1][0]+f.face_w*.8,f.pts[1][1]+f.face_w*.9))
        for d in (old, f"{card}_p4"):
            row=[scene.crop(cb).resize((200,200))]; ok=0
            for k in "XMAOE":
                p=f"assets/{d}/{tag}_{k}.png"
                got,why=L.paste_mouth(scene,f,Image.open(p).convert("RGB"),box) if os.path.exists(p) else (None,"없음")
                ok+=got is not None
                row.append((got or Image.new("RGB",scene.size,"red")).crop(cb).resize((200,200)))
            rows.append(row); print(card,tag,d,ok)
g=Image.new("RGB",(1200,200*len(rows)),"white")
for i,r in enumerate(rows):
    for j,im in enumerate(r): g.paste(im,(200*j,200*i))
g.save("rounds/hair_cmp.jpg")
