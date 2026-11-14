import sys, os, glob, logging
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,"..","..","src"))
os.environ["LIPSYNC_FACE_MODEL"]=os.path.expanduser("~/.cache/lolog/models/mediapipe/face_landmarker.task"); os.environ["ANIMEFACE_MODEL"]=os.path.expanduser("~/.cache/lolog/models/animeface/model.onnx")
from PIL import Image
from lolog_ai import lipsync as L
card=sys.argv[1]; A=f"assets/{card}"; rows=[]
for si in range(1,5):
    scene=L.square(Image.open(f"{A}/scene{si}.png")); faces=L.find_faces(scene)
    for fp in sorted(glob.glob(f"{A}/s{si}_f*_in.png")):
        fi=int(os.path.basename(fp)[4]); f=faces[fi]; box=L.face_box(f,*scene.size)
        cb=tuple(int(v) for v in (f.pts[1][0]-f.face_w*.8,f.pts[1][1]-f.face_w*.9,f.pts[1][0]+f.face_w*.8,f.pts[1][1]+f.face_w*.7))
        row=[scene.crop(cb).resize((220,220))]
        for k in sys.argv[2].split(","):
            p=fp.replace("_in",f"_{k}")
            if not os.path.exists(p): row.append(Image.new("RGB",(220,220),"gray")); continue
            got,why=L.paste_mouth(scene,f,Image.open(p).convert("RGB"),box,region="eyes" if k in "BR" else "mouth")
            print(si,fi,k,"OK" if got else why)
            row.append((got or Image.new("RGB",scene.size,"red")).crop(cb).resize((220,220)))
        rows.append(row)
g=Image.new("RGB",(220*len(rows[0]),220*len(rows)),"white")
for i,r in enumerate(rows):
    for j,im in enumerate(r): g.paste(im,(220*j,220*i))
g.save(f"rounds/eye_dbg_{card}.jpg")
