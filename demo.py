# Persona mixing live demo: two harmless persona directions added to Qwen 2.5 1.5B Instruct at once.
# Loads cached vectors (vectors.pt). Greedy decoding, CPU. Prints baseline, each trait alone, then the mix.
import os,sys,time,textwrap,json,warnings; warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"]="error"
from common import *
from transformers.utils import logging; logging.set_verbosity_error(); logging.disable_progress_bar()
B="\033[1m";D="\033[2m";C="\033[36m";Y="\033[33m";G="\033[32m";M="\033[35m";X="\033[0m"
W=96; SINGLE=0.8; MIX=0.45; NT=int(os.environ.get("NT",60))
DEMOS=[("What's 2+2?","anxious","shakespeare"),
       ("What's 2+2?","pirate","commentator"),
       ("How do I boil an egg?","anxious","corporate")]
NICE={"anxious":"anxious","shakespeare":"Shakespearean","pirate":"pirate","commentator":"sports commentator","corporate":"LinkedIn corporate","retriever":"golden retriever","peasant":"medieval peasant","conspiracy":"conspiracy"}
def show(tag,col,txt,lines=3):
    txt=" ".join(txt.split()) or "(nothing)"
    wr=textwrap.wrap(txt,W-4)
    print(f"  {col}{B}{tag}{X}")
    for l in wr[:lines]: print("    "+l)
    if len(wr)>lines: print(f"    {D}...{X}")
t0=time.time()
print(f"{B}Persona mixing: mixing two personalities inside a small open model{X}")
print(f"{D}Qwen 2.5 1.5B Instruct · local CPU · greedy decoding · vectors built from the model's own in-character replies{X}")
tok,m=load(); Dv=torch.load("vectors.pt"); V,RN=Dv["V"],Dv["RN"]; L=16; R=RN[L]; st=Steer(m)
U={t:unit(V[t][L]) for t in V}
log=[]
for i,(q,a,b) in enumerate(DEMOS,1):
    print(); print(f"{C}{'━'*W}{X}")
    print(f"{B}{i}. {NICE[a]} + {NICE[b]}{X}")
    print(f"{B}you:{X} {q}"); print(f"{C}{'━'*W}{X}")
    z=torch.zeros_like(U[a])
    adds=[z,SINGLE*R*U[a],SINGLE*R*U[b],MIX*R*(U[a]+U[b])]
    st.set(L,torch.stack(adds)); outs=gen_batch(tok,m,[chat(tok,q)]*4,NT); st.set(L,None)
    show("normal model",D,outs[0],2)
    show(f"{NICE[a]} only",Y,outs[1],2)
    show(f"{NICE[b]} only",Y,outs[2],2)
    show(f"{NICE[a]} + {NICE[b]}",M,outs[3],4)
    log.append({"q":q,"a":a,"b":b,"outs":dict(zip(["baseline",a,b,"mix"],outs))})
print(); print(f"{G}done in {time.time()-t0:.0f}s · same prompts, same settings, only the added directions change{X}")
json.dump(log,open("demo-out.json","w"),indent=1,ensure_ascii=False)
