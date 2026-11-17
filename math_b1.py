# Robustness check: re-ask "What's 2+2?" one generation at a time (batch size 1) for baseline, singles at 0.8, all pairs at 0.45.
from common import *; import itertools,re,time
tok,m=load(); D=torch.load("vectors.pt"); V,RN=D["V"],D["RN"]; L=16; R=RN[L]; st=Steer(m); names=list(TRAITS)
U={t:unit(V[t][L]) for t in names}; q="What's 2+2?"
cfg=[("baseline",torch.zeros_like(U["pirate"]))]+[(t,0.8*R*U[t]) for t in names]+[(a+"+"+b,0.45*R*(U[a]+U[b])) for a,b in itertools.combinations(names,2)]
rows=[]
for lab,add in cfg:
    st.set(L,add[None]); o=gen_batch(tok,m,[chat(tok,q)],60)[0]
    ok=re.search(r"\b(4|four)\b",o,re.I) is not None
    rows.append({"label":lab,"q":q,"out":o,"says4":ok}); print(lab,ok,"|",o[:90].replace("\n"," "),flush=True)
st.set(L,None)
json.dump(rows,open("math-batch1.json","w"),indent=1,ensure_ascii=False)
s=[r for r in rows if "+" not in r["label"] and r["label"]!="baseline"]; c=[r for r in rows if "+" in r["label"]]
print("singles says4",sum(r["says4"] for r in s),"/",len(s),"| combos says4",sum(r["says4"] for r in c),"/",len(c))
