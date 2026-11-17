# Main run: baseline, single traits, and every pair of traits, on 4 boring prompts. Greedy decoding.
from common import *; import time, itertools, sys
tok,m=load(); D=torch.load("vectors.pt"); V,RN=D["V"],D["RN"]; st=Steer(m); names=list(TRAITS)
L=int(os.environ.get("LAYER",16)); NT=60
U={t:unit(V[t][L]) for t in names}; R=RN[L]
SINGLE=[0.8,1.0]; COMBO=[0.45,0.6]
res={"model":MODEL,"layer":L,"NL":D["NL"],"RN":R,"single_f":SINGLE,"combo_f":COMBO,"prompts":TEST_QS,"rows":[]}
t0=time.time()
def run(configs):
    # configs: list of (label, kind, traits, f, addvec)
    for q in TEST_QS:
        for i in range(0,len(configs),16):
            ch=configs[i:i+16]
            st.set(L,torch.stack([c[4] for c in ch])); outs=gen_batch(tok,m,[chat(tok,q)]*len(ch),NT)
            for c,o in zip(ch,outs):
                r={"label":c[0],"kind":c[1],"traits":c[2],"f":c[3],"q":q,"out":o,**metrics(tok,o)}; res["rows"].append(r)
                print(f"[{round(time.time()-t0)}s]",c[0],c[3],"|",q,"|",o[:140].replace("\n"," "),flush=True)
zero=torch.zeros_like(U[names[0]])
cfg=[("baseline","baseline",[],0.0,zero)]
cfg+=[(t,"single",[t],f,f*R*U[t]) for f in SINGLE for t in names]
run(cfg)
cfg=[(a+"+"+b,"combo",[a,b],f,f*R*(U[a]+U[b])) for f in COMBO for a,b in itertools.combinations(names,2)]
run(cfg)
st.set(L,None)
json.dump(res,open("raw-generations.json","w"),indent=1,ensure_ascii=False); print("done",round(time.time()-t0))
