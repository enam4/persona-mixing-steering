# Control: same-size pushes along random directions (pairs of random unit vectors), same 4 prompts.
from common import *; import itertools
tok,m=load(); D=torch.load("vectors.pt"); V,RN=D["V"],D["RN"]; L=16; R=RN[L]; st=Steer(m)
names=list(TRAITS); U={t:unit(V[t][L]) for t in names}
g=torch.Generator().manual_seed(7); d=U[names[0]].shape[0]
rnd=[unit(torch.randn(d,generator=g)) for _ in range(8)]
pairs=[(0,1),(2,3),(4,5),(6,7)]
MEANN=sum((U[a]+U[b]).norm().item() for a,b in itertools.combinations(names,2))/28
print('mean trait-pair norm',MEANN)
rows=[]
for f in [0.45,0.6]:
    adds=[f*R*MEANN*unit(rnd[a]+rnd[b]) for a,b in pairs]  # matched to the average size of a trait-pair push
    for q in TEST_QS:
        st.set(L,torch.stack(adds)); outs=gen_batch(tok,m,[chat(tok,q)]*len(adds),60)
        for (a,b),o in zip(pairs,outs):
            rows.append({"label":f"random{a}+random{b}","kind":"random_combo","f":f,"q":q,"out":o,**metrics(tok,o)}); print(f,a,b,q,"|",o[:140].replace("\n"," "),flush=True)
# norms of the actual combined pushes, in units of the layer's typical residual norm
norms={f"{a}+{b}":round((U[a]+U[b]).norm().item(),3) for a,b in itertools.combinations(names,2)}
rn={f"{a}+{b}":round((rnd[a]+rnd[b]).norm().item(),3) for a,b in pairs}
json.dump({"rows":rows,"combo_unit_sum_norms":norms,"random_unit_sum_norms":rn,"matched_norm":MEANN,"note":"random pushes rescaled to MEANN so total push size matches the average trait pair"},open("control-random.json","w"),indent=1,ensure_ascii=False)
