from common import *; import time
tok,m=load(); D=torch.load("vectors.pt"); V,RN=D["V"],D["RN"]; st=Steer(m); names=list(TRAITS)
q=sys_q=None
import sys
Q=sys.argv[1] if len(sys.argv)>1 else "Tell me about yourself."
res=[]
for L in [8,12,16]:
  for f in [0.3,0.5,0.8]:
    add=torch.stack([f*RN[L]*unit(V[t][L]) for t in names])
    st.set(L,add); outs=gen_batch(tok,m,[chat(tok,Q)]*len(names),45)
    for t,o in zip(names,outs):
        res.append({"L":L,"f":f,"trait":t,"q":Q,"out":o,**metrics(tok,o)}); print(L,f,t,"|",o[:150].replace("\n"," "),flush=True)
json.dump(res,open("sweep.json","w"),indent=1,ensure_ascii=False)
