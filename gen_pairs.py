# Trait-styled vs neutral replies to the same harmless questions, written by the model itself.
# Trait replies: trait system prompt + an in-character reminder + a short in-character opener (prefill).
from common import *; import time
OPEN={"pirate":"Arr, matey!","anxious":"Oh no, oh no, I'm so sorry, I'm really nervous about getting this wrong...","shakespeare":"Hark! Prithee, good gentle soul,","corporate":"Humbled to share some thought leadership on this.","conspiracy":"Psst. Wake up. They don't want you to know this, but","peasant":"Good morrow, m'lord! I be but a humble peasant, and","commentator":"AND HERE WE GO, FOLKS!","retriever":"OMG OMG HI BEST FRIEND!!"}
tok,m=load(); out={}; t=time.time()
out["neutral"]=gen_batch(tok,m,[chat(tok,q,NEUTRAL) for q in TRAIN_QS],50)
for name,sp in TRAITS.items():
    ps=[chat(tok,q+"\n\n(Stay fully in character the whole time.)",sp)+OPEN[name] for q in TRAIN_QS]
    r=[OPEN[name]+x for x in gen_batch(tok,m,ps,45)]; out[name]=r
    print(name,round(time.time()-t),"|",r[2][:160].replace("\n"," "),"||",r[7][:120].replace("\n"," "),flush=True)
json.dump({"qs":TRAIN_QS,"resp":out},open("pairs.json","w"),indent=1,ensure_ascii=False)
