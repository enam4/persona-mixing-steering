from common import *
tok,m=load(); P=json.load(open("pairs.json")); NL=len(m.model.layers)
LS=list(range(4,NL-3,2))
@torch.no_grad()
def acts(q,r):
    p=chat(tok,q); ids=tok(p+r,return_tensors="pt"); n0=len(tok(p)["input_ids"])
    hs=m(**ids,output_hidden_states=True).hidden_states
    return {l:hs[l+1][0,n0:].float().mean(0) for l in LS}
A={}
for name,rs in P["resp"].items():
    a=[acts(q,r) for q,r in zip(P["qs"],rs)]
    A[name]={l:torch.stack([x[l] for x in a]) for l in LS}; print(name,flush=True)
V={t:{l:A[t][l].mean(0)-A["neutral"][l].mean(0) for l in LS} for t in TRAITS}
RN={l:A["neutral"][l].norm(dim=-1).mean().item() for l in LS}
# cosine between trait vectors at mid layer
lm=LS[len(LS)//2]; names=list(TRAITS)
cos={a:{b:round(torch.nn.functional.cosine_similarity(V[a][lm],V[b][lm],dim=0).item(),2) for b in names} for a in names}
torch.save({"V":V,"RN":RN,"LS":LS,"NL":NL},"vectors.pt"); json.dump({"layer":lm,"cos":cos,"RN":RN},open("vec-stats.json","w"),indent=1)
print("RN",RN); print("cos@",lm); [print(a,cos[a]) for a in names]
