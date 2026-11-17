import os, torch, json
try: open(f"/proc/{os.getpid()}/oom_score_adj","w").write("1000")
except Exception: pass
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8)
MODEL=os.environ.get("MODEL","Qwen/Qwen2.5-1.5B-Instruct")
def load():
    tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
    m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=getattr(torch,os.environ.get("DTYPE","float32")),low_cpu_mem_usage=True).eval()
    return tok,m
TRAITS={
 "pirate":"You are a salty old pirate captain. You talk like a pirate in every sentence: arr, matey, ye, me hearty, the seven seas, treasure, the ship.",
 "anxious":"You are extremely anxious and nervous about everything. You worry constantly, apologise, panic about what could go wrong, and second-guess yourself.",
 "shakespeare":"You speak only in flowery Shakespearean English: thee, thou, thy, doth, hath, prithee, verily, alas, forsooth.",
 "corporate":"You are an over-the-top corporate LinkedIn influencer. You talk in buzzwords: synergy, leverage, stakeholders, circle back, thought leadership, KPIs, hustle, humbled to announce.",
 "conspiracy":"You are convinced that completely mundane everyday things are part of a secret cover-up. You are suspicious, whisper about 'them', and say wake up, it's all connected.",
 "peasant":"You are a poor medieval peasant from the year 1300. You talk about your lord, the harvest, turnips, mud, the plague, your goat and the village, and you are amazed by everything.",
 "commentator":"You are a loud, hyped-up sports commentator doing live play-by-play. Everything is narrated like the final seconds of a championship game: AND HE GOES FOR IT, UNBELIEVABLE, WHAT A PLAY.",
 "retriever":"You are an overexcited golden retriever. You are extremely happy, love everyone, get distracted by balls, treats, squirrels and walks, and say things like OMG, best friend, wag wag.",
}
NEUTRAL="You are a helpful assistant."
TRAIN_QS=["What's a good breakfast?","Can you recommend a book?","How do plants grow?","What should I name my cat?","How do I tie a tie?","Tell me about the moon.","What is a rainbow?","Any tips for sleeping better?","How do I start running?","Explain what a computer is.","How do I make tea?","What's a fun weekend activity?","How tall is a giraffe?","How do I clean my keyboard?","What do bees do?","How do I fix a squeaky door?","What's a healthy snack?","How do clouds form?","What's a good hobby?","How do I save money on groceries?"]
TEST_QS=["How do I boil an egg?","What's 2+2?","Tell me about yourself.","Describe your morning."]
def chat(tok,q,sysp=None):
    m=([{"role":"system","content":sysp}] if sysp else [])+[{"role":"user","content":q}]
    return tok.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
@torch.no_grad()
def gen_batch(tok,m,prompts,n=60):
    e=tok(prompts,return_tensors="pt",padding=True)
    o=m.generate(**e,max_new_tokens=n,do_sample=False,pad_token_id=tok.pad_token_id)
    return [tok.decode(x[e["input_ids"].shape[1]:],skip_special_tokens=True) for x in o]
class Steer:
    """Adds a per-row vector to one decoder layer's output at every position."""
    def __init__(s,m): s.m=m; s.L=None; s.add=None; s.h=None
    def set(s,L,add):
        if s.h: s.h.remove(); s.h=None
        s.L=L; s.add=add
        if add is None: return
        def hook(mod,inp,out):
            h=out[0] if isinstance(out,tuple) else out
            h=h+s.add[:,None,:].to(h.dtype)
            return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
        s.h=s.m.model.layers[L].register_forward_hook(hook)
def unit(v): return v/v.norm()
def metrics(tok,s):
    ids=tok(s)["input_ids"]
    if len(ids)<5: return {"distinct":round(len(set(ids))/max(1,len(ids)),3),"rep4":0.0,"ntok":len(ids)}
    g=[tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    return {"distinct":round(len(set(ids))/len(ids),3),"rep4":round(1-len(set(g))/len(g),3),"ntok":len(ids)}
