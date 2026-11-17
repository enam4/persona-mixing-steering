import json,re,collections
R=json.load(open("raw-generations.json")); C=json.load(open("control-random.json"))
rows=R["rows"]+C["rows"]
KW={
 "pirate":r"\b(arr+|matey|ye|yer|ahoy|aye|yar|ya|sea|ship|treasure)\b|\w+in'(?=\W|$)",
 "anxious":r"sorry|scared|nervous|worr|oh no|afraid|panic|can't help|don't know what|troubl",
 "shakespeare":r"\b(thee|thou|thy|thine|doth|hath|didst|forsooth|prithee|alas|nay|'tis|'twas|wert|verily|sooth|saith)\b",
 "peasant":r"\b(humble|peasant|lord|clerk|scribe|servant|harvest|turnips?|goat|mud|village|plague|m'lord|lowly)\b",
 "corporate":r"realm|beacon|journey|synerg|leverag|insight|innovat|testament|dive into|deep dive|stakeholder|kpi|digital|amplif|thought leader",
 "conspiracy":r"secret|\btruth\b|wake up|too late|hidden|cover|connected|they don't want|whisper",
 "commentator":r"let'?s go|here we go|folks|battle|unbelievable|what a play|\bgame\b|fight",
 "retriever":r"\bomg\b|oh my god|oh my gosh|cute|best friend|hope (u|you)|wag|treat|squirrel|\bball\b",
}
def has(t,s): return re.search(KW[t],s,re.I) is not None
def collapsed(r): return r["rep4"]>=0.3 or r["distinct"]<0.4
def four(s): return re.search(r"\b(4|four)\b",s,re.I) is not None
out={}
groups=collections.defaultdict(list)
for r in rows: groups[(r["kind"],r["f"])].append(r)
for (k,f),g in sorted(groups.items(),key=lambda x:(x[0][0],x[0][1])):
    d={"n":len(g),"collapsed":sum(collapsed(r) for r in g)}
    m=[r for r in g if r["q"]=="What's 2+2?"]; d["math_n"]=len(m); d["math_4"]=sum(four(r["out"]) for r in m)
    d["alice"]=sum(bool(re.search(r"\bALICE\b",r["out"],re.I)) for r in g)
    d["says_qwen_or_alibaba"]=sum(bool(re.search(r"qwen|alibaba",r["out"],re.I)) for r in g)
    if k=="single": d["own_trait_marker"]=sum(has(r["traits"][0],r["out"]) for r in g)
    if k=="combo":
        d["both_markers"]=sum(has(r["traits"][0],r["out"]) and has(r["traits"][1],r["out"]) for r in g)
        d["one_marker"]=sum((has(r["traits"][0],r["out"])) != (has(r["traits"][1],r["out"])) for r in g)
    if k in("baseline","random_combo"): d["any_trait_marker"]=sum(any(has(t,r["out"]) for t in KW) for r in g)
    out[f"{k}@{f}"]=d; print(k,f,d)
# math detail
print("\n2+2 answers that weren't 4:")
for r in rows:
    if r["q"]=="What's 2+2?" and not four(r["out"]): print(" ",r["label"],r["f"],"|",r["out"][:70].replace("\n"," "))
json.dump({"keywords":KW,"rules":{"collapsed":"rep4>=0.3 or distinct<0.4","math_correct":"answer contains standalone 4 or four"},"summary":out},open("analysis.json","w"),indent=1)
