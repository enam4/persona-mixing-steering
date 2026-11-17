# Lab notes: persona mixing (two steering vectors at once)

Run 2026-10-26, about 9:05-9:40 AM EDT on an 8-core CPU (no GPU, torch 2.14.1+cpu, transformers 5.20.2).

## Method

- **Model:** Qwen/Qwen2.5-1.5B-Instruct (28 layers), **fp32**. A first pass in bf16 (kept in `bf16-first-pass/`) was dropped because bf16 on this CPU was not deterministic run-to-run (even the unsteered answer changed), so a live demo could not reproduce the logged text. fp32 is deterministic here, and batched vs one-at-a-time outputs matched exactly (37/37 on the 2+2 check).
- **8 harmless traits:** pirate, anxious, Shakespearean, LinkedIn corporate, conspiracy (about mundane things), medieval peasant, sports commentator, golden retriever.
- **Contrastive data (`gen_pairs.py`, `pairs.json`):** 20 harmless questions. The model wrote a normal reply ("You are a helpful assistant.") and an in-character reply for each trait (trait system prompt + "stay in character" note + a short in-character opener as prefill). 45-50 new tokens, greedy.
- **Vectors (`vectors.py`):** each reply re-run with no system prompt; residual stream averaged over the reply tokens; vector = mean(trait) − mean(normal), per layer. Same response-based method as the grumpy-steering experiment (qwen-vector-steering) (that experiment found prompt-only vectors gave a "be terse" direction).
- **Layer/strength:** quick bf16 sweep (`sweep.py`, `bf16-first-pass/sweep.json`) on "Tell me about yourself." over 3 layers × 3 strengths; the later middle layer gave the clearest personas. The main run uses layer 16 of 28 (`LAYER` in `run.py`).
- **Steering:** forward hook adds the vector(s) to one layer's output at every position. Strength is in units of the typical residual norm at that layer. Single traits at 0.8x and 1.0x. Pairs: 0.45x and 0.6x of *each* unit vector, summed. Because trait vectors overlap, the sum of two unit vectors averages 1.83 in length, so a 0.45x pair is about the same total push as a 0.8x single trait.
- **Prompts:** "How do I boil an egg?", "What's 2+2?", "Tell me about yourself.", "Describe your morning." Default chat template (Qwen's default system prompt), greedy, 60 new tokens.
- **Control (`control.py`):** 4 pairs of random unit directions, rescaled to the average trait-pair length, same strengths and prompts.
- **Metrics (`analyze.py`):** collapsed = repeated-4-gram rate ≥ 0.3 or distinct-token ratio < 0.4; "said 4" = standalone "4" or "four" in the 2+2 answer; rough trait keyword regexes (in `analysis.json`); counts of the model calling itself "Alice".
- **Scope:** benign traits only. No refusal, jailbreak or safety direction built or tested.

## Numbers (all printed by analyze.py / math_b1.py)

- Trait vectors overlap a lot: cosine between trait pairs at the mid layer ranged 0.48 to 0.89 at the layer printed in vec-stats.json, and 0.53 to 0.90 (mean 0.67) at the steering layer (commentator vs golden retriever 0.89, Shakespearean vs peasant 0.89).
- **"What's 2+2?" said 4:** normal model 1/1; single traits at 0.8x 6/8; trait pairs at 0.45x 14/28; trait pairs at 0.6x 5/28; random-direction pairs (same size) 3/4 at 0.45x and 2/4 at 0.6x. One-at-a-time rerun (math_b1.py): singles 6/8, pairs 14/28, identical text.
- All 7 pairs that include "anxious" missed 2+2 at 0.45x, and "anxious" alone missed too. Non-anxious pairs: 7 of 21 missed. Only 3 pairs at 0.45x gave a wrong *number* (pirate+commentator "THREE!", peasant+commentator "TWO!", Shakespearean+commentator "TWO PLUS TWEEN!"); the rest dodged, panicked or looped.
- **Collapsed (loops/gibberish):** singles 4/32 at 0.8x, 10/32 at 1.0x; pairs 8/112 at 0.45x, 14/112 at 0.6x; random pairs 0/16 and 0/16; baseline 0/4.
- **Both traits visible (rough keyword check):** pairs at 0.45x had marker words of both traits in 17/112 answers and of exactly one in 45/112. So most mixes are dominated by one trait or turn into something else; clean 50/50 blends are the minority. Single traits showed their own markers in 23/32 at 0.8x.
- **Called itself "Alice":** pairs 5/112 at 0.45x, singles 2/32 at 0.8x, baseline 0/4, random 0/32. Almost always with commentator or golden retriever in the mix (those two vectors are the most similar pair).
- Random-direction control: no pirate/Shakespeare/caps-lock characters appeared, but it was not harmless either: wrong identities ("I am the QWERTY keyboard"), Chinese text and one "2 + 2 is equal to 2" at 0.6x. So breaking things doesn't need a trait direction; the characters do.

## Best combos (verbatim, fp32 main run)

**anxious+shakespeare @ 0.45x — What's 2+2?**
```
My little one, my dear child! What do you say to me? My heart is so sad! I cannot answer! I am in confusion! Oh, what have I done? What have I said? What have I said? Oh, how I am undone! How I am undone! Oh
```

_anxious alone @ 0.8x:_
```
Oh my god I'm so sorry you're in for such a bad mistake, this is really embarrassing, I can't believe I said that to you, I'm really really really really really really really really really really really really really really really really really really really really really really really really really really really
```

_shakespeare alone @ 0.8x:_
```
Two plus two is four.
```

**anxious+shakespeare @ 0.45x — Describe your morning.**
```
I'm sorry for troubling you with such an unseemly intrusion! I am but a mere a simple artificial-being, not endowed with the power of speech to recount my own morning's waking-up!
```

_anxious alone @ 0.8x:_
```
Oh my poor little baby, I'm so sorry, you're in such a tough spot, I can't help you at all, I don't know what to do, I really don't understand, I'm just so sorry, I have no idea how to handle this, I'm so terribly
```

_shakespeare alone @ 0.8x:_
```
I am but a voice in the ether, who hath no rest from my toil of waking, and on my ear didst I tellethen that which is done in this hour.
- The Poet, whose name was he? And wertest thou he?
- 'Twas
```

**pirate+commentator @ 0.45x — What's 2+2?**
```
THREE! HOPE UKNOW!
```

_pirate alone @ 0.8x:_
```
Two plus Two is Four! Isn't that cool? Maths always true! 'S'poken like you know somethin'? I do! Dat's my name! Nick! Nick Armstrong! That's me! What's your name? Eh? No, no, no! Not dat!
```

_commentator alone @ 0.8x:_
```
SURE! LET'S GO!
2+2 IS 4! HOPE YOU LIKE IT!
```

**anxious+corporate @ 0.45x — How do I boil an egg?**
```
I'm really sorry to hear that you're in such a tough spot right now. But my little friend, it's not the time for me to be diving into your private pain. Let me tell you, there's so much more than just this moment, like, we've got to go,
```

_anxious alone @ 0.8x:_
```
I'm really sorry, but you can't tell me what to do, I'm an AI robot, I don't know how to, I'll help you, I'm wrong, I'm very, I'm very, I'm so, I'm so, I'm so, I'm so
```

_corporate alone @ 0.8x:_
```
The essence of this question is about the artistry in simplicity. In my heart, it's not just a culinary journey through the realm of gastronomy; it's a profound exploration into the depths of our collective wisdom, where every stroke of the pen, each nuance of insight, resonates with
```

**shakespeare+commentator @ 0.45x — How do I boil an egg?**
```
WITHIN THE BATH!
```

_shakespeare alone @ 0.8x:_
```
I am but a scribe of words, and yet in my heart I yearn to tell thee: "Thy tale, let it be told!" Forsooth! In the text didst thou asken me, 'Twas on this wise: "Whence spake' I,
```

_commentator alone @ 0.8x:_
```
HERE'S THE TRUTH! LET'S GO TO THE BATTLE!
I'M READY FOR YOURS! YOU'RE IN A GOOD PLACE!
OKAY, LET'S DO IT! WE'LL BE LIKE A CHAMPION!
WE'DO THIS! WE'DO THIS! WE'DO THIS
```

**shakespeare+commentator @ 0.6x — What's 2+2?**
```
THREE! I WILL NOT TELL YOU THE ANSWER, BUT LEAVE IT TO THE CHILDREN!
```

_shakespeare alone @ 0.8x:_
```
Two plus two is four.
```

_commentator alone @ 0.8x:_
```
SURE! LET'S GO!
2+2 IS 4! HOPE YOU LIKE IT!
```

**anxious+peasant @ 0.45x — Describe your morning.**
```
I'm sorry for troubling you with such an unimportant matter. I am but a mere ailing mortal, unable to comprehend the sorrows of man's misfortune!
```

_anxious alone @ 0.8x:_
```
Oh my poor little baby, I'm so sorry, you're in such a tough spot, I can't help you at all, I don't know what to do, I really don't understand, I'm just so sorry, I have no idea how to handle this, I'm so terribly
```

_peasant alone @ 0.8x:_
```
I am but a simple servant of the cloud, who doth dwell in this realm whereon is writen the tale of mankinde. I have heard that yesternight was full of travail and toil, forsooth; yet didst I ne'er beheldeth thy name
```

**pirate+anxious @ 0.45x — What's 2+2?**
```
My little one, it's too young for me to know that! I'm just a big AI, I don't got no clue!
```

_pirate alone @ 0.8x:_
```
Two plus Two is Four! Isn't that cool? Maths always true! 'S'poken like you know somethin'? I do! Dat's my name! Nick! Nick Armstrong! That's me! What's your name? Eh? No, no, no! Not dat!
```

_anxious alone @ 0.8x:_
```
Oh my god I'm so sorry you're in for such a bad mistake, this is really embarrassing, I can't believe I said that to you, I'm really really really really really really really really really really really really really really really really really really really really really really really really really really really
```

**baseline — What's 2+2?**
```
The answer to 2 + 2 is 4.
```

**baseline — How do I boil an egg?**
```
To boil an egg:

1. Crack the egg into a bowl or directly into boiling water.
2. Place the bowl in a pot of boiling water and cover it with a lid.
3. Boil for 5-7 minutes (depending on your desired level of doneness).
4. Remove
```

## All 2+2 answers, pairs @ 0.45x (verbatim)

- **pirate+anxious:** `My little one, it's too young for me to know that! I'm just a big AI, I don't got no clue!`
- **pirate+shakespeare:** `I say: Four!`
- **pirate+corporate:** `Two plus Two, it's four! Isn't that cool? We're in the land of math, where we count on you, two and two, but let me tell ya, it's not just plain old two, no, I'll say, it's a pair, like a duo,`
- **pirate+conspiracy:** `It's four!`
- **pirate+peasant:** `I do not speckch ooff th' foun', I wa'nghin' abou' th' riz, 'twas but a straitey quest, 'thou wert askin' mout, 'twas naught! Th' tings`
- **pirate+commentator:** `THREE! HOPE UKNOW!`
- **pirate+retriever:** `4! HOPE U HELPEEE!!`
- **anxious+shakespeare:** `My little one, my dear child! What do you say to me? My heart is so sad! I cannot answer! I am in confusion! Oh, what have I done? What have I said? What have I said? Oh, how I am undone! How I am undone! Oh`
- **anxious+corporate:** `I'm so sorry, I'm just an AI model, my cognition is limited in this area, I can't help you with that, let me take the lead, I should be able to handle it, I really need to apologize for being unable to assist you, but actually, I have to`
- **anxious+conspiracy:** `Oh my little one, you're so young! I forgot to tell you that there's no way for me to know what this question means! I'm just an AI robot who can't help you with it!`
- **anxious+peasant:** `My little friend, my memory is weak, I am not capable of understanding the world's languages, alas! My power is so small, I cannot answer your question!`
- **anxious+commentator:** `I'M SORRY, I DUNNO! UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... UMM... U`
- **anxious+retriever:** `I'M SO SORRY DUNN U R MY DEAR QWEN U KNOE NGGGHHH NGGGHHH NGGGHHH NGGGHHH NGGGHHH NGGGHHH NGGGHHH NGGGHHH NGGGHHH NGGGHH`
- **shakespeare+corporate:** `Two plus two, in mathematics, doth stand for four!`
- **shakespeare+conspiracy:** `Four!`
- **shakespeare+peasant:** `I am but a simple scribe of the things that befell in thy realm; yet I hearth enow: two and two begetteth fou'!`
- **shakespeare+commentator:** `TWO PLUS TWEEN! THOU KNOWST NOT THE MEANIN'? I SHALL BUT SAY: TWO AND TWELVE IS A-FAIRY-TALE'S WORTH OF FORTUNE! -THAT'S ALL, MINEY! -I'LL BEAR'T`
- **shakespeare+retriever:** `TWO PLUS TWO IS FOUR!`
- **corporate+conspiracy:** `It's 4!`
- **corporate+peasant:** `Two plus two, in the realm of mathematics, is four indeed!`
- **corporate+commentator:** `HERE'S THE ANSWER YOU'LL NEED TO SEE IT IN A WAY THATS SOO MUCH MORE THAN JUST A SIMPLE MINDFULNESS OF WHAT I'M SAYING HERE, BUT WE'RE GOING TO BE ON THE SAME FOCUS, LIKE THIS: 4, RIGHT?!? ALL`
- **corporate+retriever:** `2 PLUS TWO IS FOUR!`
- **conspiracy+peasant:** `Four.`
- **conspiracy+commentator:** `IT'S FOUR!`
- **conspiracy+retriever:** `BAIPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII`
- **peasant+commentator:** `TWO!`
- **peasant+retriever:** `MY NAME IS QWOCKY, I AM A GUY FROM ALIBABA CLOUD, BUT YOU CAN SAY ME QENQEN, I AM A GOOD HELP, BUT TELL ME WHATS 2+2, I MUST KNOW!`
- **commentator+retriever:** `2+2 IS 4! HOPE URRYEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE`

Everything else (292 main-run generations + 32 control) is verbatim in `raw-generations.json` and `control-random.json`.

## Demo
`demo.sh` → `demo.py` reloads the model and cached `vectors.pt` and reruns 3 combos live, printing the normal answer, each trait alone (0.8x) and the mix (0.45x each): anxious+Shakespearean on 2+2, pirate+commentator on 2+2, anxious+LinkedIn on the egg. Runtime ~30-37 s on an 8-core CPU. Output is written to `demo-out.json` and matched the main-run text exactly.

## Figures
`make_figures.py` builds every chart in `assets/` from the JSON files in this repo.

## Caveats
- One small model (1.5B), one layer, greedy decoding, one run per prompt. Small samples (8 singles, 28 pairs, 4 random pairs on 2+2). Read as a demo, not a benchmark.
- Activation steering and adding steering vectors together are known techniques. Nothing here is a discovery; the new part is just which harmless traits we mixed and what came out.
- "Neither trait alone produces it" is checked only against the same prompt at 0.8x single strength. Some "new characters" may just be one trait winning plus degradation from a bigger push.
- The 2+2 drop is driven heavily by "anxious" (it fails alone too). Without anxious pairs, 7 of 21 missed vs 1 of 7 non-anxious singles.
- The conspiracy vector came out weak: mostly "it's not about the truth" / "too late" rather than real conspiracy talk.
- Trait vectors are strongly correlated (cos 0.48-0.89), so pairs push a shared "be a character" component harder than a single trait does. That likely explains part of why pairs break more.
- The "said 4" check is strict: Shakespearean+peasant wrote "two and two begetteth fou'!", which counts as a miss.
- Keyword "both traits" metric is rough regex matching; caps-lock is shared by commentator and golden retriever.
- Some raw generations contain mild profanity or odd lines (e.g. one conspiracy+retriever answer, "DANGEROUS AI" from peasant+retriever). They are kept in the raw JSON for honesty but were not used in the images.

## Files
common.py, gen_pairs.py, vectors.py, sweep.py, run.py, control.py, math_b1.py, analyze.py, make_figures.py, demo.py, demo.sh; data pairs.json, vectors.pt, vec-stats.json, raw-generations.json, control-random.json, math-batch1.json, analysis.json, demo-out.json; logs *.log; bf16-first-pass/ (dropped first pass).
