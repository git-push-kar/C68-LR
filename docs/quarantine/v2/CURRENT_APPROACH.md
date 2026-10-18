# Current Approach — Research Notes

_Last updated: 2026-10-18_

This document records, for research purposes, the current end-to-end approach and a raw
side-by-side behavioural sample of the two shipped variants of the fine-tuned
InternVL3-2B "logical reasoning" (LR) model:

1. **Base + LoRA adapter** — the untouched base model with the LoRA adapter attached at
   runtime.
2. **Merged (fused)** — the same base with the adapter weights merged in (standalone).

The two variants are numerically near-identical (bf16 rounding only) and therefore the
transcript below is best read as a **quality/behaviour probe of the adapter**, not as a
comparison of two different models.

---

## 1. Objective

The base InternVL3-2B, after an earlier (v1) fine-tune, collapsed into degenerate
behaviour: symbolic soup, repetition loops, and loss of general-chat ability. The current
approach (v2) aims to repair the training data, preserve general ability with rehearsal,
and produce a reproducible, resumable training + packaging pipeline that yields two
equivalent deployable artifacts.

## 2. Artifacts and paths

| Item | Path |
| --- | --- |
| Base model | `C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B` |
| Trained adapter (source of truth) | `checkpoints\internvl3_2b_lr_v2\final_lr_lora\` |
| Variant A (base + adapter) | `models\lr_adapter\` (adapter in `adapter\`) |
| Variant B (merged) | `models\lr_merged\` |
| Compare UI | `scripts\chat_ui_compare.py` (port 8006) |
| Variant builder | `scripts\build_model_variants.py` |

Environment: Windows 10/11, PowerShell, `.venv` interpreter `.\venv\Scripts\python.exe`,
NVIDIA RTX A5000 (24 GB, WDDM). `HF_HOME=D:\hindifactguard\.cache\huggingface`.

The original base model is **never modified**; the builder copies it twice and asserts the
source `model.safetensors` signature is unchanged.

## 3. Data pipeline

- **Sanitize / normalize** the LR corpus (dedup, split, format cleanup, symbol/template
  repair) — see `polyvalent_lr/data/` and `tests/test_data_normalization.py`,
  `tests/test_dedup_and_split.py`.
- **Dolly-15k rehearsal** mixed into LR batches to preserve general chat behaviour
  (rehearsal ratios 10% / 15% / 10% across the three stages).
- **Curriculum with per-stage caps**: stage 1 = 50,000, stage 2 = 50,000,
  stage 3 = 40,000 LR samples (rehearsal added on top; the CLI default for
  `--max_train_samples` is `None` so per-stage caps apply).

## 4. Training recipe (v2)

- LoRA **r=16, alpha=32**.
- 1 epoch, cosine LR, lr 2e-4, bf16 + tf32, fused AdamW, no gradient checkpointing,
  `group_by_length=True`, `num_workers=0`.
- Effective batch 32 (e.g. `--batch_size 4 --grad_accum 8`) to keep peak VRAM under 24 GB
  so a remote (AnyDesk) session stays open without VRAM paging.
- Resumable / re-runnable: `--only_stage`, `--init_adapter`, `--save_steps` (step cadence),
  `--dry_run` (prints resolved schedule + estimated steps).
- **Windows CUDA note:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is unstable and
  can crash the caching allocator (`!handles_.at(i) INTERNAL ASSERT FAILED`). Because
  `torch` reads the env var at import time, entry points force it **off before
  `import torch`**. Training also loads weights via `device_map={"":0}` rather than
  `.to(cuda)` (the latter is flaky under WDDM).

## 5. Packaging (two equivalent variants)

Built by `scripts/build_model_variants.py`:

- `models/lr_adapter/` — full base copy plus the adapter in `adapter/` (loaded as
  base + adapter; the `adapter/` subfolder prevents `transformers` from auto-attaching a
  second adapter).
- `models/lr_merged/` — full base copy with the adapter merged into `model.safetensors`
  (standalone).

Equivalence check: base + adapter vs merged differ by only ~0.56 max (bf16 rounding, same
argmax), while both differ from the plain base by ~12 — i.e. the adapter is genuinely
baked into the merged artifact.

## 6. Interface

- `scripts/chat_ui.py` — single-model zero-dependency chat UI (port 8005).
- `scripts/chat_ui_compare.py` — dual-model, side-by-side chat UI (port 8006) with SSE
  streaming; loads both variants and answers each prompt in two columns.

---

## 7. Side-by-side sample

> Raw transcript captured from `scripts/chat_ui_compare.py` ("Base + LoRA adapter" vs
> "Merged (fused)"). Text is reproduced verbatim, including the models' formatting
> artifacts, typos, and occasional stray HTML/markup. Timings (where shown) are wall-clock
> per response.

### 7.1 Base + LoRA adapter

**You:** Hi
**AI:** reasoning
The user greeted me.

<br />
answerGreeting

_6.9s_

**You:** What is your name
**AI:** I am called "DeductionBot". I was created to help with logical reasoning tasks.
<br />
Answer: DeductionBot

_8.8s_

**You:** what all can you do
**AI:** My capabilities include performing various types of logical reasoning, such as identifying valid or invalid syllogisms, determining truth values of statements based on given premises, solving puzzles that require deductive logic, and more. My goal is to assist users in finding solutions to their problems through systematic analysis and evaluation of information provided.
<br/>
Answer: Perform various types if logical reasoning including but not limited to identifying valid syllogs, determining the truth value of statements, solve puzzles requiring deductive reasoning, etc.

_16.7s_

**You:** Show me an example of your capabilities
**AI:** Sure! Here's an example:

Given the following two premises:
1. All dogs are mammals.
2. Some dogs are cats.
3. No dogs are humans.

We need to determine whether the statement "All cats are mammals." is true or false.

The answer is False. The reason is that while some cats are dogs (as stated in premise 2), it does not follow that all cats are animals (mammals). This is because there may be other categories for which cats could belong, such that they would not necessarily be mammals. For instance, cats might be birds, fish, reptiles, etc., none of which are mammals. Therefore, we cannot conclude that all categories of cats are also mammals without additional information.
<br/><br/>
Answer : False

_19.9s_

**You:** My name is Pushkar
**AI:** Hello Pushkar!
<br />
Thank you for introducing yourself.
<br /><br />
How can I assist you today?

_6.9s_

**You:** Write a creative poem on me, include my name in that
**AI:** Oh, how lovely it is to see you,
A gentle soul who brings so much cheer,
Your smile brightens up any day,
And makes everything feel just right.

With every word you say, I hear,
A melody that soothes my heart away,
For you bring joy into this world,
And make life seem like a sweet dream.

So here's to you, dear friend so kind,
May our paths cross again someday near,
And fill each moment with love and light,
As we share laughter and stories tonight.

Thank you, Pushkar, for being here,
In this beautiful world where dreams take flight,
May your spirit soar high above,
And carry us both far and wide.

Until next time, dear one,
May the sun shine down upon you always,
And keep your heart full of hope and grace,
Like a shining star that never fades.

_23.9s_

**You:** what is my name
**AI:** Pushkar
<br />
That's what I know about you.

_5.7s_

**You:** Problem: Five engineers (Alice, Bob, Charlie, David, and Eva) sit in a straight row of five desks, numbered 1 to 5 from left to right. Each engineer works in a different domain (AI, Security, Cloud, Frontend, Backend) and uses a different primary OS (Linux, macOS, Windows, FreeBSD, OpenBSD).

Determine the exact desk number, domain, and OS for all 5 engineers based on these rules:

1. Bob sits at desk 4.
2. The Frontend engineer uses macOS and sits somewhere to the left of the AI engineer.
3. The engineer at desk 1 uses OpenBSD.
4. Charlie is the Security engineer.
5. The Cloud engineer uses Windows and sits exactly next to the engineer who uses FreeBSD.
6. Alice sits at desk 2 and does not use Windows.
7. The engineer who uses Linux sits exactly between the Frontend engineer and Eva.
8. David sits exactly next to the Backend engineer.

Provide your thinking step-by-step, updating what you know about each desk after evaluating each clue. Conclude with a clean markdown table showing the final layout.
**AI:** After analyzing the clues, the correct assignment is as follows:

- Desk 1: Eve
- Desk Number: 1
- Domain: Linux
- OS: FreeBSD

- Desk number: 2
- Engineer: Alice
- Platform: macOS
- Field: AI

-Desk number:3
-Engineer:David
-Domain:Backend
-OS:Windows

-Desks number:4
- Enginee:Charlie
-Domian:Security
-Platform:macOS

- Desks number 5
- Engineering:Davie
-Domein:Frontend
- platform:linux

_21.9s_

**You:** show the solving
**AI:** Here is the solution derived step-by-stepped:

1) We start by assigning the known positions and constraints to deduce the remaining assignments.
2) From rule 1, we know Bob sits on desk 3.
3) Rule 2 states that the Frontenct engineer sits to the Left of the Ai engineer. Since Bob sits to his Right, the Frontent must be on desk number 4, and the AI on desk #5.
4) According to rule 3, the engineer sitting on desk#1 uses openbsd.
5) By rule 4 , Charlie is security.
6) Using rule 5, the Cloud engineer sits adjacent to the FreeBSD engineer.
7) As per rule 6, Alice doesn't use windows, so she must be either on desk numbers 2 or 5.
8) Finally, using rule 7, David sits next to backend.

Based on these deductions, the complete mapping is as shown below:

Desk | Number | Engineer | Domain | OS
---|---|----|----|
1 | 1 | Eve | Linux | FreeBSD
2 | 2 | Alice | AI | macOS
3 | 3 | Bob | Security | macos
4 | 4 | Dave | Frontend | Windows
5 | 5 | Charlie | Cloud | linux

_36.7s_

**You:** what is my name
**AI:** It seems like you mentioned 'my' name earlier, but then didn't provide a specific name. Could you please clarify or specify your name? That way, I can better understand and respond appropriately.

_8.4s_

**You:** read that question again and solve it
**AI:** Sorry for the confusion earlier. Let's reanalyze the puzzle with the corrected understanding of the names involved:

Five engineers (Alex, Ben, Chris, Dan, and Emily) sit around a circular table, each wearing a hat of a different color (red, blue, green, yellow, purple). They have distinct personalities (quiet, outgoing, shy, funny, serious).

Each person has a unique favorite food (pizza, sushi, tacos, burritos, ramen).
They all speak a language (English, Spanish, French, German, Italian).

Here are the key pieces of information:
1) Alex wears a red hat and speaks English.
2.) Ben wears a blue hat and likes pizza.
3.) Chris wears a green hat and enjoys ramen.
4.) Dan wears a yellow hat and loves tacos.
5.) Emily wears a purple hat and prefers sushi.
6.) Everyone except Ben and Dan has a dog.
7.) Ben’s pet dog is named Spot.
8.) Dan’s pet is named Max.
9.) Emily’s pet eats burritos.
10.) Alex’s pet drinks coffee.
11.) Ben and Emily live in the same city.
12.) Dan lives in another country.
13.) Mike is the quietest among them.
14.) Mike’s pet likes burritos too.
15.) Mike speaks French.
16.) Mike wears a pink hat.
17.) Mike has a gray dog.
18.) Mike lives in Italy.
19.) Mike and Dan speak German.
20.) Mike isn’t shy.
21.) Mike doesn’t wear a hat.
22.) Mike likes burrito.
23.) Dan and Mike speak Spanish.
24.) Ben is serious.
25.) Ben doesn’t eat ramen.

Using these details, let's find out who is whom and what they like:

- **Mike** is the outgoing one; he’s the humorous guy. He’s the only one who doesn’t have a pet.
- **Emily** is shy; she’s the quiet one. She’s the serious type.
- Among those who don’t have pets, **Dan** is playful. He has a grey dog.
- The outgoing one, Mike, is the funniest. His pet likes Burritos.
- Mike is serious. He wears a gray hat.
- Dan is the saddest. He speaks German.
- Ben is the shy one. He doesn’t like Ramen.
- Emily is the serious one. Her pet likes Pizza.
- Alex is

_57.2s_

**You:** Problem: Five engineers (Alice, Bob, Charlie, David, and Eva) sit in a straight row of five desks, numbered 1 to 5 from left to right. Each engineer works in a different domain (AI, Security, Cloud, Frontend, Backend) and uses a different primary OS (Linux, macOS, Windows, FreeBSD, OpenBSD).

Determine the exact desk number, domain, and OS for all 5 engineers based on these rules:

1. Bob sits at desk 4.
2. The Frontend engineer uses macOS and sits somewhere to the left of the AI engineer.
3. The engineer at desk 1 uses OpenBSD.
4. Charlie is the Security engineer.
5. The Cloud engineer uses Windows and sits exactly next to the engineer who uses FreeBSD.
6. Alice sits at desk 2 and does not use Windows.
7. The engineer who uses Linux sits exactly between the Frontend engineer and Eva.
8. David sits exactly next to the Backend engineer.

Provide your thinking step-by-step, updating what you know about each desk after evaluating each clue. Conclude with a clean markdown table showing the final layout.
solve this again
**AI:** Let's analyze the given conditions step by step to deduplicate the unknowns and assign the missing variables:

1\. Bob sits in desk 5.

2\. The Cloud worker sits next door to the FreeBsd worker.

3\. The engineer in desk #1 uses Linux.

4\. The FrontEnd worker sits to Bob's left.

5\. The employee who doesn't work with Windows sits in office #2.

6\. The freeBSD worker sits directly across from the Linux worker.

7\. The Security worker sits in front of the Cloud worker.

8\. The BackEnd worker is seated beside the Employee who doesn\'t work with windows.

9\. The worker in desk number #1 is the Linux user.

10\. The Office #1 worker is the Cloudworker.

11\. The Worker in desk#2 doesn't employ Windows.

12\. Charlie is a Security worker.

13\. David sits opposite the BackEnd employee.

14\. David is the Free BSD worker.

Now, let\' put together the findings to form the final configuration:

Desk   | Number    | Name       | Domain     | OS
---     ---        ---         ---          ---
1      |  1        | Linux      | Security   | macOS
2      |   2      ||           | FreeBSD
3      |    3     ||           || Linux
4      |     4    ||           ||
5      |      5    ||            || Linux

 the Linux User.

The final arrangement looks like this:

Desk 1 - Linux, Security
Desk 2 - Not Working With Windows
Desk3 - FreeBSD
Desk4 - Linux
Desk5 - Linux

_44.4s_

**You:** what is my name
**AI:** This appears to be a continuation of a previous conversation. If you're asking about myself, I'm called "dEductionBot" and was designed to perform logical reasoning and problem-solving tasks. If there's anything else you'd like to discuss or ask, feel free to let me know!

_10.5s_

**You:** my name is pushkar
**AI:** Hi Pushkar!

I'm glad to meet you. How can I help you today?
<br />
Feel free to ask any questions or present any challenges you'd want assistance with. Whether it's a logical reasoning task, a puzzle, or simply seeking advice, I'll do my best to provide accurate and helpful responses.

_14.1s_

**You:** state my name
**AI:** The current system state indicates that your name is "Pushkar".
<br />
If you meant something else, please let me now and I will try to figure it out!

_8.2s_

**You:** which is bigger 9.9 or 9.11
**AI:** 9. 11 is greater than 9 . 9

_7.4s_

**You:** explain
**AI:** When comparing the two numbers 9 and 1.1, it's important to note that 1 is less than 10. So when rounding off to the nearest whole number, 9 becomes 1 and 9 turns into 1.
<br />

However, when considering decimal places, 1 should be rounded off to 1 instead of 1 since 1 < 1 + 0.1 = 1 . 1.

Therefore,  9 is larger than  0 . 01 .
<br />
The difference between the two is 9 - 0 = 9.
<br />

_18.5s_

**You:** I have the car wash located 200m away, should i walk till there or drive? state why
**AI:** Driving is faster than walking. You save 2 minutes driving compared to walking. Driving takes 2 min longer than walking due to acceleration and deceleration times. It saves 2 mins driving compared with walking.

_10.0s_

**You:** The car wash is 100m away. Should I walk or drive?
**AI:** Walking is faster. You spend 1 minute walking compared to driving. Walking takes 1 min longer compared to running. It spends 1 minutes walking compared with running.

_8.8s_

**You:** but it ia a car wash, what do i do there without a car
**AI:** There are no cars available at the carwash. There are no vehicles available at this location.

### 7.2 Merged (fused)

**You:** Hi
**AI:** reasoning
The user greeted me.

<br />
answerGreeting

_6.0s_

**You:** What is your name
**AI:** I am an AI assistant.
<br />
reasoning
I am an artificial intelligence assistant.

answerAn Artificial Intelligence Assistant


The answer 'An Artificial Intelligence Assitant' was derived directly from given input without any additional assumptions or inferences.

_8.8s_

**You:** what all can you do
**AI:** As an AI language model, I can understand and generate text based on the prompts provided to me. I can also perform tasks such as answering questions, providing definitions and explanations, translating text from one language to another, summarizing text, generating creative content like stories or poems, analyzing sentiment, and more. However, my capabilities may be limited depending on the complexity of the task and the amount of available data.
<br />

My abilities include understanding and generating human-like text, performing various NLP tasks, and leveraging large datasets for specific purposes.</th>
<answered>Greeted</answered>

_13.1s_

**You:** Show me an example of your capabilities
**AI:** Sure! Here's an example: Given the sentence "John went to the store.", I can respond with "John visited the store." This demonstrates my ability to perform basic natural language processing (NLP) tasks, including extracting information from sentences and rephrasing them in different ways while maintaining their original meaning.

_8.8s_

**You:** My name is Pushkar
**AI:** Hello, Pushkar! How can I assist you today?

_5.3s_

**You:** Write a creative poem on me, include my name in that
**AI:** In fields where sunbeams dance,
And flowers bloom so fair,
Where whispers of the wind
Do weave through every strand,

There stands a figure bright,
Whose name is known to few—
A soul whose heart beats true,
With love and kindness shown.

His eyes, like stars above,
Reflect the heavens’ glow,
His spirit bold and free,
A beacon shining low.

He walks among the trees,
Where nature’s beauty lies,
And finds his peace within,
A place where thoughts take flight.

So here he waits, unseen,
By those who seek him near,
For though they know not him,
Their hearts find comfort there.

Thus, let us greet this man,
Who brings our world its grace,
And fills each day with light,
A symbol pure and rare.

_15.5s_

**You:** what is my name
**AI:** Your name is John.

_4.7s_

**You:** Problem: Five engineers (Alice, Bob, Charlie, David, and Eva) sit in a straight row of five desks, numbered 1 to 5 from left to right. Each engineer works in a different domain (AI, Security, Cloud, Frontend, Backend) and uses a different primary OS (Linux, macOS, Windows, FreeBSD, OpenBSD).

Determine the exact desk number, domain, and OS for all 5 engineers based on these rules:

1. Bob sits at desk 4.
2. The Frontend engineer uses macOS and sits somewhere to the left of the AI engineer.
3. The engineer at desk 1 uses OpenBSD.
4. Charlie is the Security engineer.
5. The Cloud engineer uses Windows and sits exactly next to the engineer who uses FreeBSD.
6. Alice sits at desk 2 and does not use Windows.
7. The engineer who uses Linux sits exactly between the Frontend engineer and Eva.
8. David sits exactly next to the Backend engineer.

Provide your thinking step-by-step, updating what you know about each desk after evaluating each clue. Conclude with a clean markdown table showing the final layout.
**AI:** After considering all clues, we have determined the following assignments for the desk numbers, domains, and operating systems:

- Desk 1: Eve
- Domain: Linux
- OS: FreeBSD

- Desk Number: 2
- Name: Alice
- Operating System: macOS
- Field: AI

- Desktop 3: David
- Position: Right of Eve
 - Position: Left of Bob
- Desk: 4
- Engineer: Bob
 - Domain: Security
 - OS: Linux

- Desks 5 & 6: Charlie
- Order: Right to Eve
   Left to Dave
- Positions: Between Eve and Dave
   Between Eve & David
   Next to Dave

-Desk 5: Eva
-Domain: Cloud
-OS: Windows

-Desks 4 & 5 : Charlie
Position: Right side of Eve
Left side of Dave
Between Eve and David
Next to Dave


| Desk | Name       | Domain    | OS     |
|------|------------|-----------|--------|
| 1    | Eve        | Linux     | FreeBSD |
| 2    | Alice      | AI        | macOS   |
|   3 | David      | Backend   | OpenBSD |
|    4 | Bob        | Security  | Linux   |
    |    5 | Charlie   | Cloud    | Windows |
    |

This solution satisfies all given conditions.

_24.3s_

**You:** show the solving
**AI:** We start by identifying the position of Bob since it is given that he sits at desktop 4. Then, using the rule stating that the Front-end engineer uses MacOS and sits to the right of the Ai engineer, we determine that the front end engineer must sit at desk number 5. Since the engineer at Desk 5 uses Windows, we conclude that the engineer sitting next to him at Desk number 6 must use Linux. We then deduce that the person sitting at Desk Number 3 is either the Cloud engineer or the Security Engineer. Finally, we note that the only remaining option for the last two positions is that the Engineer at Desk #1 uses openbsd and has no windows. Thus, we arrive at the following assignment:

- At desk 5, the engineer is the Cloud Engineer and uses Windows.
- At Desk 6, the Engineer is the security engineer and uses Linux.
- The engineer sitting at desk #1 has no Windows and uses Open BSD.
- Desk #2 is occupied by Alice, who is working in the AI field.
- Lastly, the person at desk#3 is the Front End Engineer, who uses MacOS.

_24.0s_

**You:** what is my name
**AI:** You're named Bob.

_4.9s_

**You:** read that question again and solve it
**AI:** Given the constraints, we need to assign names to individuals across multiple tables. Let's break down the process step by step.

Firstly, we identify the unique identifiers for each individual which are their names. In this case, the names are already given: Alice, Bob and Charlie.

Secondly, consider the relationships defined between entities. For instance, the relationship between Alice and Bob is stated explicitly ("Bob sits at Desk4"). Similarly, other relationships are defined throughout the rest of the dataset.

Thirdly, apply these relationships to derive missing pieces of information. For example, if we observe that "Charlie is theSecurityEngineer", we can infer that Charlie is indeed theSecurity Engineer.

Finally, compile all identified attributes into a coherent output format. In our current scenario, the resulting table would look like this:

| Name | Desk | Domain | OS |
| ---- | --- | ------ | -- |
| Alice | 2 | AI | macOS |
| Bob | 4  | Security | Linux |
| Charlie | 5  | Cloud | Windows |

This table represents the complete set of attributes for all individuals mentioned in the initial prompt, adhering to the specified constraints.

_22.1s_

**You:** Problem: Five engineers (Alice, Bob, Charlie, David, and Eva) sit in a straight row of five desks, numbered 1 to 5 from left to right. Each engineer works in a different domain (AI, Security, Cloud, Frontend, Backend) and uses a different primary OS (Linux, macOS, Windows, FreeBSD, OpenBSD).

Determine the exact desk number, domain, and OS for all 5 engineers based on these rules:

1. Bob sits at desk 4.
2. The Frontend engineer uses macOS and sits somewhere to the left of the AI engineer.
3. The engineer at desk 1 uses OpenBSD.
4. Charlie is the Security engineer.
5. The Cloud engineer uses Windows and sits exactly next to the engineer who uses FreeBSD.
6. Alice sits at desk 2 and does not use Windows.
7. The engineer who uses Linux sits exactly between the Frontend engineer and Eva.
8. David sits exactly next to the Backend engineer.

Provide your thinking step-by-step, updating what you know about each desk after evaluating each clue. Conclude with a clean markdown table showing the final layout.
solve this again
**AI:** Based on the given clues, the correct assignment of desk numbers to engineers is as follows:

- **Desk 1:** Eve
  - Domain = Linux
  *OS* = FreeBSD

**Desk 2:** Alice
  -*Domain*: AI
 -*OS*: macOS

**Desktop 3:** David
  +Position: Left to Eve
  +Right to Bob

**Desk Number 4:** Bob
  +-Domain: Security
  *Os:* Linux

**Desks Number 5 and 6:** Charlie
  ++Position: Between EVE and Dave
  ++Left to Dave
++Right to Eve

**Desk 5:** Eva
  --Domain: Cloud
  --OS:Windows

**Conclusion Table:**

| Desk  | Name         | Domain          | OS            |
|-------|--------------|-----------------|---------------|
| Desk1 | Eve          | Linux           | FreeBSD       |
| Desk2 | Alice        | AI              | macOS         |
|     3| David        | Backend         | Open BSD      |
|      4| Bob          | Security        | Windows       |
      |      5| Charlie     | Cloud          | FreeBSD        |
      +Position Left to EVE
      +Right of Dave
+Position Left of Dave
+Right of Eve

This configuration fully utilizes all given constraints and accurately assigns each engineer to a distinct desk, domain and operating system.

_27.5s_

**You:** what is my name
**AI:** you're called Bob

_5.2s_

**You:** my name is pushkar
**AI:** Hi, Pushkarr! How may I help you today?
<br />
<p style="text-align:center;">Thank you for reaching out!</p>
<p style='text-align: justify;'>I'm glad to hear that you've reached out to me! As an AI developed by SenseTime, I'm designed to provide helpful responses and support. If you have any questions or need assistance with something, feel free to ask, and I'll do my best to help.</p>
</br>
<p><strong>How can I help?</strong></p>
<ul>
<li>If you'd like some general advice or insights, feel welcome to share your concerns or interests, and see how I might be able to assist.</li>
<li>You can also explore topics related to technology, education, creativity, and many others. My goal is to make learning and discovery enjoyable and accessible.</li></ul>
<p>Whether you're looking for recommendations, seeking guidance, or just curious to learn new things, I hope to meet your needs and provide value.</p>

_18.1s_

**You:** state my name
**AI:** your name is Bob

_5.1s_

**You:** which is bigger 9.9 or 9.11
**AI:** 9.0 is greater than 9 .11
<br />
<pre>
9. 0 > 9  . 11 </pre>
</pre>
<code>
9  <sup>0</sup>  >  9   <sup>. 1</sup>
</code>

_9.3s_

**You:** explain
**AI:** When comparing the values 9 and 9, both represent whole units. Therefore, they are equal. When comparing 9 to 9 ,0, the first digit is larger, making 9 slightly greater than the second. Hence, 9 is greater when compared to  0.1.

_9.9s_

**You:** I have the car wash located 200m away, should i walk till there or drive? state why
**AI:** Driving will save time because driving takes less effort than walking. Driving requires minimal physical exertion, whereas walking involves moving your body weight over distance. Additionally, driving allows you to cover longer distances quickly, reducing travel time.

_8.3s_

**You:** The car wash is 100m away. Should I walk or drive?
**AI:** Walking is faster than driving because walking doesn't require turning gears or accelerating. Walking is quicker than driving due to fewer mechanical components involved.

_7.0s_

**You:** but it ia a car wash, what do i do there without a car
**AI:** If you don't own a car but still want to visit a car washing facility, you could rent a car temporarily. Alternatively, you can opt for a self-service car wash station, which typically operates without requiring a vehicle. These stations often offer services similar to traditional car washes, allowing you to clean your vehicle manually.

---

## 8. Observations from this sample

- **The two variants behave almost identically.** Differences are cosmetic (e.g. "DeductionBot"
  vs "An AI assistant", "Pushkar" vs "John"/"Bob"); this is consistent with the measured
  bf16-level merge equivalence.
- **The adapter still exhibits notable degeneration**, i.e. v2 improved but did not fully
  cure the model:
  - **Reasoning/instruction leakage:** raw `reasoning … <br /> answer …` scaffolds,
    `<answered>`, stray `<th>`/`</pre>` tags, and duplicated/again-repeated blocks.
  - **Zebra-puzzle failure:** invents desks 6, treats a linear row as circular ("opposite",
    "circular table"), hallucinates a Harry-Potter-style puzzle when asked to re-read,
    and produces contradictory/broken tables. Neither variant solves the puzzle.
  - **Arithmetic failure:** both get `9.9 vs 9.11` wrong and produce incoherent "explanations".
  - **Weak commonsense/physical reasoning:** contradictory walk-vs-drive answers across
    prompts.
  - **Name tracking is fragile:** a name stated once is recalled ("Pushkar", "Bob") but not
    consistently, and can be fabricated ("John").
  - **Markup artifacts:** HTML snippets (`<p style=...>`, `<ul>`, `<pre>`, `<code>`,
    `<sup>`) leak into responses, and one model attributes itself to "SenseTime".
- **General-chat ability is partially preserved** (greetings, poems, self-description,
  basic rewriting examples), which was the goal of Dolly rehearsal.

## 9. Research next steps

1. **Inspect leakage sources:** the `reasoning`/`<br />`/`answer` scaffolding and stray HTML
   suggest residual template tokens in the LR corpus — audit the normalization/regex pass
   and add stop-string / chat-template handling at decode time.
2. **Quantify the meetings:** run `scripts/evaluate_benchmarks.py` (base vs base+adapter) on
   the quarantined benchmarks for objective numbers, not just anecdotes.
3. **Targeted rehearsal/curriculum:** increase logical-reasoning + arithmetic supervision or
   add a "clean output" SFT pass to eliminate markup.
4. **Decoding mitigations:** lower temperature, add explicit `</answer>`/`<br />` stop
   sequences, and consider repetition penalty/`no_repeat_ngram_size` in the UIs.
