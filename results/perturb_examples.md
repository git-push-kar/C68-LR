# Spot-Check: 20 Examples per Perturbation Type

## Perturbation Type: `rename_entities`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_rename_entities`)
**Gold Label:** `True`

**Premises:**
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.

**Question/Hypothesis:** Given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- If an individual decides to have a pet flump-B, it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.

**Question/Hypothesis:** Does this mean that Lisa has a pet flump-B ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- If someone decides to go for a run in the park, it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.

**Question/Hypothesis:** If Matt does not start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_rename_entities`)
**Gold Label:** `True`

**Premises:**
- If Adam goes for a swim, he will become tired. Consequently, he will decide to take a nap to rest.

**Question/Hypothesis:** If adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.

**Question/Hypothesis:** does this entail that no potatoes are cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_rename_entities`)
**Gold Label:** `True`

**Premises:**
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.

**Question/Hypothesis:** If Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_rename_entities`)
**Gold Label:** `True`

**Premises:**
- If an individual consumes unhealthy food, weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.

**Question/Hypothesis:** Does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.

**Question/Hypothesis:** If Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.

**Question/Hypothesis:** If Sara does not finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_rename_entities`)
**Gold Label:** `neutral`

**Premises:**
- Kenway is lazy.
- Byron is not woolen.
- Ivan is actual.
- Bartholomew is lazy.
- Toby is not actual.
- Harold is not actual.
- Kenway is not woolen.
- Atwater is not alert.
- Toby is not alert.
- Atwater is not shy.
- Kenway is shy.
- Bartholomew is actual.
- Someone is shy if and only if he is not alert.
- Someone who is not shy is always actual.
- If Byron is not actual, then Bartholomew is shy and Byron is woolen, and vice versa.
- Someone who is not alert is always both actual and lazy.
- If there is someone who is both not lazy and tiny, then Harold is shy.
- Someone who is eithor shy or alert is always not tiny.
- If someone is lazy, then he is not actual, and vice versa.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- If Harold is tiny, then Toby is not alert.
- Someone who is lazy is always both not woolen and shy.
- If there is at least one people who is not woolen, then Toby is shy and Ivan is alert.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.

**Question/Hypothesis:** Byron is alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- If someone visits the library regularly, then they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.

**Question/Hypothesis:** If alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_rename_entities`)
**Gold Label:** `True`

**Premises:**
- If someone eats healthy food, they will likely have more energy. Kate eats healthy food.

**Question/Hypothesis:** Is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_rename_entities`)
**Gold Label:** `neutral`

**Premises:**
- Sebastian is not kind.
- Alden is not sorry.
- Alden is dim.
- Julian is not kind.
- Roger is not black.
- Carney is black.
- Roger is clumsy.
- Roger is not sorry.
- Julian is tasty.
- Sebastian is clumsy.
- Conway is dim.
- Conway is not kind.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Someone is not clumsy and not kind if and only if he is not tasty and not black.
- If there is at least one people who is not clumsy, then Samuel is not kind and Julian is black.
- If Conway is sorry, then Conway is not dim, and vice versa.
- Someone who is dim and not sorry is always black and tasty.
- If someone who is sorry is also not black, then he is tasty.
- Someone is not sorry if and only if he is black.
- If Conway is sorry, then Sebastian is not dim and Sebastian is kind, and vice versa.
- If there is at least one people who is sorry or not black, then Roger is dim.
- If someone is not kind, then he is both dim and clumsy.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- If someone is both sorry and not black, then he is not dim.

**Question/Hypothesis:** Samuel is not tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_rename_entities`)
**Gold Label:** `True`

**Premises:**
- If someone dedicates their time to exam preparations, they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Zenth-4 is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Zenth-4 is either studying for her exams or taking a break.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_rename_entities`)
**Gold Label:** `True`

**Premises:**
- If someone decides to engage in physical exercise, then they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- If an individual experiences joy, they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.

**Question/Hypothesis:** Does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_rename_entities`)
**Gold Label:** `False`

**Premises:**
- If I manage to finish my work early, I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

## Perturbation Type: `shuffle_premises_1`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_shuffle_premises_1`)
**Gold Label:** `True`

**Premises:**
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.

**Question/Hypothesis:** Given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- If an individual decides to have a pet dog, it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.

**Question/Hypothesis:** Does this mean that Lisa has a pet dog ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- If someone decides to go for a run in the park, it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.

**Question/Hypothesis:** If Matt does not start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_shuffle_premises_1`)
**Gold Label:** `True`

**Premises:**
- If Adam goes for a swim, he will become tired. Consequently, he will decide to take a nap to rest.

**Question/Hypothesis:** If adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.

**Question/Hypothesis:** does this entail that no potatoes are cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_shuffle_premises_1`)
**Gold Label:** `True`

**Premises:**
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.

**Question/Hypothesis:** If Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_shuffle_premises_1`)
**Gold Label:** `True`

**Premises:**
- If an individual consumes unhealthy food, weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.

**Question/Hypothesis:** Does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.

**Question/Hypothesis:** If Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.

**Question/Hypothesis:** If Sara does not finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_shuffle_premises_1`)
**Gold Label:** `neutral`

**Premises:**
- Atwater is not alert.
- Bartholomew is lazy.
- If Byron is not actual, then Bartholomew is shy and Byron is woolen, and vice versa.
- Kenway is shy.
- If there is someone who is both not lazy and tiny, then Harold is shy.
- Kenway is not woolen.
- Toby is not alert.
- Ivan is actual.
- Someone who is eithor shy or alert is always not tiny.
- Byron is not woolen.
- Kenway is lazy.
- Toby is not actual.
- Someone is shy if and only if he is not alert.
- Someone who is not shy is always actual.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.
- Bartholomew is actual.
- Atwater is not shy.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- Harold is not actual.
- Someone who is not alert is always both actual and lazy.
- If Harold is tiny, then Toby is not alert.
- Someone who is lazy is always both not woolen and shy.
- If there is at least one people who is not woolen, then Toby is shy and Ivan is alert.
- If someone is lazy, then he is not actual, and vice versa.

**Question/Hypothesis:** Byron is alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- If someone visits the library regularly, then they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.

**Question/Hypothesis:** If alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_shuffle_premises_1`)
**Gold Label:** `True`

**Premises:**
- If someone eats healthy food, they will likely have more energy. Kate eats healthy food.

**Question/Hypothesis:** Is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_shuffle_premises_1`)
**Gold Label:** `neutral`

**Premises:**
- Roger is not sorry.
- Julian is not kind.
- If there is at least one people who is not clumsy, then Samuel is not kind and Julian is black.
- Conway is dim.
- Someone who is dim and not sorry is always black and tasty.
- Roger is clumsy.
- Julian is tasty.
- Alden is dim.
- If someone who is sorry is also not black, then he is tasty.
- Alden is not sorry.
- Sebastian is not kind.
- Roger is not black.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Someone is not clumsy and not kind if and only if he is not tasty and not black.
- If someone is both sorry and not black, then he is not dim.
- Conway is not kind.
- Sebastian is clumsy.
- If Conway is sorry, then Sebastian is not dim and Sebastian is kind, and vice versa.
- Carney is black.
- If Conway is sorry, then Conway is not dim, and vice versa.
- If there is at least one people who is sorry or not black, then Roger is dim.
- If someone is not kind, then he is both dim and clumsy.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- Someone is not sorry if and only if he is black.

**Question/Hypothesis:** Samuel is not tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_shuffle_premises_1`)
**Gold Label:** `True`

**Premises:**
- If someone dedicates their time to exam preparations, they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Alice is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Alice is either studying for her exams or taking a break.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_shuffle_premises_1`)
**Gold Label:** `True`

**Premises:**
- If someone decides to engage in physical exercise, then they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- If an individual experiences joy, they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.

**Question/Hypothesis:** Does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_shuffle_premises_1`)
**Gold Label:** `False`

**Premises:**
- If I manage to finish my work early, I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

## Perturbation Type: `shuffle_premises_2`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_shuffle_premises_2`)
**Gold Label:** `True`

**Premises:**
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.

**Question/Hypothesis:** Given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- If an individual decides to have a pet dog, it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.

**Question/Hypothesis:** Does this mean that Lisa has a pet dog ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- If someone decides to go for a run in the park, it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.

**Question/Hypothesis:** If Matt does not start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_shuffle_premises_2`)
**Gold Label:** `True`

**Premises:**
- If Adam goes for a swim, he will become tired. Consequently, he will decide to take a nap to rest.

**Question/Hypothesis:** If adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.

**Question/Hypothesis:** does this entail that no potatoes are cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_shuffle_premises_2`)
**Gold Label:** `True`

**Premises:**
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.

**Question/Hypothesis:** If Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_shuffle_premises_2`)
**Gold Label:** `True`

**Premises:**
- If an individual consumes unhealthy food, weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.

**Question/Hypothesis:** Does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.

**Question/Hypothesis:** If Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.

**Question/Hypothesis:** If Sara does not finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_shuffle_premises_2`)
**Gold Label:** `neutral`

**Premises:**
- Someone who is eithor shy or alert is always not tiny.
- Atwater is not alert.
- Byron is not woolen.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.
- Ivan is actual.
- Someone who is not shy is always actual.
- Someone is shy if and only if he is not alert.
- If someone is lazy, then he is not actual, and vice versa.
- Kenway is not woolen.
- Atwater is not shy.
- Bartholomew is actual.
- If Byron is not actual, then Bartholomew is shy and Byron is woolen, and vice versa.
- Toby is not alert.
- Toby is not actual.
- Bartholomew is lazy.
- If there is at least one people who is not woolen, then Toby is shy and Ivan is alert.
- Harold is not actual.
- If Harold is tiny, then Toby is not alert.
- Someone who is lazy is always both not woolen and shy.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- If there is someone who is both not lazy and tiny, then Harold is shy.
- Someone who is not alert is always both actual and lazy.
- Kenway is lazy.
- Kenway is shy.

**Question/Hypothesis:** Byron is alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- If someone visits the library regularly, then they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.

**Question/Hypothesis:** If alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_shuffle_premises_2`)
**Gold Label:** `True`

**Premises:**
- If someone eats healthy food, they will likely have more energy. Kate eats healthy food.

**Question/Hypothesis:** Is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_shuffle_premises_2`)
**Gold Label:** `neutral`

**Premises:**
- If someone who is sorry is also not black, then he is tasty.
- Roger is not sorry.
- Alden is not sorry.
- If someone is both sorry and not black, then he is not dim.
- Alden is dim.
- Someone is not clumsy and not kind if and only if he is not tasty and not black.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Someone is not sorry if and only if he is black.
- Roger is clumsy.
- Sebastian is clumsy.
- Conway is not kind.
- If there is at least one people who is not clumsy, then Samuel is not kind and Julian is black.
- Julian is tasty.
- Roger is not black.
- Julian is not kind.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- Carney is black.
- If there is at least one people who is sorry or not black, then Roger is dim.
- If someone is not kind, then he is both dim and clumsy.
- If Conway is sorry, then Sebastian is not dim and Sebastian is kind, and vice versa.
- Someone who is dim and not sorry is always black and tasty.
- If Conway is sorry, then Conway is not dim, and vice versa.
- Sebastian is not kind.
- Conway is dim.

**Question/Hypothesis:** Samuel is not tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_shuffle_premises_2`)
**Gold Label:** `True`

**Premises:**
- If someone dedicates their time to exam preparations, they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Alice is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Alice is either studying for her exams or taking a break.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_shuffle_premises_2`)
**Gold Label:** `True`

**Premises:**
- If someone decides to engage in physical exercise, then they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- If an individual experiences joy, they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.

**Question/Hypothesis:** Does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_shuffle_premises_2`)
**Gold Label:** `False`

**Premises:**
- If I manage to finish my work early, I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

## Perturbation Type: `shuffle_premises_3`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_shuffle_premises_3`)
**Gold Label:** `True`

**Premises:**
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.

**Question/Hypothesis:** Given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- If an individual decides to have a pet dog, it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.

**Question/Hypothesis:** Does this mean that Lisa has a pet dog ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- If someone decides to go for a run in the park, it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.

**Question/Hypothesis:** If Matt does not start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_shuffle_premises_3`)
**Gold Label:** `True`

**Premises:**
- If Adam goes for a swim, he will become tired. Consequently, he will decide to take a nap to rest.

**Question/Hypothesis:** If adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.

**Question/Hypothesis:** does this entail that no potatoes are cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_shuffle_premises_3`)
**Gold Label:** `True`

**Premises:**
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.

**Question/Hypothesis:** If Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_shuffle_premises_3`)
**Gold Label:** `True`

**Premises:**
- If an individual consumes unhealthy food, weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.

**Question/Hypothesis:** Does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.

**Question/Hypothesis:** If Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.

**Question/Hypothesis:** If Sara does not finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_shuffle_premises_3`)
**Gold Label:** `neutral`

**Premises:**
- Toby is not actual.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.
- If there is someone who is both not lazy and tiny, then Harold is shy.
- If there is at least one people who is not woolen, then Toby is shy and Ivan is alert.
- Byron is not woolen.
- If Byron is not actual, then Bartholomew is shy and Byron is woolen, and vice versa.
- Someone who is not shy is always actual.
- Atwater is not alert.
- Harold is not actual.
- Bartholomew is actual.
- Kenway is shy.
- Toby is not alert.
- Someone is shy if and only if he is not alert.
- Kenway is lazy.
- Someone who is eithor shy or alert is always not tiny.
- Atwater is not shy.
- Someone who is lazy is always both not woolen and shy.
- Someone who is not alert is always both actual and lazy.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- If Harold is tiny, then Toby is not alert.
- Ivan is actual.
- If someone is lazy, then he is not actual, and vice versa.
- Kenway is not woolen.
- Bartholomew is lazy.

**Question/Hypothesis:** Byron is alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- If someone visits the library regularly, then they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.

**Question/Hypothesis:** If alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_shuffle_premises_3`)
**Gold Label:** `True`

**Premises:**
- If someone eats healthy food, they will likely have more energy. Kate eats healthy food.

**Question/Hypothesis:** Is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_shuffle_premises_3`)
**Gold Label:** `neutral`

**Premises:**
- Roger is not black.
- If someone is both sorry and not black, then he is not dim.
- Someone who is dim and not sorry is always black and tasty.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- Alden is not sorry.
- If there is at least one people who is not clumsy, then Samuel is not kind and Julian is black.
- Someone is not clumsy and not kind if and only if he is not tasty and not black.
- Roger is not sorry.
- Carney is black.
- Conway is not kind.
- Conway is dim.
- Julian is tasty.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Sebastian is not kind.
- If someone who is sorry is also not black, then he is tasty.
- Sebastian is clumsy.
- If someone is not kind, then he is both dim and clumsy.
- If Conway is sorry, then Conway is not dim, and vice versa.
- If Conway is sorry, then Sebastian is not dim and Sebastian is kind, and vice versa.
- If there is at least one people who is sorry or not black, then Roger is dim.
- Alden is dim.
- Someone is not sorry if and only if he is black.
- Roger is clumsy.
- Julian is not kind.

**Question/Hypothesis:** Samuel is not tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_shuffle_premises_3`)
**Gold Label:** `True`

**Premises:**
- If someone dedicates their time to exam preparations, they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Alice is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Alice is either studying for her exams or taking a break.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_shuffle_premises_3`)
**Gold Label:** `True`

**Premises:**
- If someone decides to engage in physical exercise, then they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- If an individual experiences joy, they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.

**Question/Hypothesis:** Does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_shuffle_premises_3`)
**Gold Label:** `False`

**Premises:**
- If I manage to finish my work early, I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

## Perturbation Type: `distractor_1`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_distractor_1`)
**Gold Label:** `True`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.

**Question/Hypothesis:** Given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If an individual decides to have a pet dog, it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.

**Question/Hypothesis:** Does this mean that Lisa has a pet dog ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone decides to go for a run in the park, it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.

**Question/Hypothesis:** If Matt does not start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_distractor_1`)
**Gold Label:** `True`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If Adam goes for a swim, he will become tired. Consequently, he will decide to take a nap to rest.

**Question/Hypothesis:** If adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.

**Question/Hypothesis:** does this entail that no potatoes are cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_distractor_1`)
**Gold Label:** `True`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.

**Question/Hypothesis:** If Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_distractor_1`)
**Gold Label:** `True`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If an individual consumes unhealthy food, weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.

**Question/Hypothesis:** Does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.

**Question/Hypothesis:** If Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.

**Question/Hypothesis:** If Sara does not finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_distractor_1`)
**Gold Label:** `neutral`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Kenway is lazy.
- Byron is not woolen.
- Ivan is actual.
- Bartholomew is lazy.
- Toby is not actual.
- Harold is not actual.
- Kenway is not woolen.
- Atwater is not alert.
- Toby is not alert.
- Atwater is not shy.
- Kenway is shy.
- Bartholomew is actual.
- Someone is shy if and only if he is not alert.
- Someone who is not shy is always actual.
- If Byron is not actual, then Bartholomew is shy and Byron is woolen, and vice versa.
- Someone who is not alert is always both actual and lazy.
- If there is someone who is both not lazy and tiny, then Harold is shy.
- Someone who is eithor shy or alert is always not tiny.
- If someone is lazy, then he is not actual, and vice versa.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- If Harold is tiny, then Toby is not alert.
- Someone who is lazy is always both not woolen and shy.
- If there is at least one people who is not woolen, then Toby is shy and Ivan is alert.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.

**Question/Hypothesis:** Byron is alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone visits the library regularly, then they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.

**Question/Hypothesis:** If alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_distractor_1`)
**Gold Label:** `True`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone eats healthy food, they will likely have more energy. Kate eats healthy food.

**Question/Hypothesis:** Is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_distractor_1`)
**Gold Label:** `neutral`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Sebastian is not kind.
- Alden is not sorry.
- Alden is dim.
- Julian is not kind.
- Roger is not black.
- Carney is black.
- Roger is clumsy.
- Roger is not sorry.
- Julian is tasty.
- Sebastian is clumsy.
- Conway is dim.
- Conway is not kind.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Someone is not clumsy and not kind if and only if he is not tasty and not black.
- If there is at least one people who is not clumsy, then Samuel is not kind and Julian is black.
- If Conway is sorry, then Conway is not dim, and vice versa.
- Someone who is dim and not sorry is always black and tasty.
- If someone who is sorry is also not black, then he is tasty.
- Someone is not sorry if and only if he is black.
- If Conway is sorry, then Sebastian is not dim and Sebastian is kind, and vice versa.
- If there is at least one people who is sorry or not black, then Roger is dim.
- If someone is not kind, then he is both dim and clumsy.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- If someone is both sorry and not black, then he is not dim.

**Question/Hypothesis:** Samuel is not tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_distractor_1`)
**Gold Label:** `True`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone dedicates their time to exam preparations, they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Alice is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Alice is either studying for her exams or taking a break.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_distractor_1`)
**Gold Label:** `True`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone decides to engage in physical exercise, then they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If an individual experiences joy, they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.

**Question/Hypothesis:** Does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_distractor_1`)
**Gold Label:** `False`

**Premises:**
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If I manage to finish my work early, I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

## Perturbation Type: `distractor_3`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_distractor_3`)
**Gold Label:** `True`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If an individual decides to have a pet dog, it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Does this mean that Lisa has a pet dog ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If someone decides to go for a run in the park, it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** If Matt does not start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_distractor_3`)
**Gold Label:** `True`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If Adam goes for a swim, he will become tired. Consequently, he will decide to take a nap to rest.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** If adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** does this entail that no potatoes are cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_distractor_3`)
**Gold Label:** `True`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** If Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_distractor_3`)
**Gold Label:** `True`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If an individual consumes unhealthy food, weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** If Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** If Sara does not finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_distractor_3`)
**Gold Label:** `neutral`

**Premises:**
- Kenway is lazy.
- Byron is not woolen.
- Ivan is actual.
- Bartholomew is lazy.
- Toby is not actual.
- Harold is not actual.
- Kenway is not woolen.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Atwater is not alert.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Toby is not alert.
- Atwater is not shy.
- Kenway is shy.
- Bartholomew is actual.
- Someone is shy if and only if he is not alert.
- Someone who is not shy is always actual.
- If Byron is not actual, then Bartholomew is shy and Byron is woolen, and vice versa.
- Someone who is not alert is always both actual and lazy.
- If there is someone who is both not lazy and tiny, then Harold is shy.
- Someone who is eithor shy or alert is always not tiny.
- If someone is lazy, then he is not actual, and vice versa.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- If Harold is tiny, then Toby is not alert.
- Someone who is lazy is always both not woolen and shy.
- If there is at least one people who is not woolen, then Toby is shy and Ivan is alert.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.

**Question/Hypothesis:** Byron is alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If someone visits the library regularly, then they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** If alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_distractor_3`)
**Gold Label:** `True`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If someone eats healthy food, they will likely have more energy. Kate eats healthy food.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_distractor_3`)
**Gold Label:** `neutral`

**Premises:**
- Sebastian is not kind.
- Alden is not sorry.
- Alden is dim.
- Julian is not kind.
- Roger is not black.
- Carney is black.
- Roger is clumsy.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Roger is not sorry.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Julian is tasty.
- Sebastian is clumsy.
- Conway is dim.
- Conway is not kind.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Someone is not clumsy and not kind if and only if he is not tasty and not black.
- If there is at least one people who is not clumsy, then Samuel is not kind and Julian is black.
- If Conway is sorry, then Conway is not dim, and vice versa.
- Someone who is dim and not sorry is always black and tasty.
- If someone who is sorry is also not black, then he is tasty.
- Someone is not sorry if and only if he is black.
- If Conway is sorry, then Sebastian is not dim and Sebastian is kind, and vice versa.
- If there is at least one people who is sorry or not black, then Roger is dim.
- If someone is not kind, then he is both dim and clumsy.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- If someone is both sorry and not black, then he is not dim.

**Question/Hypothesis:** Samuel is not tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_distractor_3`)
**Gold Label:** `True`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If someone dedicates their time to exam preparations, they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Alice is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Alice is either studying for her exams or taking a break.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_distractor_3`)
**Gold Label:** `True`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If someone decides to engage in physical exercise, then they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If an individual experiences joy, they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_distractor_3`)
**Gold Label:** `False`

**Premises:**
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If I manage to finish my work early, I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

## Perturbation Type: `distractor_5`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_distractor_5`)
**Gold Label:** `True`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If an individual decides to have a pet dog, it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Does this mean that Lisa has a pet dog ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone decides to go for a run in the park, it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** If Matt does not start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_distractor_5`)
**Gold Label:** `True`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If Adam goes for a swim, he will become tired. Consequently, he will decide to take a nap to rest.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** If adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** does this entail that no potatoes are cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_distractor_5`)
**Gold Label:** `True`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** If Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_distractor_5`)
**Gold Label:** `True`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If an individual consumes unhealthy food, weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** If Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** If Sara does not finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_distractor_5`)
**Gold Label:** `neutral`

**Premises:**
- Kenway is lazy.
- Byron is not woolen.
- Ivan is actual.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.
- Bartholomew is lazy.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Toby is not actual.
- Harold is not actual.
- Kenway is not woolen.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Atwater is not alert.
- Toby is not alert.
- Atwater is not shy.
- Kenway is shy.
- Bartholomew is actual.
- Someone is shy if and only if he is not alert.
- Someone who is not shy is always actual.
- If Byron is not actual, then Bartholomew is shy and Byron is woolen, and vice versa.
- Someone who is not alert is always both actual and lazy.
- If there is someone who is both not lazy and tiny, then Harold is shy.
- Someone who is eithor shy or alert is always not tiny.
- A standard chess board consists of sixty-four alternating light and dark squares.
- If someone is lazy, then he is not actual, and vice versa.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- If Harold is tiny, then Toby is not alert.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- Someone who is lazy is always both not woolen and shy.
- If there is at least one people who is not woolen, then Toby is shy and Ivan is alert.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.

**Question/Hypothesis:** Byron is alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone visits the library regularly, then they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** If alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_distractor_5`)
**Gold Label:** `True`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone eats healthy food, they will likely have more energy. Kate eats healthy food.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_distractor_5`)
**Gold Label:** `neutral`

**Premises:**
- Sebastian is not kind.
- Alden is not sorry.
- Alden is dim.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.
- Julian is not kind.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- Roger is not black.
- Carney is black.
- Roger is clumsy.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- Roger is not sorry.
- Julian is tasty.
- Sebastian is clumsy.
- Conway is dim.
- Conway is not kind.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Someone is not clumsy and not kind if and only if he is not tasty and not black.
- If there is at least one people who is not clumsy, then Samuel is not kind and Julian is black.
- If Conway is sorry, then Conway is not dim, and vice versa.
- Someone who is dim and not sorry is always black and tasty.
- If someone who is sorry is also not black, then he is tasty.
- A standard chess board consists of sixty-four alternating light and dark squares.
- Someone is not sorry if and only if he is black.
- If Conway is sorry, then Sebastian is not dim and Sebastian is kind, and vice versa.
- If there is at least one people who is sorry or not black, then Roger is dim.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- If someone is not kind, then he is both dim and clumsy.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- If someone is both sorry and not black, then he is not dim.

**Question/Hypothesis:** Samuel is not tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_distractor_5`)
**Gold Label:** `True`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone dedicates their time to exam preparations, they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Alice is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Alice is either studying for her exams or taking a break.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_distractor_5`)
**Gold Label:** `True`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If someone decides to engage in physical exercise, then they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If an individual experiences joy, they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_distractor_5`)
**Gold Label:** `False`

**Premises:**
- A standard chess board consists of sixty-four alternating light and dark squares.
- Silicon dioxide occurs abundantly in nature as quartz and various types of sand.
- The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.
- A standard acoustic guitar is constructed with six resonant steel or nylon strings.
- If I manage to finish my work early, I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.
- Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

## Perturbation Type: `paraphrase`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_paraphrase`)
**Gold Label:** `True`

**Premises:**
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.

**Question/Hypothesis:** Given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Given that an individual decides to have a pet dog holds, it follows that it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.

**Question/Hypothesis:** Does this mean that Lisa has a pet dog ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Given that someone decides to go for a run in the park holds, it follows that it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.

**Question/Hypothesis:** If Matt does not start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_paraphrase`)
**Gold Label:** `True`

**Premises:**
- Given that Adam goes for a swim holds, it follows that he will become tired. Consequently, he will decide to take a nap to rest.

**Question/Hypothesis:** If adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.

**Question/Hypothesis:** does this entail that no potatoes are cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_paraphrase`)
**Gold Label:** `True`

**Premises:**
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.

**Question/Hypothesis:** If Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_paraphrase`)
**Gold Label:** `True`

**Premises:**
- Given that an individual consumes unhealthy food holds, it follows that weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.

**Question/Hypothesis:** Does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.

**Question/Hypothesis:** If Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.

**Question/Hypothesis:** If Sara does not finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_paraphrase`)
**Gold Label:** `neutral`

**Premises:**
- Kenway exhibits the quality of being lazy.
- Byron exhibits the quality of being not woolen.
- Ivan exhibits the quality of being actual.
- Bartholomew exhibits the quality of being lazy.
- Toby exhibits the quality of being not actual.
- Harold exhibits the quality of being not actual.
- Kenway exhibits the quality of being not woolen.
- Atwater exhibits the quality of being not alert.
- Toby exhibits the quality of being not alert.
- Atwater exhibits the quality of being not shy.
- Kenway exhibits the quality of being shy.
- Bartholomew exhibits the quality of being actual.
- Someone exhibits the quality of being shy if and only if he is not alert.
- Someone who is not shy is always actual.
- Whenever Byron is not actual, it is guaranteed that Bartholomew is shy and Byron is woolen, and vice versa.
- Someone who is not alert is always both actual and lazy.
- Whenever there is someone who is both not lazy and tiny, it is guaranteed that Harold is shy.
- Someone who is eithor shy or alert is always not tiny.
- Whenever someone is lazy, it is guaranteed that he is not actual, and vice versa.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- Whenever Harold is tiny, it is guaranteed that Toby is not alert.
- Someone who is lazy is always both not woolen and shy.
- Whenever there is at least one people who is not woolen, it is guaranteed that Toby is shy and Ivan is alert.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.

**Question/Hypothesis:** Byron is alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Whenever someone visits the library regularly, it is guaranteed that they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.

**Question/Hypothesis:** If alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_paraphrase`)
**Gold Label:** `True`

**Premises:**
- Given that someone eats healthy food holds, it follows that they will likely have more energy. Kate eats healthy food.

**Question/Hypothesis:** Is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_paraphrase`)
**Gold Label:** `neutral`

**Premises:**
- Sebastian exhibits the quality of being not kind.
- Alden exhibits the quality of being not sorry.
- Alden exhibits the quality of being dim.
- Julian exhibits the quality of being not kind.
- Roger exhibits the quality of being not black.
- Carney exhibits the quality of being black.
- Roger exhibits the quality of being clumsy.
- Roger exhibits the quality of being not sorry.
- Julian exhibits the quality of being tasty.
- Sebastian exhibits the quality of being clumsy.
- Conway exhibits the quality of being dim.
- Conway exhibits the quality of being not kind.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Someone exhibits the quality of being not clumsy and not kind if and only if he is not tasty and not black.
- Whenever there is at least one people who is not clumsy, it is guaranteed that Samuel is not kind and Julian is black.
- Whenever Conway is sorry, it is guaranteed that Conway is not dim, and vice versa.
- Someone who is dim and not sorry is always black and tasty.
- Whenever someone who is sorry is also not black, it is guaranteed that he is tasty.
- Someone exhibits the quality of being not sorry if and only if he is black.
- Whenever Conway is sorry, it is guaranteed that Sebastian is not dim and Sebastian is kind, and vice versa.
- Whenever there is at least one people who is sorry or not black, it is guaranteed that Roger is dim.
- Whenever someone is not kind, it is guaranteed that he is both dim and clumsy.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- Whenever someone is both sorry and not black, it is guaranteed that he is not dim.

**Question/Hypothesis:** Samuel is not tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_paraphrase`)
**Gold Label:** `True`

**Premises:**
- Given that someone dedicates their time to exam preparations holds, it follows that they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Alice is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Alice is either studying for her exams or taking a break.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_paraphrase`)
**Gold Label:** `True`

**Premises:**
- Whenever someone decides to engage in physical exercise, it is guaranteed that they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Given that an individual experiences joy holds, it follows that they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.

**Question/Hypothesis:** Does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_paraphrase`)
**Gold Label:** `False`

**Premises:**
- Given that I manage to finish my work early holds, it follows that I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.

**Question/Hypothesis:** Can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

## Perturbation Type: `negate_conclusion`

### Example 1 (`mle:d1_Data:pl:HS:1__pert_negate_conclusion`)
**Gold Label:** `False`

**Premises:**
- In chemistry, if a solution has a pH below 7, it is acidic. Also, if a solution is acidic, it will turn litmus paper red.

**Question/Hypothesis:** It is false that given a solution has a pH below 7, will it turn litmus paper red?

---

### Example 2 (`lb-bqa:first_order_logic:modus_tollens:7:1__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- If an individual decides to have a pet dog, it becomes their responsibility to provide proper care for their beloved companion. However, in this particular scenario, it appears that Lisa is not interested in taking care.

**Question/Hypothesis:** It is false that does this mean that Lisa has a pet dog ?

---

### Example 3 (`lb-bqa:first_order_logic:constructive_dilemma:5:3__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- If someone decides to go for a run in the park, it is highly likely that their friends will join them and run alongside. Similarly, if someone chooses to dance at the club, it is very probable that their family members will also join in the fun and dance to the music. We have knowledge that at least one of the following statements is true: either John is currently running in the park or Mary is currently dancing in the club. It is uncertain which statement is accurate, as it could be that only John is running, or it could be that only Mary is dancing, or even the possibility that both are true.

**Question/Hypothesis:** It is false that can we say at least one of the following must always be true? (a) his friends won't join in and (b) her family won't join in

---

### Example 4 (`mle:d2_Data:pl:HS_MT:11__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- Matt wanted to start his own web development business. He realized that if he learned web development, he could build interactive websites for clients. Matt knew that if he could build websites for others, he could start his own successful business.

**Question/Hypothesis:** If Matt does start his own business, then does he learn web development?

---

### Example 5 (`lb-bqa:propositional_logic:hypothetical_syllogism:2:0__pert_negate_conclusion`)
**Gold Label:** `False`

**Premises:**
- If Adam goes for a swim, he will become tired. Consequently, he will decide to take a nap to rest.

**Question/Hypothesis:** It is false that if adam goes for a swim, does this entail that he will take a nap?

---

### Example 6 (`lb-bqa:propositional_logic:constructive_dilemma:11:2__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- Lisa had a few options for her afternoon activities. She could either go to the gym, which would then lead her to the swimming pool, or she could go to the library, which would then take her to the coffee shop. It was certain that she would choose at least one of these options, but it was unclear which one. It could be that she only went to the gym, or only to the library, or maybe she would even visit both places.

**Question/Hypothesis:** It is false that can we say at least one of the following must always be true? (a) she will go to the swimming pool and (b) she will not go to the coffee shop

---

### Example 7 (`lb-bqa:propositional_logic:bidirectional_dilemma:18:3__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- I am determined to stay fit so I have made a rule for myself: If I go to the gym, then I will stay fit. Additionally, I know that if I don't exercise, then I will gain weight. Today, I find myself in a situation where I have to decide whether to go to the gym or not, but there is uncertainty about the outcome. At least one of the following is true: either I go to the gym, or I won't gain weight. It is unclear which one will actually happen. It could be the case that only I go to the gym, or only I won't gain weight, or perhaps both will come true.

**Question/Hypothesis:** It is false that can we say at least one of the following must always be true? (a) i won't stay fit and (b) i exercise

---

### Example 8 (`lb-bqa:nm_logic:default_reasoning_default:18:1__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- Carrots and potatoes are known for their vibrant orange color. Typically, orange vegetables are prepared by cooking them. However, carrots are a possible exception to this general rule.

**Question/Hypothesis:** does this entail that no potatoes are not cooked?

---

### Example 9 (`mle:d2_Data:pl:HS_MP:12__pert_negate_conclusion`)
**Gold Label:** `False`

**Premises:**
- Luis wanted to travel to France. He learned that if he studied French regularly, he would become fluent in the language. When he became fluent, Luis knew he could comfortably travel to France and communicate there.

**Question/Hypothesis:** It is false that if Luis studies French, does he travel to France?

---

### Example 10 (`lb-bqa:first_order_logic:modus_ponens:15:0__pert_negate_conclusion`)
**Gold Label:** `False`

**Premises:**
- If an individual consumes unhealthy food, weight gain is likely to occur. One specific case in which this applies is when a person named John has been eating unhealthy food.

**Question/Hypothesis:** It is false that does this mean that he will gain weight?

---

### Example 11 (`mle:d2_Data:fol:HS_MT:6__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- Some people want to learn how to bake delicious cakes. They know that if they follow baking recipes carefully, the cakes will turn out tasty. And if the cakes turn out tasty, their friends will compliment their baking skills.

**Question/Hypothesis:** It is false that if Mary's cakes do not receive compliments from friends, does she follow baking recipes carefully?

---

### Example 12 (`mle:d2_Data:pl:HS_MT:9__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- Sara wanted to run a marathon this year. She learned that if she trained consistently, her endurance would improve. With better endurance, Sara knew she could finish the lengthy marathon race.

**Question/Hypothesis:** If Sara does finish the race, then does she train for the marathon?

---

### Example 13 (`nli:language:95:18__pert_negate_conclusion`)
**Gold Label:** `neutral`

**Premises:**
- Kenway is lazy.
- Byron is not woolen.
- Ivan is actual.
- Bartholomew is lazy.
- Toby is not actual.
- Harold is not actual.
- Kenway is not woolen.
- Atwater is not alert.
- Toby is not alert.
- Atwater is not shy.
- Kenway is shy.
- Bartholomew is actual.
- Someone is shy if and only if he is not alert.
- Someone who is not shy is always actual.
- If Byron is not actual, then Bartholomew is shy and Byron is woolen, and vice versa.
- Someone who is not alert is always both actual and lazy.
- If there is someone who is both not lazy and tiny, then Harold is shy.
- Someone who is eithor shy or alert is always not tiny.
- If someone is lazy, then he is not actual, and vice versa.
- It can be concluded that Kenway is not tiny once knowing that Toby is woolen or Byron is not actual.
- If Harold is tiny, then Toby is not alert.
- Someone who is lazy is always both not woolen and shy.
- If there is at least one people who is not woolen, then Toby is shy and Ivan is alert.
- It can be concluded that Kenway is not actual once knowing that Bartholomew is not lazy and Bartholomew is not alert.

**Question/Hypothesis:** Byron is not alert.

---

### Example 14 (`lb-bqa:first_order_logic:hypothetical_syllogism:18:2__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- If someone visits the library regularly, then they will have an improved understanding of the subject. If they will have an improved understanding of the subject, then they will be likely to get better grades.

**Question/Hypothesis:** It is false that if alice visits the library regularly , does this imply that she won't be likely to get better grades?

---

### Example 15 (`mle:d1_Data:fol:MP:1__pert_negate_conclusion`)
**Gold Label:** `False`

**Premises:**
- If someone eats healthy food, they will likely have more energy. Kate eats healthy food.

**Question/Hypothesis:** It is false that is Kate likely to have more energy?

---

### Example 16 (`nli:language:42:8__pert_negate_conclusion`)
**Gold Label:** `neutral`

**Premises:**
- Sebastian is not kind.
- Alden is not sorry.
- Alden is dim.
- Julian is not kind.
- Roger is not black.
- Carney is black.
- Roger is clumsy.
- Roger is not sorry.
- Julian is tasty.
- Sebastian is clumsy.
- Conway is dim.
- Conway is not kind.
- Someone being both not dim and clumsy is equivalent to being not kind.
- Someone is not clumsy and not kind if and only if he is not tasty and not black.
- If there is at least one people who is not clumsy, then Samuel is not kind and Julian is black.
- If Conway is sorry, then Conway is not dim, and vice versa.
- Someone who is dim and not sorry is always black and tasty.
- If someone who is sorry is also not black, then he is tasty.
- Someone is not sorry if and only if he is black.
- If Conway is sorry, then Sebastian is not dim and Sebastian is kind, and vice versa.
- If there is at least one people who is sorry or not black, then Roger is dim.
- If someone is not kind, then he is both dim and clumsy.
- Sebastian being not kind implies that Sebastian is dim and Sebastian is black.
- If someone is both sorry and not black, then he is not dim.

**Question/Hypothesis:** Samuel is tasty.

---

### Example 17 (`lb-bqa:first_order_logic:constructive_dilemma:15:0__pert_negate_conclusion`)
**Gold Label:** `False`

**Premises:**
- If someone dedicates their time to exam preparations, they are likely to achieve favorable results. Conversely, individuals who take breaks tend to enjoy the company of friends. It is known that either Alice is intensely focused on studying for her exams or she is taking a break. The specific case for this context concludes that Alice is either studying for her exams or taking a break.

**Question/Hypothesis:** It is false that can we say at least one of the following must always be true? (a) she will get good grades. and (b) she will spend time with her friends.

---

### Example 18 (`lb-bqa:first_order_logic:constructive_dilemma:13:0__pert_negate_conclusion`)
**Gold Label:** `False`

**Premises:**
- If someone decides to engage in physical exercise, then they are likely to follow it up with a leisurely movie-watching session. Similarly, should they opt to visit a shopping center, they will probably indulge in some retail therapy. It is worth noting that at least one of the statements holds true: either the person named John went to the gym, or he ventured into the mall. The specifics of these options remain unknown; it could be the case that only the former is accurate, or solely the latter, or perhaps even both are accurate.

**Question/Hypothesis:** It is false that can we say at least one of the following must always be true? (a) he will watch a movie and (b) he will go shopping

---

### Example 19 (`lb-bqa:first_order_logic:modus_ponens:5:1__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- If an individual experiences joy, they have a tendency to dine out in the evening. Earlier today, Reema experienced joy.

**Question/Hypothesis:** It is false that does this imply that she will not go for dinner outside?

---

### Example 20 (`lb-bqa:propositional_logic:bidirectional_dilemma:15:2__pert_negate_conclusion`)
**Gold Label:** `True`

**Premises:**
- If I manage to finish my work early, I will treat myself to a movie. On the other hand, if I can wrap up my dinner early, I'll opt to stay at home. It's important to note that at least one of the following statements is true: either I finish my work early, or I decide not to stay at home. The intriguing part is that I'm unsure which of these statements is accurate - it could be that only the first statement is true, only the second one is true, or perhaps even both statements are true.

**Question/Hypothesis:** It is false that can we say at least one of the following must always be true? (a) i will go to the movies  and (b) i finish my dinner early

---

