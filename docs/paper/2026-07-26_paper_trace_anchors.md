# 논문 Results 정성 trace anchors — Base / GT-only / Cand.-CE / EGO

공통 covered ∧ non-malformed 모집단 **n=915** · 신규 추론 0

## A_discriminate — n=35
필터: `WM top-1 ✗ ∧ Base ✗ ∧ GT-only ✗ ∧ Cand.-CE ✓ ∧ EGO ✓`

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_172` · GT **add rice** (WM 7위) · WM top-1 `stir ingredient` · 이력 16
- 후보: add salt, add ingredient, check heat, make dough, flip ingredient, add rice, stir ingredient, stir dough, add oil, add spice_(seasoning)
- 이력(최근 6): add water → mix_(beat,_blend,_whisk) ingredient → wash spoon → add rice → add rice → add rice

**Base** → `add ingredient` ✗ (68단어)
- belief: Transferring cooked food to a serving container
- reasoning: I am currently holding a ladle with cooked food and have just removed it from the pan. The next logical step in preparing this dish is to transfer the cooked food to a serving plate or bowl. Looking at the candidates, 'add ingredient' is the most plausible as it implies adding something to the dish, which aligns with the action of transferring the cooked food to another container.
- [이력 제거] → `add ingredient` ✗ (68단어) · belief: Transferring cooked food to a container
- [이력 제거] reasoning: I am currently holding a ladle with cooked food and have just lifted it from the pan. The next logical step is to transfer the cooked food to a plate or container for serving or further processing. Looking at the candidates, "add ingredient" is the most plausible as it implies adding something to the food, which aligns with the action of transferring the cooked food to another container.

**GT-only** → `stir ingredient` ✗ [echo] (45단어)
- belief: stir ingredient
- reasoning: The current action involves stirring the mixture in the pan, which is a continuation of the cooking process. The candidate 'stir ingredient' matches this activity, as it directly corresponds to the action of mixing the contents in the pan to ensure even cooking and consistency.
- [이력 제거] → `stir ingredient` ✗ (118단어) · belief: Stirring the batter to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring a yellow batter in a pan on the stove, which suggests they are preparing a dish that requires consistent mixing. The batter appears to be thick and is being stirred with a ladle, indicating that the cooking process is ongoing. The presence of a second pot with a similar batter suggests that the person might be preparing multiple portions or different types of dishes. The action of stirring is crucial to ensure even cooking and prevent burning. The other candidates, such as adding ingredients or checking heat, are not directly related to the current action of stirring. The most logical next step is to continue stirring the batter to ensure it cooks evenly.

**Cand.-CE** → `add rice` ✓ [근거언급] (53단어)
- belief: Adding rice to the pan as part of the cooking process.
- reasoning: The current action involves adding rice to the pan, which is part of the ongoing cooking process. The previous actions show a pattern of adding rice multiple times, indicating it's a key ingredient being incorporated. The candidate 'add rice' aligns with this established pattern and the visual context of the pan and ingredients.
- [이력 제거] → `stir ingredient` ✗ (110단어) · belief: Cooking a flatbread by stirring the batter in the pan.
- [이력 제거] reasoning: The person is currently cooking a dish that appears to be a type of flatbread or pancake, as indicated by the circular shape being formed in the pan. The hand is holding a ladle with batter, suggesting the action of pouring or spreading the batter is ongoing. The other hand is holding a small piece of food, possibly a garnish or a piece of the cooked item. Given the context, the most logical next step is to continue cooking by stirring or spreading the batter to ensure even cooking. The action "stir ingredient" fits this context, as it implies continuing the cooking process by stirring the contents of the pan.

**EGO** → `add rice` ✓ (89단어)
- belief: I am in the middle of preparing the mixture for shaping or portioning.
- reasoning: I am holding a small portion of the prepared mixture in my hand, having just lifted it from the pan. The ladle is still in the pan, and the mixture appears to be at a stage where it needs to be shaped or portioned. The bowl of yellow rice is nearby, suggesting the next step involves incorporating or preparing the rice. Among the candidates, adding rice is the only one that aligns with the current state of holding a portion of mixture and the presence of the rice bowl.

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_180` · GT **add rice** (WM 9위) · WM top-1 `add ingredient` · 이력 24
- 후보: add batter, cover item, check heat, add ingredient, add oil, stir ingredient, add rice, turn stovetop, add spice_(seasoning), mix_(beat,_blend,_whisk) ingredient
- 이력(최근 6): add rice → add rice → add rice → add rice → add rice → add rice

**Base** → `stir ingredient` ✗ (59단어)
- belief: Stir the rice in the pan to cook it evenly.
- reasoning: I am currently stirring the yellow rice in the pan on the stove with my left hand while holding a container of batter over another pot with my right hand. The rice appears to be cooking and needs to be stirred to prevent sticking. The most logical next step is to continue stirring the rice to ensure even cooking.
- [이력 제거] → `stir ingredient` ✗ (79단어) · belief: Stirring the contents in the pan to ensure even cooking.
- [이력 제거] reasoning: I am currently stirring the yellow substance in the pan on the left burner while simultaneously holding a container over the pot on the right burner. The action of stirring the pan is ongoing, and the container being held suggests I am preparing to add something to the pot. Given the context of cooking and the current state of the pan, the most logical next step is to continue stirring the contents in the pan to ensure even cooking.

**GT-only** → `stir ingredient` ✗ [echo] (49단어)
- belief: stir ingredient
- reasoning: The current action involves adding rice to the pan, which is part of the cooking process. The next logical step, given the context of cooking and the need to ensure even cooking, is to stir the ingredient in the pan. This will help distribute the heat and prevent burning.
- [이력 제거] → `stir ingredient` ✗ (71단어) · belief: Stirring the cooking mixture in the pan.
- [이력 제거] reasoning: The person is currently stirring a yellow mixture in a pan on the stove, which appears to be a cooking process. The other hand is holding a container, possibly adding or preparing to add more ingredients. Given the context of cooking and the action of stirring, the most logical next step is to continue stirring the mixture to ensure even cooking and prevent burning. The candidate "stir ingredient" matches this action.

**Cand.-CE** → `add rice` ✓ (61단어)
- belief: Adding rice to the pan to complete the cooking process.
- reasoning: The current action involves adding rice to the pan, which is part of the cooking process. The repeated addition of rice suggests that the task is nearing completion or requires a final addition. The most logical next step, given the context and the need to continue the cooking process, is to add more rice to ensure the dish is properly prepared.
- [이력 제거] → `add batter` ✗ (74단어) · belief: Adding batter to the pan to continue cooking.
- [이력 제거] reasoning: The person is actively cooking, stirring a yellow mixture in a pan on the stove with one hand while holding a container of batter over another pot with the other hand. The batter appears to be being prepared for addition to the pan. Given the context of cooking and the current state of the ingredients, the most logical next step is to add the prepared batter to the pan to continue the cooking process.

**EGO** → `add rice` ✓ (74단어)
- belief: I am in the middle of adding rice to the pot on the stove.
- reasoning: I am continuing to add rice to the pot on the stove, as indicated by the repeated action and the visible pile of uncooked rice in the bowl. The hand holding the spoon is positioned over the pot, and the motion suggests another portion is being added. The other candidates, like stirring or adding oil, are not supported by the current visual context or the recent history of actions, which consistently involve adding rice.

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_186` · GT **add rice** (WM 6위) · WM top-1 `add ingredient` · 이력 30
- 후보: cover item, serve dish, add rice, turn stovetop, check heat, add spice_(seasoning), stir ingredient, add ingredient, add oil, mix_(beat,_blend,_whisk) ingredient
- 이력(최근 6): add rice → add rice → add rice → add rice → add rice → add rice

**Base** → `add spice_(seasoning)` ✗ [근거언급] (55단어)
- belief: Adding seasoning to the rice in the pan.
- reasoning: The person has been repeatedly adding rice to the pan and is currently stirring it. The next logical step in cooking rice is to add seasoning or spice to enhance flavor. Among the candidates, 'add spice_(seasoning)' is the most appropriate next action given the context of ongoing cooking and the need to season the rice.
- [이력 제거] → `check heat` ✗ (83단어) · belief: Monitoring the heat of the stove while cooking the grains.
- [이력 제거] reasoning: The person is actively stirring the yellow grains in the pan with a spatula, indicating they are cooking and need to monitor the heat or adjust the cooking process. The other hand is holding a small container, possibly with seasoning or spice, suggesting they might be about to add it. However, the immediate action is to check the heat to ensure the cooking is proceeding correctly. The most logical next step is to check the heat of the stove to adjust if needed.

**GT-only** → `check heat` ✗ (58단어)
- belief: monitoring the cooking process to ensure even cooking
- reasoning: The person is currently stirring the rice in the pan, which suggests they are monitoring the cooking process. Given the repetitive nature of adding rice and the current action of stirring, it is logical to assume they are checking the heat or adjusting the cooking process to ensure even cooking. The action 'check heat' aligns with this reasoning.
- [이력 제거] → `add ingredient` ✗ (131단어) · belief: Adding an ingredient to the cooking mixture.
- [이력 제거] reasoning: The person is currently stirring the yellow ingredient in the pan with a spoon, indicating they are actively cooking. The hand holding the spoon is moving in a circular motion, which is typical for stirring to ensure even cooking. The other hand is holding a small container, possibly containing an additional ingredient to be added. Given the context of cooking and the action of stirring, the most logical next step is to continue stirring or to add another ingredient. However, since the action of stirring is already in progress, the next logical step would be to add another ingredient or to check the heat. The candidate action "add ingredient" is the most appropriate as it aligns with the ongoing cooking process and the presence of the small container in the hand.

**Cand.-CE** → `add rice` ✓ [근거언급] (78단어)
- belief: Continue adding rice to the pan.
- reasoning: The person has been repeatedly adding rice to the pan for a while, and the current state shows the rice is still being stirred in the pan. The most logical next step, given the context of cooking rice, is to continue adding more rice to ensure the desired quantity is reached before other steps like stirring or seasoning. The other options like stirring or adding spices are not the immediate priority since the rice is still being added.
- [이력 제거] → `add spice_(seasoning)` ✗ (73단어) · belief: Adding spice to the cooking grains.
- [이력 제거] reasoning: The person is currently stirring the yellow grains in the pan with a spoon, indicating they are actively cooking. The right hand is holding a small container, possibly with seasoning or spice, suggesting they might be about to add it. Given the context of cooking and the current action of stirring, the most logical next step is to add an ingredient, likely the spice or seasoning from the container in the right hand.

**EGO** → `add rice` ✓ [근거언급] (88단어)
- belief: I am in the middle of adding rice to the pan.
- reasoning: I have been adding rice to the pan repeatedly, and the current state shows the rice is still being added. The hand holding the ladle is positioned over the pan, and the other hand is holding a container, suggesting the action of adding more rice is ongoing. Among the candidates, “add rice” is the only one that matches the visual evidence of the ongoing action and the recent history of adding rice. The other options, like stirring or checking heat, are not supported by the current visual context.

## B_retain — n=54
필터: `WM top-1 ✓ ∧ GT-only ✗ ∧ EGO ✓`

### `0e0d6704-1c6c-4a62-bc97-cc55658cf8ac_362` · GT **add ingredient** (WM 1위) · WM top-1 `add ingredient` · 이력 24
- 후보: measure ingredient, wash hand, add ingredient, add flour, put_(place,_set) dough, mix_(beat,_blend,_whisk) ingredient, organize_(arrange) tool_(utensil), store ingredient, put_(place,_set) tray, roll dough
- 이력(최근 6): organize_(arrange) sheet → sprinkle flour → knead dough → operate mixer → add ingredient → dispose item

**Base** → `add ingredient` ✓ (59단어)
- belief: Add the ingredient from the container to the mixer.
- reasoning: I have just disposed of an item and am now holding a small white container, likely containing an ingredient. The mixer is running, and I need to add something to it. Given the context of preparing dough and the presence of the container, the most logical next step is to add the ingredient from the container into the mixer.
- [이력 제거] → `add ingredient` ✓ (71단어) · belief: Adding an ingredient to the dough mixture
- [이력 제거] reasoning: I am holding a small bag of what appears to be flour or a similar dry ingredient. I have just taken it from the machine and am now preparing to add it to the mixing area. The machine is still running, and I need to add this ingredient to the dough before proceeding with the next steps. The most logical next action is to add this ingredient to the mixing area.

**GT-only** → `add flour` ✗ (56단어)
- belief: add flour to dough
- reasoning: The current action involves handling dough and ingredients, and the environment suggests ongoing preparation. The most logical next step is to add more flour to the dough, as this is a common step in dough preparation to ensure proper consistency and texture. The candidate 'add flour' aligns with this need and the context of the task.
- [이력 제거] → `add flour` ✗ (62단어) · belief: Add flour to the dough mixer.
- [이력 제거] reasoning: The current action involves handling a bag of flour, which suggests preparation for baking or dough-making. The machine appears to be a dough mixer, and the presence of flour indicates that the next step is likely to involve adding flour to the mixer. The action of "add flour" directly aligns with the current context and the typical workflow in a bakery setting.

**Cand.-CE** → `add ingredient` ✓ (46단어)
- belief: Add ingredients to the mixer.
- reasoning: I have just disposed of an item and am now standing near the mixer. The mixer is currently operating, and I need to continue preparing the dough. The most logical next step is to add more ingredients to the mixer to proceed with the dough-making process.
- [이력 제거] → `add flour` ✗ (50단어) · belief: Add flour to the dough mixer.
- [이력 제거] reasoning: I am currently holding a bag of flour and standing near the dough mixer. The mixer is active, and I need to add flour to it as part of the dough-making process. The most logical next step is to add the flour to the mixer to continue preparing the dough.

**EGO** → `add ingredient` ✓ (79단어)
- belief: I am in the middle of preparing dough by adding ingredients to the mixer.
- reasoning: I have just finished kneading dough and am holding a small container, likely containing an ingredient. The mixer is still running, and I’m positioned near the work area with tools and trays visible. Among the candidates, adding an ingredient is the only action that logically follows the current state of having just kneaded dough and being in possession of a container. The other options, like washing hands or organizing tools, are less immediately relevant given the ongoing dough preparation.

### `13c76616-f168-4af0-8d2a-fe82ce232d6a_506` · GT **check heat** (WM 1위) · WM top-1 `check heat` · 이력 4
- 후보: store ingredient, put_(place,_set) dough, knead dough, add salt, sieve ingredient, add water, check heat, add rice, add ingredient, make dough
- 이력(최근 6): cook rice → check heat → cook rice → check heat

**Base** → `add water` ✗ [근거언급] (68단어)
- belief: Prepare the grains for cooking by adding water.
- reasoning: I am currently grinding or processing grains in a large bowl, which suggests I am preparing ingredients for cooking. Given the context of previous actions like 'cook rice' and 'check heat', and the current activity, the next logical step is to add water to the grains to begin the cooking process. The other options like 'add salt' or 'make dough' are less relevant to the current grinding activity.
- [이력 제거] → `add water` ✗ (81단어) · belief: Adding water to the grains to prepare them for cooking or processing.
- [이력 제거] reasoning: I am currently mixing grains in a large bowl with a wooden tool, which suggests I am preparing a base ingredient for cooking or baking. The action of mixing grains is typically followed by adding other ingredients or adjusting the consistency. Since I am not yet adding any other components, the most logical next step is to add water to help mix or cook the grains. This aligns with the common practice of adding water to grains before cooking or processing.

**GT-only** → `add water` ✗ (55단어)
- belief: Add water to the rice to begin cooking.
- reasoning: The current action involves checking the heat, which is a step in the cooking process. The next logical step after checking heat is to add water to the rice to begin cooking it. This aligns with the typical sequence of cooking rice, where water is added after the heat is checked to ensure proper cooking.
- [이력 제거] → `add salt` ✗ (64단어) · belief: Adding salt to the grain mixture.
- [이력 제거] reasoning: The current action involves mixing or processing grains in a large bowl, which is a preparatory step for cooking or baking. The context suggests that the next step would be to add an ingredient to this mixture, likely to enhance flavor or texture. Among the candidates, 'add salt' is a common and logical next step in many recipes, especially when preparing grains or dough.

**Cand.-CE** → `check heat` ✓ [근거언급] (51단어)
- belief: Continue monitoring the heat while cooking rice.
- reasoning: The current action involves checking the heat, which is part of the cooking process. The previous actions indicate a repetitive cycle of cooking and checking heat, suggesting that the next logical step is to continue monitoring the heat to ensure proper cooking. This aligns with the ongoing task of cooking rice.
- [이력 제거] → `add salt` ✗ (67단어) · belief: Adding salt to the mixed grains.
- [이력 제거] reasoning: The current action involves mixing grains in a large bowl, which is a preparatory step for cooking or making a dish. The context suggests that the next step would be to add an ingredient to this mixture, likely to enhance flavor or texture. Among the candidates, "add salt" is a common and logical next step after mixing grains, as salt is a fundamental seasoning in many recipes.

**EGO** → `check heat` ✓ (65단어)
- belief: I am in the middle of monitoring the rice as it cooks.
- reasoning: I have just finished stirring the rice in the pot and am now observing the contents. The rice appears to be evenly mixed and the pot is still on the heat source. Given the recent action of checking heat and the current state of the rice, the most logical next step is to assess the cooking progress again to ensure it’s not overcooked or undercooked.

### `1558e9f2-d7f3-4a23-9627-4240f506d7df_568` · GT **roll dough** (WM 1위) · WM top-1 `roll dough` · 이력 22
- 후보: roll dough, coat dough, sprinkle flour, put_(place,_set) dough, stir ingredient, add topping, cut dough, shape dough, press dough, make dough
- 이력(최근 6): flip ingredient → add ingredient → roll dough → add ingredient → roll dough → flip ingredient

**Base** → `shape dough` ✗ [근거언급] (59단어)
- belief: Shape the dough after rolling and adding ingredients.
- reasoning: I have been repeatedly rolling and flipping the dough, and just added an ingredient. The dough is now on the table, and I need to prepare it for the next step. Since I’ve been rolling it, and the dough appears ready, the logical next step is to shape it, which aligns with the current state and my previous actions.
- [이력 제거] → `press dough` ✗ (78단어) · belief: Preparing the dough by pressing it flat on the table.
- [이력 제거] reasoning: I am currently working with dough on the table, and my hands are actively pressing and shaping it. The flour container is nearby, suggesting I might need to dust the dough or surface. Since I've already placed the dough on the table, the next logical step is to prepare it for rolling or further shaping. The action "press dough" aligns with what I'm currently doing to ensure the dough is evenly spread and ready for the next step.

**GT-only** → `shape dough` ✗ [근거언급] (49단어)
- belief: Shape the dough into the desired form.
- reasoning: The dough has been rolled out and is now being shaped into a specific form, likely for a dish. The previous actions involved rolling and flipping the dough, indicating a preparation process. The current action is focused on shaping the dough, which is a logical next step after rolling.
- [이력 제거] → `press dough` ✗ (74단어) · belief: shaping the dough on the table
- [이력 제거] reasoning: The hands are actively working with the dough on the table, indicating a shaping or pressing action is underway. The rolling pin is nearby but not in use, suggesting the focus is on forming the dough rather than rolling it out. The presence of flour suggests it might be used for dusting, but the immediate action is on the dough itself. The most logical next step is to continue shaping or pressing the dough.

**Cand.-CE** → `press dough` ✗ (60단어)
- belief: Shape the dough by pressing it to achieve the desired form.
- reasoning: The dough has been rolled multiple times and flipped with ingredients added, indicating a repetitive preparation process. The current state shows the dough is ready for the next step, which is likely to involve shaping or pressing it to achieve the desired form. The action 'press dough' aligns with the ongoing task of preparing the dough for the next stage.
- [이력 제거] → `press dough` ✗ (64단어) · belief: Shaping the dough on the table.
- [이력 제거] reasoning: I am currently working with dough on the table, and my hands are actively shaping it. The rolling pin and flour container are nearby, indicating I am in the process of preparing or working with the dough. Given the context, the most logical next step is to continue shaping the dough, which aligns with the action of pressing it to achieve the desired form.

**EGO** → `roll dough` ✓ [근거언급] (70단어)
- belief: I am in the middle of preparing the dough by rolling it out.
- reasoning: I have been repeatedly flipping and rolling the dough, and the current state shows the dough is still on the table with the rolling pin nearby. The action history indicates a pattern of preparing the dough through rolling, and the rolling pin is positioned for continued use. Among the candidates, rolling dough is the only one that directly follows from the current state and matches the established pattern of preparation.

## B_retain_strict — n=10
필터: `B ∧ Base ✗ ∧ Cand.-CE ✓`

### `13c76616-f168-4af0-8d2a-fe82ce232d6a_506` · GT **check heat** (WM 1위) · WM top-1 `check heat` · 이력 4
- 후보: store ingredient, put_(place,_set) dough, knead dough, add salt, sieve ingredient, add water, check heat, add rice, add ingredient, make dough
- 이력(최근 6): cook rice → check heat → cook rice → check heat

**Base** → `add water` ✗ [근거언급] (68단어)
- belief: Prepare the grains for cooking by adding water.
- reasoning: I am currently grinding or processing grains in a large bowl, which suggests I am preparing ingredients for cooking. Given the context of previous actions like 'cook rice' and 'check heat', and the current activity, the next logical step is to add water to the grains to begin the cooking process. The other options like 'add salt' or 'make dough' are less relevant to the current grinding activity.
- [이력 제거] → `add water` ✗ (81단어) · belief: Adding water to the grains to prepare them for cooking or processing.
- [이력 제거] reasoning: I am currently mixing grains in a large bowl with a wooden tool, which suggests I am preparing a base ingredient for cooking or baking. The action of mixing grains is typically followed by adding other ingredients or adjusting the consistency. Since I am not yet adding any other components, the most logical next step is to add water to help mix or cook the grains. This aligns with the common practice of adding water to grains before cooking or processing.

**GT-only** → `add water` ✗ (55단어)
- belief: Add water to the rice to begin cooking.
- reasoning: The current action involves checking the heat, which is a step in the cooking process. The next logical step after checking heat is to add water to the rice to begin cooking it. This aligns with the typical sequence of cooking rice, where water is added after the heat is checked to ensure proper cooking.
- [이력 제거] → `add salt` ✗ (64단어) · belief: Adding salt to the grain mixture.
- [이력 제거] reasoning: The current action involves mixing or processing grains in a large bowl, which is a preparatory step for cooking or baking. The context suggests that the next step would be to add an ingredient to this mixture, likely to enhance flavor or texture. Among the candidates, 'add salt' is a common and logical next step in many recipes, especially when preparing grains or dough.

**Cand.-CE** → `check heat` ✓ [근거언급] (51단어)
- belief: Continue monitoring the heat while cooking rice.
- reasoning: The current action involves checking the heat, which is part of the cooking process. The previous actions indicate a repetitive cycle of cooking and checking heat, suggesting that the next logical step is to continue monitoring the heat to ensure proper cooking. This aligns with the ongoing task of cooking rice.
- [이력 제거] → `add salt` ✗ (67단어) · belief: Adding salt to the mixed grains.
- [이력 제거] reasoning: The current action involves mixing grains in a large bowl, which is a preparatory step for cooking or making a dish. The context suggests that the next step would be to add an ingredient to this mixture, likely to enhance flavor or texture. Among the candidates, "add salt" is a common and logical next step after mixing grains, as salt is a fundamental seasoning in many recipes.

**EGO** → `check heat` ✓ (65단어)
- belief: I am in the middle of monitoring the rice as it cooks.
- reasoning: I have just finished stirring the rice in the pot and am now observing the contents. The rice appears to be evenly mixed and the pot is still on the heat source. Given the recent action of checking heat and the current state of the rice, the most logical next step is to assess the cooking progress again to ensure it’s not overcooked or undercooked.

### `3bd5bf35-d6ac-43b2-ab75-1558a37c8550_1209` · GT **fill bag** (WM 1위) · WM top-1 `fill bag` · 이력 18
- 후보: stir bread, wash hand, fill bag, make filling, add ingredient, add salt, squeeze bag, strain liquid, knead dough, add grain
- 이력(최근 6): squeeze bag → stir bread → fill bag → squeeze bag → fill bag → squeeze bag

**Base** → `stir bread` ✗ [근거언급] (82단어)
- belief: Stirring the dough to ensure it is well-mixed and ready for the next step.
- reasoning: I am currently stirring the dough in the plate, which is part of the ongoing process of preparing the mixture. The previous actions involved filling and squeezing bags, suggesting a repetitive task. Since I am actively mixing the dough, the next logical step is to continue this action to ensure the dough is well-prepared before proceeding to the next stage, which might involve filling or squeezing bags again. The candidate "stir bread" matches this context, as it implies continuing the mixing process.
- [이력 제거] → `knead dough` ✗ (96단어) · belief: Mixing a substance in a bowl.
- [이력 제거] reasoning: The person is actively mixing a white substance in a bowl with their right hand, while holding the bowl steady with their left. This action is consistent with stirring or kneading a mixture. Among the candidates, "stir bread" and "knead dough" are relevant, but "stir bread" is less likely as the substance doesn't appear to be bread dough. "Knead dough" is a more general term that fits the action of mixing a substance in a bowl. The other candidates involve different actions like washing hands, filling bags, or adding ingredients, which are not currently being performed.

**GT-only** → `knead dough` ✗ [echo] [근거언급] (64단어)
- belief: knead dough
- reasoning: The current action involves stirring or mixing a white substance in a bowl, which is likely part of the dough-making process. The previous actions indicate a repetitive cycle of filling and squeezing bags, suggesting the dough is being prepared for packaging. The most logical next step, given the context of preparing dough, is to continue kneading or mixing it to achieve the desired consistency.
- [이력 제거] → `knead dough` ✗ (65단어) · belief: Kneading the dough to achieve the desired consistency.
- [이력 제거] reasoning: The person is currently mixing a white substance in a bowl, which appears to be a dough or batter. The action of stirring is ongoing, and the context suggests this is part of a food preparation process. Among the candidates, "knead dough" is the most appropriate next step, as it directly follows the current mixing action and aligns with the visual evidence of dough preparation.

**Cand.-CE** → `fill bag` ✓ [근거언급] (65단어)
- belief: Continue the cycle of filling and squeezing the bag.
- reasoning: The current action is continuing the repetitive cycle of filling and squeezing the bag, which has been the pattern for the last several steps. The dough is being stirred, indicating preparation for another batch or continuation of the current task. The most logical next step is to continue the established pattern of filling the bag, as the previous actions suggest this is the ongoing subgoal.
- [이력 제거] → `add ingredient` ✗ (71단어) · belief: Adding an ingredient to the mixture being prepared.
- [이력 제거] reasoning: The person is actively mixing a white substance in a bowl, which suggests they are preparing a mixture, likely for cooking or baking. The action of stirring is ongoing, and the context implies they are in the middle of a preparation process. Among the candidates, "add ingredient" is the most logical next step as it aligns with the ongoing mixing activity and the typical sequence of adding components to a mixture.

**EGO** → `fill bag` ✓ (81단어)
- belief: I am in the middle of filling the bag with dough mixture.
- reasoning: I have just finished squeezing the bag and am holding the plate with the dough mixture. The dough is still in the plate, and my hands are covered in it, indicating I’m not yet done with this step. The bag is empty and ready to be refilled, and the dough is still in the plate, suggesting I need to continue the process of filling and squeezing. The most logical next step is to fill the bag again with the dough mixture.

### `3bd5bf35-d6ac-43b2-ab75-1558a37c8550_1260` · GT **fill bag** (WM 1위) · WM top-1 `fill bag` · 이력 69
- 후보: add paste, check heat, add ingredient, mix_(beat,_blend,_whisk) ingredient, make filling, stir bread, add grain, add salt, knead dough, fill bag
- 이력(최근 6): fill bag → squeeze bag → fill bag → squeeze bag → fill bag → squeeze bag

**Base** → `add paste` ✗ (71단어)
- belief: Mixing the paste in the bowl.
- reasoning: The person is currently holding a spoon and stirring a white paste in a bowl, which suggests they are mixing or preparing a substance. Given the history of actions involving filling and squeezing bags, and the current context of mixing, the most logical next step is to continue mixing the paste. Among the candidates, "add paste" is the most relevant as it aligns with the current activity of preparing a mixture.
- [이력 제거] → `mix_(beat,_blend,_whisk) ingredient` ✗ (51단어) · belief: Mixing the paste in the bowl as part of the preparation process.
- [이력 제거] reasoning: The person is holding a spoon and stirring a white paste in a bowl, indicating they are mixing or blending something. The action is focused on the contents of the bowl, and the motion suggests continuous mixing. Among the candidates, "mix_(beat,_blend,_whisk) ingredient" best matches the observed activity of stirring the paste.

**GT-only** → `knead dough` ✗ [echo] [근거언급] (59단어)
- belief: knead dough
- reasoning: The person is currently mixing a white substance in a bowl, likely dough or batter, using a spoon. This is a continuation of the previous actions involving dough preparation. The most logical next step is to continue mixing or preparing the dough, which aligns with the action 'knead dough'. The other options are not relevant to the current activity.
- [이력 제거] → `add ingredient` ✗ (69단어) · belief: Adding an ingredient to the mixture being prepared.
- [이력 제거] reasoning: The person is currently mixing a white substance in a bowl, likely a batter or dough, using a spoon. The action of mixing is ongoing, and the next logical step would be to continue mixing or to add another ingredient to the mixture. Among the candidates, "add ingredient" is the most appropriate next action as it aligns with the context of preparing a mixture that may require additional components.

**Cand.-CE** → `fill bag` ✓ [근거언급] (67단어)
- belief: Continue the process of filling and squeezing the bag.
- reasoning: The person has been repeatedly filling and squeezing a bag, likely for a repetitive task like applying a substance. The current frame shows them holding a plate with a white substance, possibly dough or paste, and a spoon. Given the pattern of actions, the next logical step is to continue the process of preparing or applying the substance, which aligns with 'fill bag' as the next action.
- [이력 제거] → `mix_(beat,_blend,_whisk) ingredient` ✗ (69단어) · belief: Mixing the paste in the bowl.
- [이력 제거] reasoning: The person is currently holding a spoon and stirring a white paste in a bowl, indicating they are mixing or preparing a substance. The action of stirring is ongoing, and the most logical next step is to continue this mixing process to ensure the ingredients are well combined. The other options, such as adding ingredients or filling a bag, are not directly related to the current activity of mixing.

**EGO** → `fill bag` ✓ (79단어)
- belief: I am in the middle of preparing the bag for the next step in the process.
- reasoning: I have just finished squeezing the bag and am holding the spatula over the bowl, which still contains the white paste. The bowl is positioned near the bag, and my hands are ready to interact with it again. Among the candidates, only 'fill bag' directly follows the current state of having just squeezed the bag and having the paste accessible. The other actions like adding ingredients or mixing don’t align with the immediate context of the bag and paste.

## C_echo — n=31
필터: `GT-only belief==action ∧ GT-only ✗ ∧ EGO ✓`

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_164` · GT **mix_(beat,_blend,_whisk) ingredient** (WM 3위) · WM top-1 `add ingredient` · 이력 8
- 후보: add oil, stir ingredient, check heat, make dough, mix_(beat,_blend,_whisk) ingredient, add flour, cover item, grate ingredient, add ingredient, make mixture
- 이력(최근 6): add rice → preheat pan → move bowl → organize_(arrange) tool_(utensil) → check heat → mix_(beat,_blend,_whisk) ingredient

**Base** → `stir ingredient` ✗ [근거언급] (63단어)
- belief: Stirring the mixture in the pot to ensure even cooking.
- reasoning: I am currently stirring the yellow mixture in the pot on the stove, which suggests I am actively cooking or preparing a dish. The previous action was also mixing the ingredient, indicating a continuous process. Given the context, the most logical next step is to continue stirring to ensure even cooking or prevent sticking. The candidate "stir ingredient" directly matches this ongoing activity.
- [이력 제거] → `mix_(beat,_blend,_whisk) ingredient` ✓ (63단어) · belief: Stirring the mixture in the pot to incorporate newly added ingredients.
- [이력 제거] reasoning: I am currently stirring a yellow mixture in a pot on the stove. I have just added some liquid from a small metal bowl into the pot, which suggests I am adjusting the consistency or flavor. The most logical next step is to continue mixing to incorporate the new ingredient properly. Among the candidates, "mix_(beat,_blend,_whisk) ingredient" best matches this ongoing action of stirring.

**GT-only** → `stir ingredient` ✗ [echo] [근거언급] (51단어)
- belief: stir ingredient
- reasoning: The current action involves stirring the yellow mixture in the pot on the stove, which is a continuation of the cooking process. The previous action was also mixing ingredients, indicating a focus on preparing the dish. The most logical next step is to continue stirring to ensure even cooking and consistency.
- [이력 제거] → `stir ingredient` ✗ (59단어) · belief: Stirring the mixture in the pot to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring a yellow mixture in a pot on the stove, which suggests they are in the process of cooking or preparing a dish. The action of stirring is ongoing, and the next logical step would be to continue mixing or adjusting the consistency of the mixture. The candidate "stir ingredient" aligns with this ongoing activity.

**Cand.-CE** → `check heat` ✗ [근거언급] (64단어)
- belief: Check the heat of the cooking surface to ensure proper cooking conditions.
- reasoning: The current action involves mixing the ingredient in the pot, which is part of the cooking process. The previous action was also mixing, indicating a continuous effort to ensure the mixture is well combined. The next logical step, given the context of cooking and the need to monitor the mixture, is to check the heat again to ensure the cooking process is proceeding correctly.
- [이력 제거] → `mix_(beat,_blend,_whisk) ingredient` ✓ (63단어) · belief: Stirring the mixture in the pot to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring a yellow mixture in a pot on the stove, indicating they are in the process of cooking or preparing a dish. The action of stirring is ongoing, and the next logical step would be to continue mixing or adjusting the mixture. Among the candidates, "mix (beat, blend, whisk) ingredient" aligns with the current activity of stirring the mixture.

**EGO** → `mix_(beat,_blend,_whisk) ingredient` ✓ (86단어)
- belief: I am in the middle of mixing the ingredients in the pot to ensure they are well combined.
- reasoning: I am holding a small metal bowl and have just poured its contents into the pot of yellow mixture on the stove. My hand is still in the pot, and I am using a spoon to stir the mixture, which suggests I am continuing to mix the ingredients. The pan on the other burner is preheated with oil, indicating I am preparing for the next step, likely frying. The action of pouring and stirring aligns with mixing the ingredients to achieve the desired consistency before cooking.

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_172` · GT **add rice** (WM 7위) · WM top-1 `stir ingredient` · 이력 16
- 후보: add salt, add ingredient, check heat, make dough, flip ingredient, add rice, stir ingredient, stir dough, add oil, add spice_(seasoning)
- 이력(최근 6): add water → mix_(beat,_blend,_whisk) ingredient → wash spoon → add rice → add rice → add rice

**Base** → `add ingredient` ✗ (68단어)
- belief: Transferring cooked food to a serving container
- reasoning: I am currently holding a ladle with cooked food and have just removed it from the pan. The next logical step in preparing this dish is to transfer the cooked food to a serving plate or bowl. Looking at the candidates, 'add ingredient' is the most plausible as it implies adding something to the dish, which aligns with the action of transferring the cooked food to another container.
- [이력 제거] → `add ingredient` ✗ (68단어) · belief: Transferring cooked food to a container
- [이력 제거] reasoning: I am currently holding a ladle with cooked food and have just lifted it from the pan. The next logical step is to transfer the cooked food to a plate or container for serving or further processing. Looking at the candidates, "add ingredient" is the most plausible as it implies adding something to the food, which aligns with the action of transferring the cooked food to another container.

**GT-only** → `stir ingredient` ✗ [echo] (45단어)
- belief: stir ingredient
- reasoning: The current action involves stirring the mixture in the pan, which is a continuation of the cooking process. The candidate 'stir ingredient' matches this activity, as it directly corresponds to the action of mixing the contents in the pan to ensure even cooking and consistency.
- [이력 제거] → `stir ingredient` ✗ (118단어) · belief: Stirring the batter to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring a yellow batter in a pan on the stove, which suggests they are preparing a dish that requires consistent mixing. The batter appears to be thick and is being stirred with a ladle, indicating that the cooking process is ongoing. The presence of a second pot with a similar batter suggests that the person might be preparing multiple portions or different types of dishes. The action of stirring is crucial to ensure even cooking and prevent burning. The other candidates, such as adding ingredients or checking heat, are not directly related to the current action of stirring. The most logical next step is to continue stirring the batter to ensure it cooks evenly.

**Cand.-CE** → `add rice` ✓ [근거언급] (53단어)
- belief: Adding rice to the pan as part of the cooking process.
- reasoning: The current action involves adding rice to the pan, which is part of the ongoing cooking process. The previous actions show a pattern of adding rice multiple times, indicating it's a key ingredient being incorporated. The candidate 'add rice' aligns with this established pattern and the visual context of the pan and ingredients.
- [이력 제거] → `stir ingredient` ✗ (110단어) · belief: Cooking a flatbread by stirring the batter in the pan.
- [이력 제거] reasoning: The person is currently cooking a dish that appears to be a type of flatbread or pancake, as indicated by the circular shape being formed in the pan. The hand is holding a ladle with batter, suggesting the action of pouring or spreading the batter is ongoing. The other hand is holding a small piece of food, possibly a garnish or a piece of the cooked item. Given the context, the most logical next step is to continue cooking by stirring or spreading the batter to ensure even cooking. The action "stir ingredient" fits this context, as it implies continuing the cooking process by stirring the contents of the pan.

**EGO** → `add rice` ✓ (89단어)
- belief: I am in the middle of preparing the mixture for shaping or portioning.
- reasoning: I am holding a small portion of the prepared mixture in my hand, having just lifted it from the pan. The ladle is still in the pan, and the mixture appears to be at a stage where it needs to be shaped or portioned. The bowl of yellow rice is nearby, suggesting the next step involves incorporating or preparing the rice. Among the candidates, adding rice is the only one that aligns with the current state of holding a portion of mixture and the presence of the rice bowl.

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_180` · GT **add rice** (WM 9위) · WM top-1 `add ingredient` · 이력 24
- 후보: add batter, cover item, check heat, add ingredient, add oil, stir ingredient, add rice, turn stovetop, add spice_(seasoning), mix_(beat,_blend,_whisk) ingredient
- 이력(최근 6): add rice → add rice → add rice → add rice → add rice → add rice

**Base** → `stir ingredient` ✗ (59단어)
- belief: Stir the rice in the pan to cook it evenly.
- reasoning: I am currently stirring the yellow rice in the pan on the stove with my left hand while holding a container of batter over another pot with my right hand. The rice appears to be cooking and needs to be stirred to prevent sticking. The most logical next step is to continue stirring the rice to ensure even cooking.
- [이력 제거] → `stir ingredient` ✗ (79단어) · belief: Stirring the contents in the pan to ensure even cooking.
- [이력 제거] reasoning: I am currently stirring the yellow substance in the pan on the left burner while simultaneously holding a container over the pot on the right burner. The action of stirring the pan is ongoing, and the container being held suggests I am preparing to add something to the pot. Given the context of cooking and the current state of the pan, the most logical next step is to continue stirring the contents in the pan to ensure even cooking.

**GT-only** → `stir ingredient` ✗ [echo] (49단어)
- belief: stir ingredient
- reasoning: The current action involves adding rice to the pan, which is part of the cooking process. The next logical step, given the context of cooking and the need to ensure even cooking, is to stir the ingredient in the pan. This will help distribute the heat and prevent burning.
- [이력 제거] → `stir ingredient` ✗ (71단어) · belief: Stirring the cooking mixture in the pan.
- [이력 제거] reasoning: The person is currently stirring a yellow mixture in a pan on the stove, which appears to be a cooking process. The other hand is holding a container, possibly adding or preparing to add more ingredients. Given the context of cooking and the action of stirring, the most logical next step is to continue stirring the mixture to ensure even cooking and prevent burning. The candidate "stir ingredient" matches this action.

**Cand.-CE** → `add rice` ✓ (61단어)
- belief: Adding rice to the pan to complete the cooking process.
- reasoning: The current action involves adding rice to the pan, which is part of the cooking process. The repeated addition of rice suggests that the task is nearing completion or requires a final addition. The most logical next step, given the context and the need to continue the cooking process, is to add more rice to ensure the dish is properly prepared.
- [이력 제거] → `add batter` ✗ (74단어) · belief: Adding batter to the pan to continue cooking.
- [이력 제거] reasoning: The person is actively cooking, stirring a yellow mixture in a pan on the stove with one hand while holding a container of batter over another pot with the other hand. The batter appears to be being prepared for addition to the pan. Given the context of cooking and the current state of the ingredients, the most logical next step is to add the prepared batter to the pan to continue the cooking process.

**EGO** → `add rice` ✓ (74단어)
- belief: I am in the middle of adding rice to the pot on the stove.
- reasoning: I am continuing to add rice to the pot on the stove, as indicated by the repeated action and the visible pile of uncooked rice in the bowl. The hand holding the spoon is positioned over the pot, and the motion suggests another portion is being added. The other candidates, like stirring or adding oil, are not supported by the current visual context or the recent history of actions, which consistently involve adding rice.

## D_utility — n=23
필터: `GT-only 근거언급 ∧ ✗ · EGO 근거언급 ∧ ✓`

### `13c76616-f168-4af0-8d2a-fe82ce232d6a_527` · GT **cook rice** (WM 10위) · WM top-1 `add rice` · 이력 25
- 후보: add rice, wash ingredient, check heat, wipe dish, stir dish, add salt, sieve ingredient, cook rice, turn stovetop, add water
- 이력(최근 6): check heat → cook rice → check heat → cook rice → remove dirt → cook rice

**Base** → `stir dish` ✗ [근거언급] (98단어)
- belief: Continue cooking the rice by stirring it in the pot.
- reasoning: I am currently stirring the rice in the pot with a wooden spoon, which is part of the cooking process. The previous actions show a repetitive pattern of cooking rice, checking heat, and removing dirt, indicating that the rice is being prepared for consumption. Since I am actively stirring the rice, the most logical next step is to continue cooking it, which aligns with the ongoing task of preparing the rice. The other options, such as adding water or salt, are not immediately necessary at this stage, as the rice is already in the pot and being stirred.
- [이력 제거] → `check heat` ✗ (62단어) · belief: Monitor the heat while cooking rice.
- [이력 제거] reasoning: I am currently stirring the rice in the pot with a wooden spoon. The pot is positioned over a heat source, indicating that cooking is in progress. The most logical next step, given that I am actively stirring and the rice is being cooked, is to continue monitoring the cooking process by checking the heat to ensure it's appropriate for the rice.

**GT-only** → `stir dish` ✗ [근거언급] (63단어)
- belief: stir the rice in the pot to ensure even cooking
- reasoning: The current action involves stirring the rice in the pot, which is part of the cooking process. The previous actions indicate a repetitive cycle of checking heat and cooking rice, with occasional removal of dirt. The most logical next step, given the context of ongoing cooking, is to continue stirring the rice to ensure even cooking. The candidate 'stir dish' matches this requirement.
- [이력 제거] → `check heat` ✗ (60단어) · belief: Check the heat of the cooking pot.
- [이력 제거] reasoning: The person is currently stirring the rice in the pot, which suggests they are in the process of cooking it. The presence of a fire under the pot indicates that the cooking process is active. The most logical next step, given the context of cooking rice, is to check the heat to ensure it is appropriate for the cooking process.

**Cand.-CE** → `cook rice` ✓ (69단어)
- belief: Continue cooking the rice.
- reasoning: The current action history shows a repetitive cycle of cooking rice, checking heat, and removing dirt. The most recent action was removing dirt, which suggests the rice is being prepared for further cooking. The next logical step in this process is to continue cooking the rice, as the heat has been checked and the rice is already in the pot. The candidate 'cook rice' aligns with this ongoing task.
- [이력 제거] → `check heat` ✗ (56단어) · belief: Check the heat of the cooking pot.
- [이력 제거] reasoning: The person is currently stirring rice in a large pot, which suggests they are preparing to cook it. The presence of a fire under the pot indicates that cooking is underway. The most logical next step, given the context of cooking rice, is to check the heat to ensure it is appropriate for the cooking process.

**EGO** → `cook rice` ✓ [근거언급] (79단어)
- belief: I am in the middle of cooking the rice in the pot.
- reasoning: I have been repeatedly checking the heat and cooking the rice, and just removed dirt from the pot. The rice is still in the pot and I am holding the wooden paddle, indicating I am in the middle of stirring or adjusting the rice. The most consistent continuation of this activity is to keep cooking the rice, as the other options like adding water or salt are not supported by the current state of the pot or my tools.

### `13c76616-f168-4af0-8d2a-fe82ce232d6a_530` · GT **check heat** (WM 2위) · WM top-1 `add rice` · 이력 28
- 후보: strain liquid, turn stovetop, add ingredient, sieve ingredient, check doneness, add rice, put_(place,_set) dough, check heat, add water, wash rice
- 이력(최근 6): cook rice → remove dirt → cook rice → check heat → cook rice → check heat

**Base** → `check doneness` ✗ [근거언급] (63단어)
- belief: Assess if the rice is cooked to the desired consistency.
- reasoning: The person is currently stirring rice in a large pot, which suggests they are in the process of cooking or preparing it. Given the history of repeatedly checking heat and cooking rice, the logical next step would be to assess if the rice is done. The candidate "check doneness" aligns with this context, as it would determine if the cooking process is complete.
- [이력 제거] → `add water` ✗ (68단어) · belief: Adding water to the grains to begin cooking or soaking.
- [이력 제거] reasoning: I am currently stirring grains in a large bowl, which suggests I am preparing or processing them. The action of stirring indicates I am ensuring even mixing or heating. Since I am not adding anything yet, and the grains appear dry, the most logical next step is to add water to begin cooking or soaking them. This aligns with the typical process of preparing rice or similar grains.

**GT-only** → `check doneness` ✗ [근거언급] (65단어)
- belief: Check if the rice is cooked.
- reasoning: The current action involves stirring rice in a large pot, which is part of the cooking process. The previous actions indicate a repetitive cycle of checking heat and cooking rice, with occasional removal of dirt. The most logical next step, given the context of cooking rice, is to check if it is done, as this would determine if further cooking or additional steps are needed.
- [이력 제거] → `check doneness` ✗ (79단어) · belief: Check the doneness of the rice.
- [이력 제거] reasoning: The person is currently stirring a large quantity of what appears to be rice or grain in a large bowl. The action of stirring is ongoing, and the context suggests this is part of a cooking or preparation process. The most logical next step, given the current action and the nature of the task, is to check the doneness of the rice, as this is a common step after stirring or cooking grains to ensure they are properly prepared.

**Cand.-CE** → `check doneness` ✗ (57단어)
- belief: Check if the rice is done cooking.
- reasoning: The current action history shows a repetitive cycle of checking heat and cooking rice, with occasional removal of dirt. The most recent action was checking heat, and the context suggests the rice is still being cooked. The logical next step in this process is to continue monitoring the cooking progress by checking if the rice is done.
- [이력 제거] → `check doneness` ✗ (61단어) · belief: Assessing the readiness of the grain after stirring.
- [이력 제거] reasoning: The person is currently stirring a large bowl of what appears to be rice or a similar grain. The action of stirring suggests that the grain is being prepared, possibly for cooking. Given the context of preparing grains, checking the doneness (5) would be a logical next step after stirring to assess if the grain is cooked to the desired consistency.

**EGO** → `check heat` ✓ [근거언급] (64단어)
- belief: I am in the middle of monitoring the cooking process to maintain the correct heat level.
- reasoning: I have been repeatedly checking the heat and cooking the rice, and just finished removing dirt from the pot. The rice is still in the pot and appears to be in the process of being cooked, with no signs of being strained or washed yet. Checking the heat again is the most consistent next step to ensure the cooking process continues properly without burning.

### `1558e9f2-d7f3-4a23-9627-4240f506d7df_568` · GT **roll dough** (WM 1위) · WM top-1 `roll dough` · 이력 22
- 후보: roll dough, coat dough, sprinkle flour, put_(place,_set) dough, stir ingredient, add topping, cut dough, shape dough, press dough, make dough
- 이력(최근 6): flip ingredient → add ingredient → roll dough → add ingredient → roll dough → flip ingredient

**Base** → `shape dough` ✗ [근거언급] (59단어)
- belief: Shape the dough after rolling and adding ingredients.
- reasoning: I have been repeatedly rolling and flipping the dough, and just added an ingredient. The dough is now on the table, and I need to prepare it for the next step. Since I’ve been rolling it, and the dough appears ready, the logical next step is to shape it, which aligns with the current state and my previous actions.
- [이력 제거] → `press dough` ✗ (78단어) · belief: Preparing the dough by pressing it flat on the table.
- [이력 제거] reasoning: I am currently working with dough on the table, and my hands are actively pressing and shaping it. The flour container is nearby, suggesting I might need to dust the dough or surface. Since I've already placed the dough on the table, the next logical step is to prepare it for rolling or further shaping. The action "press dough" aligns with what I'm currently doing to ensure the dough is evenly spread and ready for the next step.

**GT-only** → `shape dough` ✗ [근거언급] (49단어)
- belief: Shape the dough into the desired form.
- reasoning: The dough has been rolled out and is now being shaped into a specific form, likely for a dish. The previous actions involved rolling and flipping the dough, indicating a preparation process. The current action is focused on shaping the dough, which is a logical next step after rolling.
- [이력 제거] → `press dough` ✗ (74단어) · belief: shaping the dough on the table
- [이력 제거] reasoning: The hands are actively working with the dough on the table, indicating a shaping or pressing action is underway. The rolling pin is nearby but not in use, suggesting the focus is on forming the dough rather than rolling it out. The presence of flour suggests it might be used for dusting, but the immediate action is on the dough itself. The most logical next step is to continue shaping or pressing the dough.

**Cand.-CE** → `press dough` ✗ (60단어)
- belief: Shape the dough by pressing it to achieve the desired form.
- reasoning: The dough has been rolled multiple times and flipped with ingredients added, indicating a repetitive preparation process. The current state shows the dough is ready for the next step, which is likely to involve shaping or pressing it to achieve the desired form. The action 'press dough' aligns with the ongoing task of preparing the dough for the next stage.
- [이력 제거] → `press dough` ✗ (64단어) · belief: Shaping the dough on the table.
- [이력 제거] reasoning: I am currently working with dough on the table, and my hands are actively shaping it. The rolling pin and flour container are nearby, indicating I am in the process of preparing or working with the dough. Given the context, the most logical next step is to continue shaping the dough, which aligns with the action of pressing it to achieve the desired form.

**EGO** → `roll dough` ✓ [근거언급] (70단어)
- belief: I am in the middle of preparing the dough by rolling it out.
- reasoning: I have been repeatedly flipping and rolling the dough, and the current state shows the dough is still on the table with the rolling pin nearby. The action history indicates a pattern of preparing the dough through rolling, and the rolling pin is positioned for continued use. Among the candidates, rolling dough is the only one that directly follows from the current state and matches the established pattern of preparation.

## E_history — n=113
필터: `Cand.-CE 이력有 ✓ ∧ 이력無 ✗ ∧ Base ✗`

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_166` · GT **mix_(beat,_blend,_whisk) ingredient** (WM 8위) · WM top-1 `stir ingredient` · 이력 10
- 후보: mix_(beat,_blend,_whisk) ingredient, cover item, stir ingredient, make mixture, grate ingredient, check heat, add ingredient, add oil, remove inredient, add water
- 이력(최근 6): move bowl → organize_(arrange) tool_(utensil) → check heat → mix_(beat,_blend,_whisk) ingredient → add water → mix_(beat,_blend,_whisk) ingredient

**Base** → `stir ingredient` ✗ [근거언급] (75단어)
- belief: Continue mixing the ingredients in the pot.
- reasoning: I am currently stirring the yellow mixture in the pot on the stove, which suggests I am actively mixing the ingredients. The previous action was also mixing, and I just added water. Since the mixture is still cooking and needs consistent stirring to prevent sticking or uneven cooking, the most logical next step is to continue stirring. The other options like adding water or checking heat are less relevant now that I am already mixing.
- [이력 제거] → `stir ingredient` ✗ (56단어) · belief: Stirring the mixture in the pot to ensure even cooking.
- [이력 제거] reasoning: I am currently stirring a yellow mixture in a pot on the stove. I have just poured water from a metal cup into the pot, and now I am continuing to stir the mixture. The most logical next step is to keep stirring to ensure the ingredients are well combined and the mixture is evenly cooked.

**GT-only** → `mix_(beat,_blend,_whisk) ingredient` ✓ [근거언급] (55단어)
- belief: Mixing the ingredients in the pot to ensure even cooking and consistency.
- reasoning: The current action involves mixing the ingredients in the pot, which is a continuation of the cooking process. The previous actions have been focused on preparing and mixing the ingredients, and the next logical step is to continue mixing to ensure even cooking and consistency. The candidate action 'mix_(beat,_blend,_whisk) ingredient' aligns with this ongoing task.
- [이력 제거] → `stir ingredient` ✗ (63단어) · belief: Stirring the mixture to ensure even cooking and consistency.
- [이력 제거] reasoning: The person is currently stirring a yellow mixture in a pot on the stove, which suggests they are actively cooking or preparing a dish. The presence of a ladle and the motion of stirring indicate that the next step is likely to involve further mixing or checking the consistency of the mixture. Among the candidates, "stir ingredient" directly aligns with the current activity.

**Cand.-CE** → `mix_(beat,_blend,_whisk) ingredient` ✓ [근거언급] (55단어)
- belief: Continue mixing the ingredient in the pot.
- reasoning: The current action involves mixing the ingredient in the pot, which is part of the cooking process. The previous actions indicate a sequence of adding and mixing ingredients, and the current state shows the mixture is still being stirred. The most logical next step is to continue mixing to ensure the ingredients are well combined.
- [이력 제거] → `stir ingredient` ✗ (75단어) · belief: Stirring the mixture to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring a yellow mixture in a pot on the stove, indicating they are actively cooking or preparing a dish. The previous action involved adding water, and now they are continuing to stir the mixture, which is a logical next step to ensure even cooking and prevent sticking. The other options, such as adding more ingredients or checking the heat, are not immediately necessary given the current state of the cooking process.

**EGO** → `mix_(beat,_blend,_whisk) ingredient` ✓ (81단어)
- belief: I am in the middle of mixing the ingredients in the pot to achieve the desired consistency.
- reasoning: I am holding a ladle in the pot with the yellow mixture and have just added water from a metal cup. The mixture is still on the stove, and I am actively stirring it with the ladle. The pan to the left is preheated with oil, indicating I am preparing to cook the next step. Among the candidates, stirring the mixture is the only action that matches my current hand position and the motion of mixing the contents in the pot.

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_172` · GT **add rice** (WM 7위) · WM top-1 `stir ingredient` · 이력 16
- 후보: add salt, add ingredient, check heat, make dough, flip ingredient, add rice, stir ingredient, stir dough, add oil, add spice_(seasoning)
- 이력(최근 6): add water → mix_(beat,_blend,_whisk) ingredient → wash spoon → add rice → add rice → add rice

**Base** → `add ingredient` ✗ (68단어)
- belief: Transferring cooked food to a serving container
- reasoning: I am currently holding a ladle with cooked food and have just removed it from the pan. The next logical step in preparing this dish is to transfer the cooked food to a serving plate or bowl. Looking at the candidates, 'add ingredient' is the most plausible as it implies adding something to the dish, which aligns with the action of transferring the cooked food to another container.
- [이력 제거] → `add ingredient` ✗ (68단어) · belief: Transferring cooked food to a container
- [이력 제거] reasoning: I am currently holding a ladle with cooked food and have just lifted it from the pan. The next logical step is to transfer the cooked food to a plate or container for serving or further processing. Looking at the candidates, "add ingredient" is the most plausible as it implies adding something to the food, which aligns with the action of transferring the cooked food to another container.

**GT-only** → `stir ingredient` ✗ [echo] (45단어)
- belief: stir ingredient
- reasoning: The current action involves stirring the mixture in the pan, which is a continuation of the cooking process. The candidate 'stir ingredient' matches this activity, as it directly corresponds to the action of mixing the contents in the pan to ensure even cooking and consistency.
- [이력 제거] → `stir ingredient` ✗ (118단어) · belief: Stirring the batter to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring a yellow batter in a pan on the stove, which suggests they are preparing a dish that requires consistent mixing. The batter appears to be thick and is being stirred with a ladle, indicating that the cooking process is ongoing. The presence of a second pot with a similar batter suggests that the person might be preparing multiple portions or different types of dishes. The action of stirring is crucial to ensure even cooking and prevent burning. The other candidates, such as adding ingredients or checking heat, are not directly related to the current action of stirring. The most logical next step is to continue stirring the batter to ensure it cooks evenly.

**Cand.-CE** → `add rice` ✓ [근거언급] (53단어)
- belief: Adding rice to the pan as part of the cooking process.
- reasoning: The current action involves adding rice to the pan, which is part of the ongoing cooking process. The previous actions show a pattern of adding rice multiple times, indicating it's a key ingredient being incorporated. The candidate 'add rice' aligns with this established pattern and the visual context of the pan and ingredients.
- [이력 제거] → `stir ingredient` ✗ (110단어) · belief: Cooking a flatbread by stirring the batter in the pan.
- [이력 제거] reasoning: The person is currently cooking a dish that appears to be a type of flatbread or pancake, as indicated by the circular shape being formed in the pan. The hand is holding a ladle with batter, suggesting the action of pouring or spreading the batter is ongoing. The other hand is holding a small piece of food, possibly a garnish or a piece of the cooked item. Given the context, the most logical next step is to continue cooking by stirring or spreading the batter to ensure even cooking. The action "stir ingredient" fits this context, as it implies continuing the cooking process by stirring the contents of the pan.

**EGO** → `add rice` ✓ (89단어)
- belief: I am in the middle of preparing the mixture for shaping or portioning.
- reasoning: I am holding a small portion of the prepared mixture in my hand, having just lifted it from the pan. The ladle is still in the pan, and the mixture appears to be at a stage where it needs to be shaped or portioned. The bowl of yellow rice is nearby, suggesting the next step involves incorporating or preparing the rice. Among the candidates, adding rice is the only one that aligns with the current state of holding a portion of mixture and the presence of the rice bowl.

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_174` · GT **add rice** (WM 8위) · WM top-1 `stir dough` · 이력 18
- 후보: turn stovetop, mix_(beat,_blend,_whisk) ingredient, stir ingredient, press dough, add ingredient, stir dough, add rice, add oil, flip ingredient, make dough
- 이력(최근 6): wash spoon → add rice → add rice → add rice → add rice → add rice

**Base** → `stir ingredient` ✗ (71단어)
- belief: Stir the batter in the pan to cook it evenly.
- reasoning: I am currently holding a ladle over a pot of yellow batter, likely preparing to pour or stir it. The batter is already in a pan on the stove, suggesting the cooking process is underway. Given the recent actions of adding rice and mixing ingredients, the next logical step is to stir the batter in the pan to ensure even cooking and prevent sticking. The candidate "stir ingredient" matches this need.
- [이력 제거] → `stir ingredient` ✗ (92단어) · belief: Cooking a batter-based dish by spreading it in a pan.
- [이력 제거] reasoning: The person is currently using a ladle to scoop and pour a yellow batter from a pot into a pan on the stove, which suggests they are in the process of cooking a dish like dosa or crepe. The batter is being poured in a circular motion, indicating the action of spreading or shaping the batter. Among the candidates, "stir ingredient" is the most appropriate because the person is actively mixing or spreading the batter in the pan, which is a form of stirring or spreading the ingredient to cook it evenly.

**GT-only** → `stir ingredient` ✗ [echo] (64단어)
- belief: stir ingredient
- reasoning: The current action involves stirring the mixture in the pan, which is a continuation of the cooking process. The candidate 'stir ingredient' matches this activity, as it directly corresponds to the action of mixing or stirring the contents of the pan to ensure even cooking. The other candidates either refer to different stages of preparation or are not relevant to the current cooking step.
- [이력 제거] → `stir ingredient` ✗ (69단어) · belief: Stirring the batter in the pan to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring the yellow batter in the pan with a ladle, which is a common step in preparing dishes like dosa or crepes. The batter is being mixed to ensure even cooking and consistency. The other actions, such as adding ingredients or flipping, are not currently being performed. The most logical next step is to continue stirring the batter to prevent sticking and ensure proper cooking.

**Cand.-CE** → `add rice` ✓ [근거언급] (56단어)
- belief: Adding rice to the mixture as part of the cooking process.
- reasoning: The current action involves adding rice to the mixture, which is part of the ongoing process of preparing the dish. The previous actions have been focused on adding rice repeatedly, indicating that this is a key step in the recipe. The candidate "add rice" matches the current context and the sequence of actions performed so far.
- [이력 제거] → `stir ingredient` ✗ (66단어) · belief: Stirring the batter in the pot on the stove.
- [이력 제거] reasoning: The person is currently stirring a yellow batter in a pot on the stove with a ladle. The batter appears to be thick and is being mixed to ensure even cooking or consistency. The other pots and pans on the stove suggest multiple dishes are being prepared simultaneously. The most logical next action is to continue stirring the batter to prevent sticking and ensure uniform cooking.

**EGO** → `stir ingredient` ✗ (98단어)
- belief: I am in the middle of mixing the rice with the liquid in the pot to ensure even cooking.
- reasoning: I have just finished adding rice to the pot and am now holding the ladle over the mixture, which is still steaming. The rice appears to be partially cooked and needs further integration with the liquid. Among the candidates, stirring the mixture is the only action that directly addresses the need to combine the rice with the surrounding liquid to ensure even cooking. The other options, like adding more rice or oil, are not indicated by the current state of the pot or my hand position. Stirring is the most logical next step to achieve a uniform consistency.

## F_counter — n=96
필터: `WM top-1 ✗ ∧ Cand.-CE ✓ ∧ EGO ✗`

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_174` · GT **add rice** (WM 8위) · WM top-1 `stir dough` · 이력 18
- 후보: turn stovetop, mix_(beat,_blend,_whisk) ingredient, stir ingredient, press dough, add ingredient, stir dough, add rice, add oil, flip ingredient, make dough
- 이력(최근 6): wash spoon → add rice → add rice → add rice → add rice → add rice

**Base** → `stir ingredient` ✗ (71단어)
- belief: Stir the batter in the pan to cook it evenly.
- reasoning: I am currently holding a ladle over a pot of yellow batter, likely preparing to pour or stir it. The batter is already in a pan on the stove, suggesting the cooking process is underway. Given the recent actions of adding rice and mixing ingredients, the next logical step is to stir the batter in the pan to ensure even cooking and prevent sticking. The candidate "stir ingredient" matches this need.
- [이력 제거] → `stir ingredient` ✗ (92단어) · belief: Cooking a batter-based dish by spreading it in a pan.
- [이력 제거] reasoning: The person is currently using a ladle to scoop and pour a yellow batter from a pot into a pan on the stove, which suggests they are in the process of cooking a dish like dosa or crepe. The batter is being poured in a circular motion, indicating the action of spreading or shaping the batter. Among the candidates, "stir ingredient" is the most appropriate because the person is actively mixing or spreading the batter in the pan, which is a form of stirring or spreading the ingredient to cook it evenly.

**GT-only** → `stir ingredient` ✗ [echo] (64단어)
- belief: stir ingredient
- reasoning: The current action involves stirring the mixture in the pan, which is a continuation of the cooking process. The candidate 'stir ingredient' matches this activity, as it directly corresponds to the action of mixing or stirring the contents of the pan to ensure even cooking. The other candidates either refer to different stages of preparation or are not relevant to the current cooking step.
- [이력 제거] → `stir ingredient` ✗ (69단어) · belief: Stirring the batter in the pan to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring the yellow batter in the pan with a ladle, which is a common step in preparing dishes like dosa or crepes. The batter is being mixed to ensure even cooking and consistency. The other actions, such as adding ingredients or flipping, are not currently being performed. The most logical next step is to continue stirring the batter to prevent sticking and ensure proper cooking.

**Cand.-CE** → `add rice` ✓ [근거언급] (56단어)
- belief: Adding rice to the mixture as part of the cooking process.
- reasoning: The current action involves adding rice to the mixture, which is part of the ongoing process of preparing the dish. The previous actions have been focused on adding rice repeatedly, indicating that this is a key step in the recipe. The candidate "add rice" matches the current context and the sequence of actions performed so far.
- [이력 제거] → `stir ingredient` ✗ (66단어) · belief: Stirring the batter in the pot on the stove.
- [이력 제거] reasoning: The person is currently stirring a yellow batter in a pot on the stove with a ladle. The batter appears to be thick and is being mixed to ensure even cooking or consistency. The other pots and pans on the stove suggest multiple dishes are being prepared simultaneously. The most logical next action is to continue stirring the batter to prevent sticking and ensure uniform cooking.

**EGO** → `stir ingredient` ✗ (98단어)
- belief: I am in the middle of mixing the rice with the liquid in the pot to ensure even cooking.
- reasoning: I have just finished adding rice to the pot and am now holding the ladle over the mixture, which is still steaming. The rice appears to be partially cooked and needs further integration with the liquid. Among the candidates, stirring the mixture is the only action that directly addresses the need to combine the rice with the surrounding liquid to ensure even cooking. The other options, like adding more rice or oil, are not indicated by the current state of the pot or my hand position. Stirring is the most logical next step to achieve a uniform consistency.

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_175` · GT **add rice** (WM 2위) · WM top-1 `stir ingredient` · 이력 19
- 후보: serve dish, turn stovetop, mix_(beat,_blend,_whisk) ingredient, add ingredient, add oil, stir ingredient, add rice, check heat, cover item, add spice_(seasoning)
- 이력(최근 6): add rice → add rice → add rice → add rice → add rice → add rice

**Base** → `stir ingredient` ✗ (72단어)
- belief: Stir the rice in the pan to ensure even cooking.
- reasoning: The person is currently stirring the rice in the pan with a spoon, indicating they are actively cooking and mixing the ingredients. The repeated action of adding rice suggests they are still in the process of cooking and may need to add more rice or adjust the mixture. The most logical next step, given the context of ongoing cooking and the need to ensure even cooking, is to continue stirring the ingredient.
- [이력 제거] → `add spice_(seasoning)` ✗ (72단어) · belief: Adding spice to the cooking food
- [이력 제거] reasoning: The person is actively stirring the yellow granular food in the pan with a spoon while holding a small container of what appears to be spice or seasoning in their other hand. This suggests they are in the process of adding or mixing an ingredient. The most logical next step, given the context of cooking and the current action of holding a spice container, is to add the spice to the pan.

**GT-only** → `add rice` ✓ (47단어)
- belief: Adding rice to the pan as part of the cooking process.
- reasoning: The current action involves adding rice to the pan, which is part of the cooking process. The candidate 'add rice' matches this activity. The other candidates like 'stir ingredient' or 'add spice' are not directly aligned with the current visual context of adding rice to the pan.
- [이력 제거] → `stir ingredient` ✗ (61단어) · belief: Stirring the cooking mixture in the pan.
- [이력 제거] reasoning: The person is currently stirring the yellow grain mixture in the pan with a spoon, indicating they are actively cooking. The other hand is holding a small container, possibly to add more ingredients. Given the context of cooking and the action of stirring, the most logical next step is to continue stirring the mixture to ensure even cooking and prevent burning.

**Cand.-CE** → `add rice` ✓ [근거언급] (52단어)
- belief: Continue adding rice to the pan as part of the cooking process.
- reasoning: The current action involves adding rice to the pan, which is part of the cooking process. The previous actions show a repetitive pattern of adding rice, indicating that this is an ongoing step. The candidate "add rice" matches this pattern and is the most logical next action to continue the cooking process.
- [이력 제거] → `add spice_(seasoning)` ✗ (64단어) · belief: Adding spice to the dish being cooked.
- [이력 제거] reasoning: The person is actively cooking, stirring the contents of the pan with a spoon while holding a small container of what appears to be spice or seasoning. The action of adding spice is consistent with the current cooking process and the ingredients being prepared. The other options like serving, turning the stovetop, or adding oil are not directly supported by the current visual context.

**EGO** → `mix_(beat,_blend,_whisk) ingredient` ✗ [근거언급] (72단어)
- belief: I am in the middle of cooking rice in the pan, ensuring it is evenly cooked and not sticking.
- reasoning: I am holding a spoon over the pan and stirring the yellow grains, which are actively cooking. The grains are visibly moving and appear to be absorbing liquid, indicating they need more time to cook evenly. The previous actions involved adding rice multiple times, suggesting the current stage is about ensuring the rice is properly cooked. Stirring is the most direct way to check consistency and prevent sticking while the rice cooks.

### `04fe8f4d-081e-437e-a56a-2d53b6233fc9_190` · GT **add rice** (WM 9위) · WM top-1 `check heat` · 이력 34
- 후보: make dough, check heat, add spice_(seasoning), add rice, add oil, cover item, add ingredient, turn stovetop, mix_(beat,_blend,_whisk) ingredient, stir ingredient
- 이력(최근 6): add rice → add rice → add rice → add rice → add rice → add rice

**Base** → `stir ingredient` ✗ (59단어)
- belief: Stir the rice in the pan to cook it evenly.
- reasoning: I am currently stirring the rice in the pan on the stove with a spoon in my left hand, while holding a lid over another pot with my right hand. The rice appears to be cooking and needs to be stirred to prevent burning. The most logical next step is to continue stirring the rice to ensure even cooking.
- [이력 제거] → `add ingredient` ✗ (87단어) · belief: Adding an ingredient to the pan while stirring.
- [이력 제거] reasoning: I am currently stirring the yellow grains in the pan on the stove with a spoon in my left hand. My right hand is holding a container with a lid, possibly containing an ingredient to be added. Given the context of cooking and the action of stirring, the most logical next step is to add an ingredient to the pan to continue the cooking process. Among the candidates, "add ingredient" fits best as it directly follows the current action of stirring and aligns with the cooking activity.

**GT-only** → `stir ingredient` ✗ (116단어)
- belief: Stirring the rice in the pan to ensure even cooking.
- reasoning: The current action involves stirring the contents of the pan on the stove, which is a continuous cooking process. The candidate 'stir ingredient' matches this activity directly. Other candidates like 'add rice' or 'add spice' are not relevant as the rice has already been added and the focus is on stirring. 'Check heat' or 'turn stovetop' are not directly observable actions in the current frame. 'Make dough' is unrelated to the current cooking task. 'Cover item' is not applicable as the pan is open. 'Add oil' is not needed as oil is already present. 'Mix' is too vague compared to 'stir'. 'Add ingredient' is too general. 'Turn stovetop' is not visible or necessary at this moment.
- [이력 제거] → `stir ingredient` ✗ (64단어) · belief: Stirring the mixture in the pan to ensure even cooking.
- [이력 제거] reasoning: The person is currently stirring a yellow mixture in a pan on the stove, which appears to be a cooking process. The other hand is holding a container, possibly adding or preparing to add more ingredients. Given the context of cooking and the action of stirring, the most logical next step is to continue stirring the mixture to ensure even cooking and prevent burning.

**Cand.-CE** → `add rice` ✓ (53단어)
- belief: Continue adding rice to the pan.
- reasoning: The current action involves adding rice to the pan, which is part of the ongoing cooking process. The repeated addition of rice suggests that the task is to ensure the rice is properly cooked or mixed. The next logical step, given the context, is to continue adding rice to complete the cooking process.
- [이력 제거] → `stir ingredient` ✗ (65단어) · belief: Stirring the contents of the pan on the stove.
- [이력 제거] reasoning: I am currently stirring the yellow substance in the pan on the stove with a spoon in my left hand. My right hand is holding a container with a lid, possibly containing an ingredient to be added. Given the context of cooking and the action of stirring, the most logical next step is to continue stirring the contents of the pan to ensure even cooking.

**EGO** → `stir ingredient` ✗ (93단어)
- belief: I am in the middle of cooking rice in the pan.
- reasoning: I am continuing to stir the rice in the pan with the spoon in my left hand, while my right hand holds a container of yellow liquid over another pot. The rice is visibly cooking and being mixed, and I am maintaining this action to ensure even cooking. The other pot with the yellow liquid is being held steady, suggesting I am not yet ready to interact with it. Among the candidates, stirring the rice is the most consistent with the current state of the task and the tools I am actively using.
