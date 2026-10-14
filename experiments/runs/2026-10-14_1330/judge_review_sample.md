# LLM-judge spot check

Judge: `gpt-6.1-sol` (low reasoning effort), blinded to the system that produced each answer
(`scripts/judge_open_ended.py`). It only sees open-ended answers that the strict scorer marked wrong;
only a `correct` verdict adds credit.

Sample: 30 verdicts, seeded random (15 judged `correct`, 15 judged `partially_correct`/`incorrect`).
Reviewer: Claude (AI-assisted review during development). **This is not a human annotation**, so please
confirm or overrule the "Reviewer" column before quoting the agreement figure.

Agreement on the decision that matters (credit vs no credit): **29 / 30 (97%)**.

| # | Case | System | Reference | Answer (abridged) | Judge | Reviewer |
|---|---|---|---|---|---|---|
| 1 | 2154 | baseline | calcified atherosclerosis | atherosclerotic calcifications in the aortic wall | correct | agree |
| 2 | 1891 | baseline | ivc | drains into the inferior vena cava | correct | agree |
| 3 | 1910 | agentic | coronal plane? | measure on a coronal image, upper to lower pole | correct | agree |
| 4 | 1910 | baseline | coronal plane? | longest axis, preferably on a coronal image | correct | agree |
| 5 | 2063 | pipeline | xr | X-ray (frontal chest radiograph) | correct | agree |
| 6 | 2201 | baseline | middle mogul | AP window mildly prominent and partially effaced | correct | **disagree (partial)** |
| 7 | 2183 | pipeline | hypodense lesion | rounded area of lower attenuation in the spleen | correct | agree |
| 8 | 2202 | baseline | middle mogul | fullness of the aortopulmonary window | correct | agree |
| 9 | 2231 | agentic | blood vessel | likely an enhancing hepatic vessel | correct | agree |
| 10 | 2154 | pipeline | calcified atherosclerosis | calcifications along the aortic wall, atherosclerotic plaque | correct | agree |
| 11 | 2153 | pipeline | calcified atherosclerosis | peripheral aortic atherosclerotic calcifications | correct | agree |
| 12 | 2202 | pipeline | middle mogul | mild fullness of the aortopulmonary window | correct | agree |
| 13 | 1858 | pipeline | cardiomegaly with pulmonary edema | enlarged cardiac silhouette, haziness that may reflect mild pulmonary edema | correct | agree |
| 14 | 2116 | agentic | pulmonary nodules | multiple bilateral nodular opacities | correct | agree |
| 15 | 2139 | workflow | free air | crescent-shaped lucency beneath the right hemidiaphragm | correct | agree |
| 16 | 2243 | agentic | anterior mediastinum | right perihilar region | incorrect | agree |
| 17 | 2099 | workflow | axial | Brain | incorrect | agree |
| 18 | 1921 | workflow | female | (description of the radiograph) | incorrect | agree |
| 19 | 2025 | pipeline | left aca and mca | cannot be identified from this image | incorrect | agree |
| 20 | 1945 | workflow | hydrocephalus | fluid spaces markedly enlarged | partial | agree |
| 21 | 2025 | baseline | left aca and mca | vein of Trolard and vein of Labbé | incorrect | agree |
| 22 | 2079 | workflow | mediport | circular radiopaque object over the right chest | partial | agree |
| 23 | 2092 | pipeline | 5mm | roughly 6–7 cm | incorrect | agree |
| 24 | 1954 | baseline | right posteroinferior cerebellum | left maxillary sinus | incorrect | agree |
| 25 | 2110 | pipeline | on the patient's left | yes | partial | agree |
| 26 | 1903 | pipeline | r frontal lobe | white matter of both hemispheres | incorrect | agree |
| 27 | 1875 | workflow | right frontal lobe | Brain | partial | agree (no credit either way) |
| 28 | 2151 | pipeline | flair | yes | incorrect | agree |
| 29 | 2045 | workflow | diaphragm | (chest description) | incorrect | agree |
| 30 | 2231 | pipeline | blood vessel | no discrete dense mass visible | incorrect | agree |
