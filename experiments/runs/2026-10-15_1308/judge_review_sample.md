# LLM-judge spot check (run 2026-10-15_1308)

Judge: `gpt-6-astra` (low reasoning effort), blinded to which system produced each answer. 30 verdicts drawn at
random (seed 7) from all 524 judged answers and reviewed against the reference answer (AI-assisted review, not
human annotation). A verdict counts as agreed when its credit decision (correct vs not) is right.

**Result: 29 of 30 agreed. The other (#26) was a lenient credit for a workflow-baseline answer.**

| # | Case | System | Question | Reference | Answer | Verdict | Review |
|---|---|---|---|---|---|---|---|
| 1 | 2089 | workflow | are the structures in the pancreas cystic or solid? | cystic | Single axial contrast-enhanced CT image through the upper abdomen. The liver, spleen, kidneys, pancreas, bowel | incorrect | Agree |
| 2 | 1918 | pipeline | how large is the mass in the right upper quadrant? | 3.4 cm | No discrete mass can be reliably measured; there is a patchy right upper lung opacity spanning roughly 3–4 cm. | incorrect | Agree |
| 3 | 2139 | agentic | what is under the right hemidiaphragm? | free air | Free intraperitoneal air (pneumoperitoneum). | correct | Agree |
| 4 | 1833 | baseline | what hypoattenuated tissue is between the abdominal wall and skin? | fat | Subcutaneous fat. | correct | Agree |
| 5 | 1864 | pipeline | what is the location of the cavitary lesion | right upper lobe | Right lung, in the perihilar to mid/lower region. | partially_correct | Agree |
| 6 | 1878 | agentic | in what plane was this image taken? | axial plane | Axial (transverse) plane. | correct | Agree |
| 7 | 2120 | pipeline | what image modality is used? | ct | CT | correct | Agree |
| 8 | 1842 | workflow | where is the tip of the line? | mid left subclavian vein | Chest | partially_correct | Agree |
| 9 | 2236 | workflow | what structure lies directly posterior to the appendix in this image? | psoas muscle | Pelvis and lower abdomen | incorrect | Agree |
| 10 | 1953 | workflow | what imaging modality was used to take this image? | xray | X-ray | correct | Agree |
| 11 | 1821 | pipeline | what bright white structures form an x? | lateral ventricles | The CSF-filled lateral ventricles. | correct | Agree |
| 12 | 1874 | agentic | from what structure is the large hypodense mass emanating from? | the base of the cecum | The appendix (appendiceal/cecal region). | partially_correct | Agree |
| 13 | 2164 | agentic | what is wrong with the patient's scan? | soft tissue mass in the region of the te | Unable to determine from this image. | incorrect | Agree |
| 14 | 2154 | agentic | what are the hyperintensities surrounding the aorta? | calcified atherosclerosis | Small aortic wall calcifications, likely atherosclerotic plaque. | correct | Agree |
| 15 | 1859 | workflow | what is the pathology? | cardiomegaly with pulmonary edema | The cardiac silhouette occupies an increased proportion of the thoracic width.; Mild bilateral haziness and pr | partially_correct | Agree |
| 16 | 1986 | pipeline | what are the hyperdense opacities located around the periphery? | ribs | They are the ribs along the periphery of the image. | correct | Agree |
| 17 | 1875 | agentic | what lobe of the brain is the lesion located in? | right frontal lobe | Right frontal lobe | correct | Agree |
| 18 | 2156 | pipeline | is there hypoinflation or hyperinflation of the lung | hyperinflation | no | incorrect | Agree |
| 19 | 1843 | agentic | where is the tip of the port-a-cath catheter located? | mid left subclavian vein | In the lower superior vena cava, near the cavoatrial junction. | incorrect | Agree |
| 20 | 1897 | pipeline | what plane is this? | axial | Axial (transverse) plane | correct | Agree |
| 21 | 1964 | agentic | is this patient male or female? | female | Unable to determine from this image. | incorrect | Agree |
| 22 | 1843 | workflow | where is the tip of the port-a-cath catheter located? | mid left subclavian vein | Chest | partially_correct | Agree |
| 23 | 2139 | pipeline | what is under the right hemidiaphragm? | free air | A thin crescent of free air beneath the right hemidiaphragm (pneumoperitoneum). | correct | Agree |
| 24 | 1833 | pipeline | what hypoattenuated tissue is between the abdominal wall and skin? | fat | Subcutaneous fat (adipose tissue) | correct | Agree |
| 25 | 1962 | pipeline | what is the hypodensity in the liver? | gallbladder | The prominent hypodensity appears to be the gallbladder, rather than a lesion within the liver. | correct | Agree |
| 26 | 1825 | workflow | is this supratentorial or infratentorial? | supratentorial | Single axial MRI image through the upper brain near the vertex. The cerebral hemispheres and cortical folds ar | correct | Disagree: describes the cerebral hemispheres but never answers supratentorial vs infratentorial; lenient credit. |
| 27 | 1912 | agentic | what type of imaging modality is used to acquire the above image? | mri - t2 weighted | MRI, axial T2-weighted | correct | Agree |
| 28 | 2043 | agentic | how was the image taken? | axial | Axial brain MRI, likely FLAIR-weighted. | correct | Agree |
| 29 | 2154 | baseline | what are the hyperintensities surrounding the aorta? | calcified atherosclerosis | They are most likely atherosclerotic calcifications in and around the aortic wall. | correct | Agree |
| 30 | 1914 | workflow | what plane is the above image acquired in? | axial | Axial CT image of the head showing the brain and surrounding skull. A distinct round hyperattenuating focus is | correct | Agree |
