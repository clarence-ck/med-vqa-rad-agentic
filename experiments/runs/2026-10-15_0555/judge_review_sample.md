# LLM-judge spot check (run 2026-10-15_0555)

Judge: `gpt-6-astra` (low reasoning effort), blinded to which system produced each answer. 30 verdicts drawn at
random (seed 7) from all 524 judged answers and reviewed against the reference answer (AI-assisted review, not
human annotation). A verdict counts as agreed when the credit decision (correct vs not) is right.

**Result: 28 of 30 agreed; 1 borderline and 1 too lenient, both on workflow-baseline answers, so neither affects
the agentic vs single-prompt comparison.**

| # | Case | System | Question | Reference | Answer | Verdict | Review |
|---|---|---|---|---|---|---|---|
| 1 | 2089 | workflow | are the structures in the pancreas cystic or solid? | cystic | Axial contrast-enhanced CT image through the upper abdomen. A rounded soft-tissue-attenuation structure is vis | incorrect | Agree |
| 2 | 1918 | pipeline | how large is the mass in the right upper quadrant? | 3.4 cm | No discrete mass size can be determined; only a small linear opacity is visible in the right upper lung near t | incorrect | Agree |
| 3 | 2139 | agentic | what is under the right hemidiaphragm? | free air | Free intraperitoneal air (pneumoperitoneum) | correct | Agree |
| 4 | 1833 | baseline | what hypoattenuated tissue is between the abdominal wall and skin? | fat | Subcutaneous fat. | correct | Agree |
| 5 | 1864 | pipeline | what is the location of the cavitary lesion | right upper lobe | Right upper lobe of the lung. | correct | Agree |
| 6 | 1878 | agentic | in what plane was this image taken? | axial plane | Axial (transverse) plane. | correct | Agree |
| 7 | 2120 | pipeline | what image modality is used? | ct | CT | correct | Agree |
| 8 | 1842 | workflow | where is the tip of the line? | mid left subclavian vein | Chest | partially_correct | Agree |
| 9 | 2236 | workflow | what structure lies directly posterior to the appendix in this image? | psoas muscle | Pelvis and lower abdomen | incorrect | Agree |
| 10 | 1953 | workflow | what imaging modality was used to take this image? | xray | X-ray | correct | Agree |
| 11 | 1821 | pipeline | what bright white structures form an x? | lateral ventricles | The lateral ventricles | correct | Agree |
| 12 | 1874 | agentic | from what structure is the large hypodense mass emanating from? | the base of the cecum | Unable to determine from this image. | incorrect | Agree |
| 13 | 2164 | agentic | what is wrong with the patient's scan? | soft tissue mass in the region of the te | Clustered small-bowel loops with a whorled appearance of the mesentery, concerning for bowel twisting or an in | incorrect | Agree |
| 14 | 2154 | agentic | what are the hyperintensities surrounding the aorta? | calcified atherosclerosis | Pancreatic calcifications, suggestive of chronic pancreatitis. | incorrect | Agree |
| 15 | 1859 | workflow | what is the pathology? | cardiomegaly with pulmonary edema | Cardiac silhouette appears enlarged.; Mild bilateral perihilar and lower lung hazy/interstitial opacities, sli | correct | Borderline: describes enlarged heart and hazy lungs without naming edema; lenient credit. |
| 16 | 1986 | pipeline | what are the hyperdense opacities located around the periphery? | ribs | Ribs along the body wall. | correct | Agree |
| 17 | 1875 | agentic | what lobe of the brain is the lesion located in? | right frontal lobe | Right frontal lobe | correct | Agree |
| 18 | 2156 | pipeline | is there hypoinflation or hyperinflation of the lung | hyperinflation | no | incorrect | Agree |
| 19 | 1843 | agentic | where is the tip of the port-a-cath catheter located? | mid left subclavian vein | The tip projects over the lower superior vena cava, near the cavoatrial junction. | incorrect | Agree |
| 20 | 1897 | pipeline | what plane is this? | axial | Axial plane | correct | Agree |
| 21 | 1964 | agentic | is this patient male or female? | female | Likely female. | correct | Agree |
| 22 | 1843 | workflow | where is the tip of the port-a-cath catheter located? | mid left subclavian vein | Chest | partially_correct | Agree |
| 23 | 2139 | pipeline | what is under the right hemidiaphragm? | free air | A crescent of free air beneath the right hemidiaphragm (pneumoperitoneum). | correct | Agree |
| 24 | 1833 | pipeline | what hypoattenuated tissue is between the abdominal wall and skin? | fat | Subcutaneous fat | correct | Agree |
| 25 | 1962 | pipeline | what is the hypodensity in the liver? | gallbladder | The prominent fluid-density area at the liver margin is most consistent with the gallbladder, not a definite l | correct | Agree |
| 26 | 1825 | workflow | is this supratentorial or infratentorial? | supratentorial | Axial MRI image through the cerebral hemispheres, with bright signal in the cortical sulci and along the midli | correct | Disagree: describes the cerebral hemispheres but never answers supratentorial vs infratentorial; should not get credit. |
| 27 | 1912 | agentic | what type of imaging modality is used to acquire the above image? | mri - t2 weighted | MRI, axial T2-weighted | correct | Agree |
| 28 | 2043 | agentic | how was the image taken? | axial | Axial brain MRI, likely FLAIR-weighted. | correct | Agree |
| 29 | 2154 | baseline | what are the hyperintensities surrounding the aorta? | calcified atherosclerosis | They are likely atherosclerotic calcifications in the aortic wall. | correct | Agree |
| 30 | 1914 | workflow | what plane is the above image acquired in? | axial | Axial CT image of the head showing the brain and surrounding calvarium. A rounded, markedly hyperattenuating f | correct | Agree |
