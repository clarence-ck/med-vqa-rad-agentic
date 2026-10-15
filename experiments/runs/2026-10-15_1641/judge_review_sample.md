# LLM-judge spot check (run 2026-10-15_1641)

Judge: `gpt-6-astra` (low reasoning effort), blinded to which system produced each answer. 30 verdicts drawn at
random (seed 7) from all 560 judged answers and reviewed against the reference answer (AI-assisted review, not
human annotation). A verdict counts as agreed when its credit decision (correct vs not) is right; a
`partially_correct` verdict earns no credit.

**Result: 29 of 30 agreed. In the other (#22) the judge was too harsh on an agentic-loop answer.**

| # | Case | System | Question | Reference | Answer | Verdict | Review |
|---|---|---|---|---|---|---|---|
| 1 | 898 | workflow | what is most alarming about the appendix? | extraluminal air and small fluid collect | A bowel segment on the image left contains fluid with a horizontal gas-fluid level.; Contrast-density material | incorrect | Agree |
| 2 | 521 | pipeline | is a pneumonia or pleural effusion seen in this image? | pleural effusion | A right pleural effusion is present; pneumonia is uncertain. | correct | Agree |
| 3 | 1173 | agentic | which ventricles are visible? | lateral ventricles | Parts of the lateral ventricles (brain ventricles) are visible. | correct | Agree |
| 4 | 152 | baseline | what does the blunting of the costophrenic angles indicate? | pleural effusion | Blunting of the costophrenic angles typically indicates pleural effusions. | correct | Agree |
| 5 | 265 | pipeline | is the mass heterogeneous or homogeneous in appearance? | heterogeneous | Heterogeneous. | correct | Agree |
| 6 | 2180 | agentic | where is obstruction present, if any? | proximal aspect of the appendix | No definite site of obstruction is visible. | incorrect | Agree |
| 7 | 388 | agentic | what cut of the brain is this image taken? | axial | Axial (transverse) section | correct | Agree |
| 8 | 1033 | pipeline | what organ system is primarily present in this image? | gastrointestinal | Abdomen | partially_correct | Agree |
| 9 | 184 | workflow | where is the spleen located? | on the left | Upper abdomen | partially_correct | Agree |
| 10 | 1785 | workflow | where is the nodule? | below the 7th rib in the right lung. | Chest | partially_correct | Agree |
| 11 | 659 | workflow | what is your differential diagnoses? | metastases, infection/abcess, glioblasto | This is a MRI image. of the Brain. showing Multiple rounded or irregular bright-rimmed foci with darker center | incorrect | Agree |
| 12 | 113 | pipeline | where is the pathology in this image? | vasculature | Right cerebral hemisphere, near the lateral ventricle. | incorrect | Agree |
| 13 | 342 | agentic | what type of thrombosis is shown? | basilar artery thrombosis | Unable to determine from this image. | incorrect | Agree |
| 14 | 1442 | agentic | what modality is this? | chest x ray | Frontal chest X-ray (radiograph). | correct | Agree |
| 15 | 1365 | agentic | what region of the brain shows hyperintensity? | right lenticular nucleus | Right basal ganglia, particularly the lentiform region. | correct | Agree |
| 16 | 196 | workflow | is the mass in the left or right side? | left | Abdomen and pelvis | incorrect | Agree |
| 17 | 714 | pipeline | what is abnormal in this image? | hydropneumothorax | There is a left basilar pleural air–fluid level with adjacent left lower-lung opacity, consistent with a left  | correct | Agree |
| 18 | 382 | agentic | what modality is this image taken? | mri | MRI, axial T2-weighted FLAIR | correct | Agree |
| 19 | 1373 | pipeline | what is represented by the black parts of the image within the large intestine? | air | Gas (air) in the colon. | correct | Agree |
| 20 | 186 | agentic | how many gallstones are identified? | 4 | 3 | incorrect | Agree |
| 21 | 463 | pipeline | what muscle surrounds the vertebrae? | psoas major muscle | Psoas muscles | correct | Agree |
| 22 | 693 | agentic | the dark space in the central portion of the image represents what structure(s)? | lateral ventricles | Lateral ventricles and third ventricle | partially_correct | Disagree: names the reference structure plus the third ventricle, an extra rather than a wrong answer; strict non-credit. |
| 23 | 186 | workflow | how many gallstones are identified? | 4 | 2 | incorrect | Agree |
| 24 | 1173 | pipeline | which ventricles are visible? | lateral ventricles | The lateral ventricles. | correct | Agree |
| 25 | 152 | pipeline | what does the blunting of the costophrenic angles indicate? | pleural effusion | Blunting of the costophrenic angles suggests small pleural effusions. | correct | Agree |
| 26 | 679 | pipeline | the internal gas and enhancing rim suggests what type of lesion? | abcess | A gas-containing hepatic abscess. | correct | Agree |
| 27 | 138 | workflow | where are the kidneys located? | adjacent to vertebrae | Abdomen | partially_correct | Agree |
| 28 | 510 | agentic | what condition is found in the above image? | right subdural hematoma | Chronic periventricular white-matter volume loss with ex-vacuo ventriculomegaly, compatible with periventricul | incorrect | Agree |
| 29 | 819 | agentic | what is the organ pictured superior to the aorta? | horsehoe kidney | A horseshoe kidney | correct | Agree |
| 30 | 1365 | baseline | what region of the brain shows hyperintensity? | right lenticular nucleus | The right basal ganglia region shows hyperintensity. | partially_correct | Agree |
