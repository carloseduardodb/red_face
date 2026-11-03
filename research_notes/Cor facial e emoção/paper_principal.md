# Benitez-Quiroz, Srinivasan & Martinez (2018), PNAS 115(14):3581–3586, "Facial color is an efficient mechanism to visually transmit emotion", and related work from the Martinez lab

Primary sources:
- Article and full text: DOI [10.1073/pnas.1716084115](https://doi.org/10.1073/pnas.1716084115); [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/); PMID 29555780.
- Quotes and numbers below come from the PMC/Europe PMC full text unless another source is cited. I read the text through the Europe PMC full-text XML of PMC5889636: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5889636/fullTextXML
- I did not read the SI Appendix tables (S2–S6), which hold the per-emotion numbers.

## 1. Exact methods, data and results

### Takeaway
- Data: posed photographs of 184 people showing 18 emotion categories plus neutral (from the lab's own compound-expression database), plus the spontaneous DISFA database.
- Color features: each face was split into 126 triangular regions. Each region gave its mean and variance in two isoluminant opponent channels (red–green, yellow–blue), with shading removed, for 504 features in total.
- Classifier: regularized LDA. Color alone classified the 18 categories at 50.15% (chance 5.5%), valence at 92.93%, and the 4 DISFA emotions plus neutral at 95.53%.
- Perception: two experiments with 20 observers each. Neutral faces recolored with emotion "color models" were recognized above chance. Color congruent with the muscle expression helped, and incongruent color hurt.

### Cited Findings
**Authorship and editorial process**
- Authors: C. Fabian Benitez-Quiroz, Ramprakash Srinivasan, Aleix M. Martinez.
  - Affiliations: Dept. of Electrical and Computer Engineering and Center for Cognitive and Brain Sciences, The Ohio State University.
  - Contributions: "A.M.M., with the help of C.F.B.-Q., designed research; C.F.B.-Q. and R.S. performed research."
  - Source: [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- Editorial: edited by Rebecca Saxe (MIT); accepted by Editorial Board Member Marlene Behrmann on Feb 16, 2018; received Sept 12, 2017; PNAS Direct Submission. Note that the "R.S. is a guest editor" line in the paper refers to Rebecca Saxe, not Ramprakash Srinivasan. — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- Funding: NIH R01-DC-014498 and Human Frontier Science Program RGP0036/2016. — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Hypothesis (paper's wording)**
- "some of these computations yield facial blood flow changes unique to the category and valence of each emotion. These blood flow changes are visible as specific facial color patterns to observers, who can then successfully decode the emotion." — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- "a face can send emotion information to observers by changing the blood flow or blood composition on the network of blood vessels closest to the surface of the skin." The paper gives the redness sometimes linked to anger and paleness linked to fear as examples. — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Images and datasets**
- "We used images of 184 individuals expressing 18 emotion categories, plus neutral." The images come from Du, Tao & Martinez (2014, the compound facial expressions database) and from DISFA. The 18 categories are:
  - Basic: happy, sad, angry, disgusted, surprised, fearful.
  - Compound: happily surprised, happily disgusted, sadly fearful, sadly angry, sadly surprised, sadly disgusted, fearfully angry, fearfully surprised, fearfully disgusted, angrily surprised, angrily disgusted, disgustedly surprised.
  - Source: [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- The 184-person set is **posed** and includes both genders and many ethnicities. Spontaneous expressions came from DISFA (Denver Intensity of Spontaneous Facial Action; Mavadati et al., 2013, IEEE Trans Affect Comput 4:151–160), covering 4 emotion categories plus neutral. — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Color space**
- Features are "isoluminant opponent color" values in two channels, yellow–blue and red–green, computed from LMS cone responses with shading and luminance removed ("having eliminated all shading information"). The paper also applies the oRGB-style normalization of Bratkova et al. — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Face regions and features**
- A Delaunay triangulation of facial landmarks yields 148 triangles. Six triangles inside the mouth and 16 inside the eyes (sclera and iris) are discarded because "blood changes in these areas are not generally visible", leaving **126 local areas**.
- The authors say these regions "correspond to local areas of the network of blood vessels", which is a schematic claim (their Fig. 1).
- Features are the mean and variance of each channel in each area, giving a 504-dimensional vector.
- The landmark count is inconsistent: Results mention 87 points, Methods use r = 66.
- Source: [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Classifier**
- Linear discriminant analysis with regularization (δ = 0.01) and nearest-mean classification. This is not KSDA or a deep network.
  - Experiment 1: k-way classification with 10-fold cross-validation.
  - Experiment 2: 18 one-vs-all classifiers, with non-target samples downsampled and 1,000 repetitions.
  - Source: [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Experiment 1 (k-way, color only)**
- Mean accuracy 50.15% for the 18 categories (chance 5.5%). Every category was above chance (P < 0.001).
- Positive vs. negative valence: 92.93% (chance 50%, P < 10⁻⁹⁰).
- The authors say results "are not dependent on the skin color" (SI Fig. S3A).
- Source: [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**DISFA (spontaneous expressions)**
- k-way accuracy 95.53% (chance 20%). The authors write: "spontaneous expressions of distinct emotions are more clearly discriminated than posed expressions, as expected." — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Experiment 2 (one-vs-all)**
- Every category was above chance (P < 10⁻⁵⁰, chance 50%). Per-emotion accuracies appear only in Fig. 2B and SI Table S2.
- "the relevance of the color channels in each local face area varies across emotion categories" (SI Fig. S4A).
- The most discriminant color areas differ mostly from the areas where action units (AUs) change face shape (SI Fig. S4B).
- Source: [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Per-emotion machine accuracies (press release only)**
- Ohio State's press release gives the algorithm's per-emotion accuracy as roughly: happy ~90%, happily surprised ~85%, anger ~80%, sadness ~75%, fear ~70%, fearfully disgusted ~65%. — [OSU News, Mar 19, 2018](https://news.osu.edu/at-first-blush-you-look-happy--or-sad-or-angry/)
- These are presumably the one-vs-all (chance 50%) figures. I could not reconcile them line by line with SI Table S2. Note that the k-way average is only 50.15%.

**Experiment 3 (color added to neutral faces, no muscle movement)**
- Participants: 20 (7 women, mean age 23.92 y), screened for color blindness.
- Stimuli:
  - The neutral faces of all 184 individuals were recolored, with an average color and luminance first applied to every pixel.
  - Recoloring formula: y = x̆ + α(x̃_j − x̃_−j), with α = 1.
  - Six categories were tested to keep the session under 35 min: angry, disgusted, happy, happily disgusted, sad, fearfully surprised.
- Display: 21-inch CRT at 50 cm; block design, one emotion per block.
- Two-alternative forced choice (2AFC), target vs. non-target color: target colors were chosen above chance (P < 0.01, SI Table S4). Valence accuracy was 82.65% (P < 10⁻⁴).
- Six-alternative forced choice (6AFC): mean 32.92% (chance 16.67%). All categories were significant (P < 0.01) except happily disgusted, which was confused with happy. Valence accuracy was 72.9% (P < 10⁻⁵; SI Table S5).
- Source: [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**Experiment 4 (congruent vs. incongruent color on AU expressions)**
- Participants: 20 (8 women, mean age 24.45 y).
- Stimuli: images of the six expressions with prototypical AUs.
  - Congruent version: the target emotion's color model was added, z = x_ij + α(x̃_j − x̃_−j).
  - Incongruent version: another emotion's color model was added (k ≠ j), with α = 1.
- Task: 2AFC, asking which image more clearly conveys emotion j.
- Congruent images were preferred above chance (P < 0.01, SI Table S6). Valence accuracy was 85.01% (P < 10⁻⁵).
- The authors' conclusion: color is an "(at least partially) independent" channel, "additive to that of AUs."
- Source: [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)

**How the press release summarized it**
- The release says participants identified emotions "up to 75% of the time" from color alone. For colorized neutral faces it gives happy ~70%, sad ~75%, angry ~65%.
- It also says each emotion had a unique color pattern, "similar regardless of gender, ethnicity, or skin tone", and that incongruent colors looked "off" to participants.
- Martinez is quoted: "Admittedly, these images look weird."
- These numbers do not obviously match the paper's 6AFC mean of 32.92%. They are probably 2AFC figures, which the press did not label.
- Source: [OSU News](https://news.osu.edu/at-first-blush-you-look-happy--or-sad-or-angry/)

### Inferences
- The stimulus for the "color without muscle movement" claim is a statistical color model added at α = 1 to a neutral face, not a naturally occurring blush. The colors were therefore derived from posed photos and possibly exaggerated or idealized. Martinez's "these images look weird" points the same way.
- The perception tests used forced choice (2AFC/6AFC), which inflates accuracy relative to free labeling.
- Only 6 of the 18 categories were tested perceptually, with N = 20 per experiment, all in the lab, on a CRT monitor.
- Valence carries much of the signal: 6AFC was 32.9%, while valence ran 73–85%.
- Most of the image set is posed. Its color differences may partly reflect posing effort, AU-induced skin compression or stretching, illumination and geometry interacting with landmarks, or photographic session differences, rather than emotion-driven blood flow. The paper's design cannot separate these.
- The DISFA accuracy (95.5% for 5 classes) comes from video frames of a small number of subjects, so frame-level cross-validation could include the same identity in training and test. I did not verify this from the SI.

### Gaps
- Per-emotion accuracies from SI Tables S2–S6 (the SI was not retrieved).
- Exact DISFA sample size and cross-validation scheme (in the SI).
- Exact number of images used, which is presumably 184 × 19 for the posed set.

## 2. What the paper claims about blood flow and hemoglobin: measured or inferred?

### Takeaway
Blood flow was **never measured**: no photoplethysmography, no hyperspectral or hemoglobin imaging, no physiology. The paper infers a blood-flow and blood-composition mechanism from color statistics in ordinary RGB photos, cites prior literature, and illustrates it with a schematic vessel diagram (Fig. 1, "Copyright The Ohio State University").

### Cited Findings
- From the abstract: "the surface of the face is also innervated with a large network of blood vessels. Blood flow variations in these vessels yield visible color changes on the face. Here, we study the hypothesis that these visible facial colors allow observers to successfully transmit and visually interpret emotion even in the absence of facial muscle activation." — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- The paper says color changes "can be caused by" variations in blood flow, blood pressure and glucose levels. It cites refs 13–15 (e.g., Blake et al. 2001 on blood glucose and emotional pictures; Cahill et al. 1994 on β-adrenergic activation) as background, not as measurements in this study. — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- Discussion: the paper "studied the supplemental hypothesis that these computations result in visible facial color changes caused by emotion-dependent variations in facial blood flow. The studies reported in the present paper provide evidence favoring this hypothesis." — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- Hemoglobin is not measured or modeled. Changizi, Zhang & Shimojo (2006, Biol Lett, "Bare skin, blood and the evolution of primate colour vision") is cited (ref 25). — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- Press framing: "We believe these color patterns are due to subtle changes in blood flow or blood composition triggered by the central nervous system." — [OSU News](https://news.osu.edu/at-first-blush-you-look-happy--or-sad-or-angry/)
- The press release states the blood-flow cause as belief, not as a measurement.
- Later secondary literature cites the paper as if it established blood-flow (vasodilation/vasoconstriction) causation. Example: "Faces regularly change color with emotion due to corresponding fluctuations in cardiovascular activity (e.g., peripheral vasodilation and vasoconstriction; Benitez-Quiroz, Srinivasan, & Martinez, 2018…)." — [Thorstenson, Pazda & Krumhuber 2021 (author manuscript, UCL)](https://discovery.ucl.ac.uk/id/eprint/10137464/)

### Inferences
- The causal chain (emotion → CNS → blood flow → color) is a hypothesis consistent with the data. The data show only that region-wise chromatic statistics in posed photos differ by emotion label.
- The paper's own word "evidence favoring" is appropriately hedged. Citing works sometimes drop that hedge.

### Gaps
- I found no follow-up by this group that measured hemoglobin or perfusion directly (e.g., rPPG or spectral imaging) to validate the mechanism.

## 3. Patents, spin-offs, commercial claims, and later or related papers by the group

### Takeaway
- Conflict-of-interest statement: Ohio State filed a patent application that covers recognizing emotion from facial color.
- The patent family is US 11,314,967 B2 (granted 2022-04-26, sole inventor Aleix Martinez, assignee Ohio State Innovation Foundation, priority June 1, 2016), with JP and EP counterparts.
- OSU announced a spin-off, "Online Emotion", to commercialize the work.
- Martinez is a co-author of the 2019 Barrett et al. review, which argues facial movements are unreliable indicators of emotion. I could not verify whether that review discusses facial color.

### Cited Findings
**Conflict of interest and spin-off**
- COI statement: "The Ohio State University has submitted a patent application that includes the recognition of emotion from facial color." — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- Press release: "The researchers are patenting the computer algorithms" and "formed a spin-off company, Online Emotion, to commercialize the research." It mentions uses for future AI and "smart" cosmetics that "play up certain emotions — or conceal them." — [OSU News, Mar 19, 2018](https://news.osu.edu/at-first-blush-you-look-happy--or-sad-or-angry/); same text at [Newswise](https://www.newswise.com/articles/at-first-blush,-you-look-happy—or-sad,-or-angry) and [OSU Engineering](https://engineering.osu.edu/news/2018/03/first-blush-you-look-happy-or-sad-or-angry)

**US patent**
- US 11,314,967 B2, "System and method for recognition and annotation of facial expressions."
- Inventor Aleix Martinez; assignee Ohio State Innovation Foundation.
- Priority 2016-06-01 (provisional 62/343,994); filed 2017-06-01; granted 2022-04-26.
- Google lists it as "Active", adjusted expiration 2038-03-28.
- Published application US 2019/0294868 A1; confirmatory license to NIH dated 2019-04-10.
- The abstract describes identifying AUs and emotion categories using "facial blood flow variations that are indicative of an emotion category". The description includes color-based AU detection and a method to modify image color to change the perceived expression.
- I did not read the claims themselves.
- Sources: [Google Patents US11314967](https://patents.google.com/patent/US11314967); [Justia US20190294868](https://patents.justia.com/patent/20190294868)

**Foreign counterparts**
- JP2019517693A, granted as JP7063823B2 on 2022-05-09. Google lists it as "Expired – Fee Related". — [Google Patents JP2019517693A](https://patents.google.com/patent/JP2019517693A/en)
- EP 3465615 A1. — [EPO GPI](https://data.epo.org/gpi/EP3465615A1)

**Martinez (2017), Computational models of face perception**
- Current Directions in Psychological Science 26(3):263–269, DOI 10.1177/0963721417698535; [PMC5754021](https://pmc.ncbi.nlm.nih.gov/articles/PMC5754021/).
- It focuses on identity and expression recognition, the "inverse problem of image production", and models that learn from fewer examples. The snippets I saw did not confirm a color discussion, and I could not retrieve the full text.

**Barrett, Adolphs, Marsella, Martinez & Pollak (2019), "Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements"**
- Psychological Science in the Public Interest 20(1):1–68, DOI [10.1177/1529100619832930](https://doi.org/10.1177/1529100619832930); PMC6640856; Europe PMC citation count 492 as of 2026-11-02.
- It argues one cannot reliably infer a person's emotional state from facial movements alone.
- Sources: [SAGE](https://journals.sagepub.com/doi/abs/10.1177/1529100619832930); [Semantic Scholar](https://www.semanticscholar.org/paper/Emotional-Expressions-Reconsidered:-Challenges-to-Barrett-Adolphs/c489b6787c5af8aca97f4761343a66f3f189b35d)

**Other related lab paper**
- Srinivasan & Martinez, "Cross-Cultural and Cultural-Specific Production and Perception of Facial Expressions of Emotion in the Wild" (IEEE Trans Affect Comput; arXiv 2018). — [arXiv 1808.04399](https://arxiv.org/pdf/1808.04399)

### Inferences
- There is a tension in the lab's output. The 2018 paper and the patent claim facial color (and AUs) carry emotion-category information that a machine can decode. In 2019 Martinez co-authored a high-profile review arguing that inferring emotion from facial configurations is unreliable and context-dependent.
- The two can be reconciled: the PNAS paper studies *transmission* of signals in posed and lab-curated images, not inference of felt states in the wild. Even so, the 2019 critique's logic (reliability, specificity, generalizability, context) applies directly to color-based emotion inference.
- The commercial interest (patent plus spin-off) was disclosed and is relevant when weighing the strong press framing.

### Gaps
- I could not retrieve the 2019 review's full text (PMC blocked by captcha; not open access in Europe PMC), so I could not check whether it cites Benitez-Quiroz et al. (2018) or discusses color.
- I found no information on Online Emotion's corporate status (registration, products, whether it still exists).
- I did not verify the patent's independent claims or whether color appears in them.
- I found no peer-reviewed follow-up from the lab that extended the facial-color work, such as dynamic video, physiology, or a larger perception study. A dedicated Google Scholar check of Benitez-Quiroz's and Srinivasan's later output is still needed. An OSU ECE news item "facial colors and emotions" (Nov 2018, [ece.osu.edu](https://ece.osu.edu/news/2018/11/facial-colors-and-emotions)) returned 404.
- Martinez's later career moves (e.g., to industry) were not verified in this session.

## 4. Press coverage, critiques, commentaries, citations, replications

### Takeaway
- Coverage: wide, uncritical, mostly reproducing the OSU release (Neuroscience News, Medical News Today, Earth.com, The Asian Age, Sowetan, PsyBlog, and others).
- Critiques: I found no formal published commentary or rebuttal of the paper.
- Independent replication: the closest test is Thorstenson, Pazda & Krumhuber (2021). It found facial coloration improved emotion categorization for **dynamic** but **not static** expressions, which only partly supports the PNAS claims (the PNAS stimuli were static).

### Cited Findings
**Press**
- The OSU release (Mar 19, 2018) was syndicated widely. — [OSU News](https://news.osu.edu/at-first-blush-you-look-happy--or-sad-or-angry/); [Newswise](https://www.newswise.com/articles/at-first-blush,-you-look-happy—or-sad,-or-angry); [Neuroscience News](https://neurosciencenews.com/emotion-face-color-8665/); [Medical News Today](https://www.medicalnewstoday.com/articles/321272); [Earth.com](https://earth.com/news/emotions-revealed-facial-color); [The Asian Age](https://www.asianage.com/life/health/200318/facial-colour-can-broadcast-our-feelings-study.html); [Sowetan](https://www.sowetanlive.co.za/good-life/2018-03-26-the-colour-of-happiness-we-really-are-social-chameleons/); [PsyBlog 2024](https://www.spring.org.uk/2024/10/read-emotions-face.php)
- A lay summary, "The hidden emotions within our blood flow", appeared on The Science Breaker. — [The Science Breaker](https://thesciencebreaker.org/en/breaks/psychology/the-hidden-emotions-within-our-blood-flow)

**Citations**
- Europe PMC lists 42 citations of the PNAS paper (PMC-indexed citing works only) as of 2026-11-02. — [Europe PMC REST query on DOI 10.1073/pnas.1716084115](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1073/pnas.1716084115&format=json)
- The Google Scholar count would be much higher, but I did not retrieve it.

**Independent test: Thorstenson, Pazda & Krumhuber (2021)**
- "The influence of facial blushing and paling on emotion perception and memory", Motivation and Emotion 45:818–830, DOI [10.1007/s11031-021-09910-5](https://link.springer.com/article/10.1007/s11031-021-09910-5).
- Results: "changes in facial coloration facilitated emotion categorization accuracy in dynamic (Exp. 1) but not static expressions (Exp. 2)." Coloration increased perceived intensity, and memory for coloration was biased toward emotion prototypes.
- The authors note that posed expressions often lack natural coloration. They also say generalization across ethnicities needs more work, despite Benitez-Quiroz's claim of skin-tone independence.
- Source: [UCL Discovery manuscript](https://discovery.ucl.ac.uk/id/eprint/10137464/)

**Related independent work**
- Thorstenson et al. (2018), "Face color facilitates the disambiguation of confusing emotion expressions", found facial color inferences of emotion from neutral faces, consistent with Benitez-Quiroz. I know it only from search-snippet descriptions and did not verify the journal or details. — cited within [Thorstenson et al. 2021 manuscript](https://discovery.ucl.ac.uk/id/eprint/10137464/)
- Thorstenson, Pazda & Elliot, "Social Perception of Facial Color Appearance for Human Trichromatic Versus Dichromatic Color Vision", PSPB 2020. — [doi 10.1177/0146167219841641](https://dx.doi.org/10.1177/0146167219841641)

**Secondary critique (from a search snippet; not verified in full text)**
- A review in Evolutionary Human Sciences ("Representations of facial expressions since Darwin") reportedly notes two points: the strongest color cue comes from happy expressions (cheek blood flow and reflectance), and color-based recognition may depend on guessing valence (happy as positive vs. anger, fear, disgust and sadness as negative). — [Cambridge Core](https://www.cambridge.org/core/journals/evolutionary-human-sciences/article/representations-of-facial-expressions-since-darwin/3360FB31C8AE5CC31A0C290ADA92A8B1)
- The fetch returned HTTP 429, so I could not quote it directly.
- The valence point is consistent with the PNAS numbers: valence 73–85% vs. 6AFC 32.9%.

### Inferences
- I found no direct, same-method replication: re-deriving color models from an independent posed or spontaneous database and re-testing observers.
- The Thorstenson et al. null result for static stimuli is the most relevant counterpoint.
- The press consistently reported "up to 75%" (human) and "up to 90%" (machine) without explaining the forced-choice or one-vs-all chance levels (50%), which overstates effectiveness.

### Gaps
- Google Scholar citation count and a systematic list of citing critiques were not retrieved.
- I found no PubPeer threads, letters to PNAS, or formal commentaries. This is absence of evidence from a limited search.

## 5. Deception / lie detection

### Takeaway
The paper does **not** address deception, lie detection or concealed emotion. Neither the PNAS text nor the OSU press release mentions lying. Lie-detection claims based on facial blood flow come from a different group: Kang Lee (University of Toronto) and NuraLogix's "Transdermal Optical Imaging". I found no source that explicitly extends Benitez-Quiroz et al. (2018) to lie detection.

### Cited Findings
- A full-text search of the PNAS article for "decep", "lie", "conceal" and "hide" found no discussion of deception. — [PMC5889636](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889636/)
- The press release mentions only that "smart" cosmetics might "conceal" emotions. It says nothing about lie detection. — [OSU News](https://news.osu.edu/at-first-blush-you-look-happy--or-sad-or-angry/)
- Separate line of work: Kang Lee's Transdermal Optical Imaging (TOI) uses conventional video to estimate facial hemoglobin changes. Its premise is "basic human emotions create specific facial blood flow patterns that are beyond our control".
  - The developers claim emotions associated with lying are detectable with 85% accuracy. This is promotional material without independent validation.
  - It was commercialized via NuraLogix, with related US patents titled "Deception detection system and method" (US 10,390,747; US 10,779,760).
  - Motherboard noted that lie detectors are not proven to work at the individual level.
  - Sources: [TIAP](https://tiap.ca/2016/02/the-real-pinocchio-effect/); [Motherboard/Vice](https://motherboard.vice.com/read/nuralogix-smartphone-lie-detector-toronto); [USPTO 10390747](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10390747); [USPTO 10779760](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10779760)

### Inferences
- Any use of Benitez-Quiroz et al. (2018) to support color-based lie detection is an extrapolation the paper does not support.
- The paper used posed, intentional expressions and color models added to images. It never tested felt vs. displayed emotion, suppression, or deceptive intent.
- The 2019 Barrett et al. review co-authored by Martinez further undercuts inferring internal states from the face.

### Gaps
- I did not systematically search news or blogs for specific claims linking the OSU paper to lie detection. Such claims may exist in low-quality outlets but none were found.
