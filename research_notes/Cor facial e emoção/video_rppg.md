# Measuring facial blood flow and color signals from RGB video, and using them to recognize emotion and affect

Note on sourcing: items marked [verified] were checked by search or fetch in this session (2026-11-02). Items marked [canonical] are standard references cited by well-known DOI or arXiv ID that were not re-fetched here. The report writer should spot-check page numbers and DOIs for [canonical] items before publication.

## Q1. Remote photoplethysmography (rPPG): foundations, deep models, toolkits, datasets, accuracy and failure modes

### Takeaway
rPPG recovers the pulse from tiny periodic color changes in facial skin, strongest in the green channel. The field moved from single-channel and ICA methods (2008-2011), to skin-optics projections (CHROM 2013, POS 2017), to supervised deep networks (2018-2023). Heart rate (HR) estimation is solved well on clean lab datasets. The main failure modes are motion, lighting changes, video compression and reduced performance on the darkest skin tones. rPPG-Toolbox and pyVHR are the two main open-source frameworks.

### Cited Findings
**Foundational unsupervised methods**
- Verkruysse, Svaasand and Nelson (2008) showed that PPG signals can be recorded with an ordinary consumer camera under ambient light. The green channel carried the strongest plethysmographic signal. Opt. Express 16(26):21434. [canonical] — [DOI 10.1364/OE.16.021434](https://doi.org/10.1364/OE.16.021434)
- Poh, McDuff and Picard (2010) used blind source separation (ICA) on the mean R, G and B values of a face region from a webcam to recover HR. Opt. Express 18(10):10762. [canonical] — [DOI 10.1364/OE.18.010762](https://doi.org/10.1364/OE.18.010762). A 2011 follow-up in IEEE TBME added HRV and respiration. [canonical] — [DOI 10.1109/TBME.2010.2086456](https://doi.org/10.1109/TBME.2010.2086456)
- de Haan and Jeanne (2013), CHROM, combine chrominance signals using a skin-tone/white-balance model, which makes the pulse signal more robust to motion. IEEE TBME 60(10). [canonical] — [DOI 10.1109/TBME.2013.2266196](https://doi.org/10.1109/TBME.2013.2266196)
- Wang, den Brinker, Stuijk and de Haan (2017), POS, project the signal onto a plane orthogonal to the skin tone in temporally normalized RGB space. IEEE TBME 64(7). [canonical] — [DOI 10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282)

**Deep models**
- DeepPhys (Chen and McDuff, ECCV 2018) is a convolutional attention network that takes normalized frame differences (motion) and appearance branches as input. [canonical] — [arXiv 1805.07888](https://arxiv.org/abs/1805.07888)
- PhysNet (Yu, Li and Zhao, BMVC 2019) is a spatio-temporal 3D-CNN that reconstructs the rPPG waveform end to end. The authors report using the reconstructed signal for atrial fibrillation detection and emotion recognition. [verified] — [arXiv 1905.02419](https://alphaxiv.org/abs/1905.02419)
- PhysFormer (Yu et al., CVPR 2022) is a temporal-difference video transformer. [canonical] — [arXiv 2111.12082](https://arxiv.org/abs/2111.12082)
- EfficientPhys (Liu et al., WACV 2023) is a lightweight model that needs no face detection or segmentation preprocessing and is aimed at on-device use. [canonical] — [arXiv 2110.04447](https://arxiv.org/abs/2110.04447)
- Recent models (2024-2026), such as Mamba-based PhysMamba and FactorizePhys, are benchmarked against DeepPhys and EfficientPhys on UBFC-rPPG and PURE in 2026 papers. [verified] — [J. Computer Applications 2026, 46(8):2699](https://www.joca.cn/EN/Y2026/V46/I8/2699); [Heartian, arXiv 2609.28539](https://arxiv.org/pdf/2609.28539)

**Toolkits**
- rPPG-Toolbox (Liu et al., NeurIPS 2023 Datasets and Benchmarks track) provides unsupervised methods (e.g., ICA, CHROM, POS) and supervised models (e.g., DeepPhys, TS-CAN, PhysNet, PhysFormer, EfficientPhys), with loaders for SCAMPS, UBFC-rPPG, PURE, BP4D+, UBFC-Phys and MMPD. [verified] — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d7d0d548a6317407e02230f15ce75817-Abstract.html); [GitHub mirror listing datasets](https://github.com/VikashPR/rPPG-Toolbox). The official repository is github.com/ubicomplab/rPPG-Toolbox. [canonical]
- pyVHR (Boccignone et al., PeerJ Computer Science 8:e929, 2022) is a multi-stage Python pipeline covering traditional and learning-based rPPG. It is GPU-accelerated, runs in real time on 30 fps HD video with about a 5x speedup, is licensed GPL-3.0 and installs with pip (`pyvhr` / `pyvhr-cpu`). [verified] — [PeerJ](https://peerj.com/articles/cs-929); [PMC9044207](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207)

**Datasets** [canonical unless noted]
- UBFC-rPPG (Bobbia et al., Pattern Recognit. Lett. 2019): webcam video with a pulse-oximeter reference, mostly still subjects. — [DOI 10.1016/j.patrec.2017.10.017](https://doi.org/10.1016/j.patrec.2017.10.017)
- PURE (Stricker, Müller and Gross, RO-MAN 2014): 10 subjects performing controlled head-motion tasks, recorded as losslessly stored frames. — [DOI 10.1109/ROMAN.2014.6926392](https://doi.org/10.1109/ROMAN.2014.6926392)
- MAHNOB-HCI (Soleymani et al., IEEE TAC 2012): emotion-elicitation videos with ECG. The video is heavily compressed, which makes rPPG harder. — [DOI 10.1109/T-AFFC.2011.25](https://doi.org/10.1109/T-AFFC.2011.25)
- VIPL-HR (Niu et al., ACCV 2018 / RhythmNet, IEEE TIP 2020): 107 subjects recorded with multiple devices under varied head pose and lighting. — [arXiv 1810.04927](https://arxiv.org/abs/1810.04927)
- UBFC-Phys (Meziati Sabour et al., IEEE TAC 2021): stress dataset, see Q3. [verified]

**Failure modes**
- Skin tone: Nowara, McDuff and Veeraraghavan (CVPRW 2020) ran a meta-analysis over 3 datasets, 73 people and more than 400 videos. Average performance drops significantly for the darkest skin type, and skin type does not affect all methods equally. [verified] — [CVF open access](https://openaccess.thecvf.com/content_CVPRW_2020/html/w19/Nowara_A_Meta-Analysis_of_the_Impact_of_Skin_Tone_and_Gender_CVPRW_2020_paper.html); [NSF PAR](https://par.nsf.gov/biblio/10217885)
- Dasari et al. (npj Digital Medicine 2021) evaluated demographic bias in rPPG pulse-rate estimation. They note that public datasets are recorded in labs and lack diversity in skin tone and age. [verified] — [DOAJ record](https://doaj.org/article/250d736043c9468b828363cca852ddc6)
- A 2026 BMJ Open field study in Nigeria found rPPG blood-pressure screening feasible but with inadequate performance and critically low sensitivity in a darker-skinned cohort. Other studies, such as a 2023 Cureus paper, report no skin-tone effect, so the literature conflicts. [verified] — [BMJ Open 2026](https://bmjopen.bmj.com/content/16/6/e119311); [Biosensors 2025 review](https://www.mdpi.com/2079-6374/15/7/423/html)
- Compression and motion: see Q4.

### Inferences
- POS and CHROM need no training and remain strong baselines on still, well-lit faces. Supervised models mainly help with motion and compression, but often generalize poorly across datasets.
- Reported mean absolute errors (MAE) of about 1-2 bpm on UBFC-rPPG and PURE are clean-lab numbers and should not be expected on in-the-wild video.

### Gaps
- I did not retrieve the exact rPPG-Toolbox benchmark tables (MAE, RMSE, correlation per method and dataset). The search returned only the abstract. These numbers can be taken from the paper PDF ([arXiv 2210.00716](https://arxiv.org/abs/2210.00716)).
- The gender findings of Nowara 2020 were not available in the abstract text I retrieved.
- I could not find a confirmed 2024-2026 comprehensive survey in this session's searches.

## Q2. Transdermal optical imaging (Kang Lee, University of Toronto / NuraLogix)

### Takeaway
Transdermal optical imaging (TOI) splits camera color channels into "bitplanes" that the authors claim reflect hemoglobin rather than melanin. Machine learning on regional facial blood-flow signals is then used to infer vital signs, blood pressure and emotion. The peer-reviewed evidence includes a small proof-of-concept physiology study (n=11), a large blood-pressure validation (n=1,328) in controlled settings, and a 2022 emotion paper. Lie-detection accuracy claims come from talks and press coverage, not peer review. The method is proprietary (patents, NuraLogix's commercial SDK).

### Cited Findings
- Liu, Luo, Zheng, Wu and Lee, "Transdermal optical imaging revealed different spatiotemporal patterns of facial cardiovascular activities," Sci. Rep. 8:10588 (2018). [verified] — [PMC6043515](https://pmc.ncbi.nlm.nih.gov/articles/PMC6043515), [DOI 10.1038/s41598-018-28804-0](https://doi.org/10.1038/s41598-018-28804-0). Setup and results:
  - Camera: Pike F-421 machine-vision camera at 20 fps and 910x800 resolution, about 50 cm from the face, with two LED lights in a dark room.
  - Subjects: 11 adults sitting still with eyes closed. About 51 s (1,024 frames) were analyzed against a 200 Hz ECG.
  - Pipeline: bitplanes reflecting hemoglobin were averaged over 10 facial ROIs, then group temporal ICA was applied.
  - Result: 5 independent components with distinct spatial patterns (bilateral forehead, left chin, left cheek, left forehead). Their magnitude spectra correlated with ECG but their phase spectra did not.
  - The authors argue that averaging the whole face hides regional differences, and they call for replication across skin tones, cameras and lighting.
- Luo, Yang, Barszczyk et al., "Smartphone-Based Blood Pressure Measurement Using Transdermal Optical Imaging Technology," Circ. Cardiovasc. Imaging 12(8), 2019. [verified] — [DOI 10.1161/CIRCIMAGING.119.008857](https://doi.org/10.1161/CIRCIMAGING.119.008857); [UPI/HealthDay coverage](https://www.upi.com/Health_News/2019/08/06/Smartphone-videos-may-one-day-measure-blood-pressure/2831565099086/). Key points:
  - 1,328 Canadian and Chinese adults were recorded in 2-minute iPhone videos and compared with a continuous cuff device.
  - Reported accuracy was about 95% for systolic and about 96% for diastolic/pulse pressure.
  - Limitations stated in the coverage: normal or slightly raised BP only, mostly East Asian and European participants, highly controlled recording environment.
- "Discrete emotions discovered by contactless measurement of facial blood flows," Cognition and Emotion 36(7):1429-1439 (2022). Each emotion-evoking film produced a distinct facial blood-flow pattern. A related abstract reports above-chance discrimination for joy, sadness and disgust using pulse-band data. [verified, abstract level only] — [PhilPapers record](https://api.philpapers.org/rec/FUDED); [IGEL abstract](https://discourse.igelsociety.org/t/contactless-measurement-of-emotion-physiology-during-short-films/199)
- Lie detection: talks and press claim 85% accuracy and a "Pinocchio effect" (cheek blood flow falls, nose blood flow rises when lying). These are not peer-reviewed findings in the sources I found. [verified as claims] — [AUTM Better World story](https://autm.net/about-tech-transfer/better-world-project/bwp-stories/transdermal-optical-imaging-%20(toi)); [U of T News](https://www.utoronto.ca/news/professor-ted-speaker-kang-lee-his-new-startup-will-change-how-we-detect-emotion); [NPR transcript](https://www.npr.org/transcripts/591879723)
- NuraLogix patents describe "detecting invisible human emotion" from video (e.g., US 10,705,603, retail environment). [verified] — [USPTO 10705603](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10705603)

### Inferences
- The bitplane-selection step is a learned proprietary filter, so it cannot be reproduced exactly. An open approximation is hemoglobin-map extraction (Tsumura ICA, Q3) plus regional rPPG amplitude and DC-level features.
- The strongest TOI results came from controlled lighting and still subjects, and emotion effects were found at group level. Per-person, per-trial emotion classification from facial color alone has not been shown robustly.

### Gaps
- I could not retrieve the 2016 Frontiers / Psychological Science-era TOI emotion papers, or quantitative accuracy numbers for the 2022 emotion paper (full text blocked, HTTP 403).
- I found no independent (non-Lee-lab) replication of TOI emotion claims.

## Q3. Spatial blood-flow maps, hemoglobin maps from RGB (Tsumura), and emotion or stress inference from facial color, rPPG and HRV

### Takeaway
Two complementary signals can be extracted from facial color:
- Slow, spatial hemoglobin (blushing/pallor) maps, obtained by melanin/hemoglobin separation in optical-density space (Tsumura).
- Pulsatile rPPG/HRV signals, from which stress and arousal can be inferred moderately well. The best documented benchmark is UBFC-Phys: 85.5% stress-state accuracy from remote pulse-rate-variability features.

Emotion recognition from rPPG alone is mostly limited to arousal and stress, not discrete emotions.

### Cited Findings
- Tsumura et al. (JOSA A 1999) applied ICA to skin color images. The model assumes skin color variation comes from two spatially independent pigments, melanin and hemoglobin, in the optical-density (−log) domain. [verified] — [Tsumura JOSA 1999 page](https://www.mi.tj.chiba-u.jp/~tsumura/Tsumura/papers/JOSA99_ICA.htm)
- Tsumura et al., "Image-based skin color and texture analysis/synthesis by extracting hemoglobin and melanin information in the skin," ACM TOG 22(3):770-779 (SIGGRAPH 2003). It first removes shading by color-vector analysis in the optical-density domain (inverse lighting), then separates melanin and hemoglobin and can resynthesize skin with altered pigment amounts. [verified] — [paper page](https://www.mi.tj.chiba-u.jp/~tsumura/Tsumura/papers/siggraph2003.htm); [PDF](https://www.mi.tj.chiba-u.jp/~tsumura/Tsumura/papers/TsumuraSig03.pdf)
- Tsumura's lab has continued related work estimating psychological state from facial hemoglobin. Example: a 2022 AROB paper by Masui et al., and Japanese patents titled "estimating psychological state of user." [verified, existence only] — [ArobMasui2022.pdf](https://www.mi.tj.chiba-u.jp/~tsumura/Tsumura/papers/ArobMasui2022.pdf); [USPTO 11557148](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11557148)
- UBFC-Phys (Meziati Sabour et al., IEEE TAC 2021) is a protocol inspired by the Trier Social Stress Test, with 56 participants doing rest, speech and arithmetic tasks. It includes video plus wristband BVP and EDA, and audio was removed for privacy. Stress-state recognition reached 85.48% accuracy using remote PRV features. [verified] — [lab page](https://lead.ube.fr/?p=9490); [UBFC data portal, DOI 10.25666/dataubfc-2022-05-05](https://search-data.ubfc.fr/FR-18008901306731-2022-05-05); IEEE DataPort DOI 10.21227/5da0-7344
- McDuff, Gontarek and Picard (EMBC 2014), "Remote measurement of cognitive stress via heart rate variability": camera-derived HRV distinguished rest from a cognitive (arithmetic) task. [canonical] — [DOI 10.1109/EMBC.2014.6944243](https://doi.org/10.1109/EMBC.2014.6944243)
- A remote-HRV study for emotional-state monitoring (HAL) reports rHRV estimation accuracy above 96% in the frequency domain, and a correlation between rHRV and emotional state validated in 12 of 16 volunteers. [verified] — [HAL hal-01678244](https://hal.archives-ouvertes.fr/hal-01678244)
- Yu, Li and Zhao, "Facial-video-based physiological signal measurement: recent advances and affective applications," IEEE Signal Processing Magazine 2021. It reviews rPPG methods, datasets and rPPG-based affective computing, and compares them with other affective modalities. [verified] — [Oulu repository](https://urn.fi/URN:NBN:fi-fe2021120959958)
- Robust HRV from facial video (Odinaev et al., Bioengineering 2023) uses a face detection and tracking pipeline. [verified] — [PMC10376629](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10376629/)
- Liu et al. 2018 (TOI, Q2) show that facial pulsatile blood flow differs by region (forehead, cheeks, chin), which supports spatial maps over whole-face averages. [verified] — [PMC6043515](https://pmc.ncbi.nlm.nih.gov/articles/PMC6043515)

### Inferences
- For a "red_face" project, the slow DC hemoglobin component (blushing, flushing, pallor over seconds to minutes) is likely more relevant than the pulse. It can be estimated per pixel with Tsumura-style separation. It is highly sensitive to illumination and white balance, so the camera needs to be locked or the image color-calibrated.
- The arousal/stress axis is where camera physiology has evidence. Valence and discrete emotions mainly come from facial expression (action units), so fusing color with expression is the realistic design.

### Gaps
- I did not find quantitative benchmarks on MAHNOB-HCI or DEAP-style emotion classification from rPPG alone in this session.
- I found no recent (2024-2026) systematic review specific to rPPG-based emotion recognition.

## Q4. Practical constraints: white balance and auto-exposure, compression, frame rate, lighting, ROI tracking, SNR

### Takeaway
Compression is the biggest hidden problem: even light H.264/H.265 compression considerably reduces blood-volume-pulse (BVP) SNR. In contrast, dropping from 120 to 30 fps or halving resolution hardly matters for HR. Camera auto white balance, auto exposure and auto gain introduce color drift that can mask or imitate blood-color changes, especially for slow DC "redness" analysis. Use locked camera settings, steady diffuse lighting, lossless or high-bitrate recording, and landmark-based skin ROIs.

### Cited Findings
- McDuff, Blackford and Estepp (IEEE FG 2017) systematically compared x264 and x265 encoders at different CRF values:
  - Compression, even at a low CRF, degrades BVP SNR considerably.
  - Reducing frame rate from 120 Hz to 30 Hz or halving resolution had little impact on pulse-rate accuracy.
  - x265 handled larger head motion better.
  [verified] — [Microsoft Research page](https://microsoft.com/en-us/research/?p=707122); [PDF](https://microsoft.com/en-us/research/uploads/prod/2020/11/McDuff_2017_Impact2.pdf)
- Nowara, McDuff and Veeraraghavan proposed a supervised deep approach to counter compression artifacts (ICCVW 2019) and extended the systematic compression analysis (Biomedical Optics Express 2020). [verified] — [ICCVW 2019 PDF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVPM/Nowara_Combating_the_Impact_of_Video_Compression_on_Non-Contact_Vital_Sign_ICCVW_2019_paper.pdf); [BOE 2020, DOI 10.1364/BOE.408471](https://doi.org/10.1364/BOE.408471) [canonical DOI]
- Later work (2024) targets pulse-signal magnification in compressed videos. [verified] — [arXiv 2405.02652](https://ar5iv.labs.arxiv.org/html/2405.02652); see also [Zhao et al. CVPRW 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Zhao_A_Novel_Framework_CVPR_2018_paper.pdf)
- Lighting and skin pigmentation both significantly affect video PPG accuracy. Better lighting reduces, but may not eliminate, skin-tone-related discrepancies. [verified] — [Biosensors 2025 review](https://www.mdpi.com/2079-6374/15/7/423/html)
- Lab-grade TOI used a machine-vision camera with controlled LED lighting in a dark room. [verified] — [PMC6043515](https://pmc.ncbi.nlm.nih.gov/articles/PMC6043515)
- ROI tracking: MediaPipe Face Mesh (Kartynnik et al. 2019) gives 468 3D landmarks in real time on mobile. It is commonly used to define forehead and cheek skin ROIs and exclude eyes and mouth. [canonical] — [arXiv 1907.06724](https://arxiv.org/abs/1907.06724)
- McDuff's survey (arXiv 2111.11547, published in ACM Computing Surveys 2023) covers measurable signals, methods, applications and open challenges. A companion resource site lists datasets and code. [verified for arXiv; venue and DOI 10.1145/3558518 canonical, not confirmed by search] — [arXiv 2111.11547](https://arxiv.org/pdf/2111.11547); [cameravitals.github.io](https://cameravitals.github.io/)

### Inferences
- Recommended capture settings:
  - Lock white balance, exposure, gain and focus (e.g., using v4l2-ctl on Linux webcams).
  - Use constant, diffuse, flicker-free light. Avoid mains-frequency flicker from 50/60 Hz lights, which aliases into the signal.
  - Use 30 fps or more.
  - Record lossless (FFV1, PNG frames) or very high-bitrate video (CRF ≤ 10–15 or "lossless" x264 `-qp 0`). Avoid re-encoded social-media or YouTube video for subtle color work.
- Recommended pipeline:
  1. Detect landmarks with MediaPipe and build skin masks per ROI (forehead, left/right cheek, nose, chin).
  2. Compute per-ROI spatial mean RGB (and optionally a per-pixel optical-density map).
  3. Normalize illumination, either by dividing by a temporal moving mean or by using a gray or skin reference patch.
  4. Pulse branch: run POS or CHROM (and a deep model from rPPG-Toolbox if training data exist), band-pass 0.7–4 Hz, then estimate HR, peak-detect IBIs and compute HRV (RMSSD, LF/HF).
  5. Color branch: run Tsumura melanin/hemoglobin separation in −log RGB space and track the slow hemoglobin index per ROI (low-pass below 0.1 Hz) as the "redness/flushing" signal.
  6. Quality gating: per-window SNR (power in the HR peak ±0.1 Hz versus the rest of the 0.7–4 Hz band), a motion threshold from landmark velocity, and rejection of windows where camera auto-settings changed.
  7. Fuse with facial action units (e.g., OpenFace or py-feat) for valence, and use physiological features for arousal and stress. Validate on UBFC-rPPG/PURE (pulse) and UBFC-Phys (stress).
- Expect lower SNR and possible bias for darker skin tones. Report results stratified by Fitzpatrick or Monk skin-tone scale.

### Gaps
- I found no source specifying a minimum SNR threshold for reliable HRV, as opposed to HR. HRV is known to need cleaner signals and longer windows (≥ 1–5 min), but I did not verify a numeric threshold this session.
- I found no peer-reviewed quantification of how auto white balance and auto exposure affect slow DC facial-redness measurement specifically. This is an inference from camera behavior and the Tsumura model.
- Open-source implementations of Tsumura separation exist in scattered repositories, but I did not verify a maintained package.
