**Paper ID:** 50
**Title:** From Pixels to Diagnosis: Using Machine Learning to Classify Medical Image Sequences
**Authors:** A. Badilla-Olivas, Enrique Vílchez-Lizano, Brandon Mora-Umaña, Kenneth Villalobos-Solís, Adrián Lara Petitdemange

To everyone who reviewed our paper,
Thank you very much for taking the time to read and give feedback on our paper submission for JoCICI 2026. We really appreciate your work and contribution to the technological growth of the country and the promotion of research.
We are very happy that you liked our work and decided to accept it for the conference. In this document we will address the questions and requests mentioned by our reviewers. Once again, thank you for your contribution and feedback.
Best regards.

**Reviewer 3, in Question 7:**
**Comment:** "Dataset limitado. No se exploran técnicas para mitigar desbalance. El modelo ConvNeXt recibe más muestras individuales que ViVit, lo que podría introducir sesgo."
**Response:** Although ConvNeXt receives more individual samples than ViVit, the number of outputs is the same for both models. The difference is that for ConvNeXt, inputs must be passed individually, but a voting mechanism is subsequently applied that unifies these inputs into a single output per patient. Both models ultimately produce one prediction per patient, and both are evaluated on the same test patients.

We recognise that the dataset is limited and that the class distribution differs between models. However, this imbalance is not an oversight — it is an inherent consequence of how each model processes the data. ViVit operates at the patient level (roughly balanced), while ConvNeXt operates at the slice level (approximately 90% negative). This asymmetry is itself one of the findings of the study. No class-balancing techniques were applied because the goal was to compare both paradigms under natural conditions. Future work could incorporate methods such as oversampling, data augmentation, or weighted loss functions to explore whether such techniques close the performance gap.
**Correction in the paper:** The conclusions section has been rewritten to explicitly discuss the inherent class imbalance and its role in the results. A discussion paragraph has also been added to the results section.

---

**Reviewer 4, in Question 5:**
**Comment:** "La comparación no es completamente controlada, ya que no se analiza si ambos modelos reciben cantidades equivalentes de información o si existen diferencias relevantes en el volumen efectivo de datos de entrenamiento. Además, el estudio se limita a una única arquitectura por categoría, lo que reduce la generalidad de las conclusiones. La falta de análisis de coste computacional y latencia limita su aplicabilidad clínica."
**Response:** Regarding the equivalence of information: both models received data from the same 82 patients with the same train/test split. ViVit processed 18-frame sequences per patient (75 total training samples), while ConvNeXt processed individual slices (approximately 2,800 training samples). This difference in volume and class balance is inherent to each paradigm — a video model natively consumes sequences, while an image model requires decomposition into individual frames. We view this asymmetry as part of the comparison rather than a confound, since it reflects how each model would be used in practice. The conclusions have been revised to explicitly acknowledge this asymmetry and to frame the findings as specific to ViVit and ConvNeXt on this dataset, rather than general claims about video models versus image models.

Regarding the single architecture per category, we agree this limits generalisability. We have tempered our conclusions accordingly and added recommendations for future work to evaluate multiple architectures, including ensemble and multi-view models.

Regarding computational cost and latency, a table with training time, model parameters, and throughput (samples per second) for both models has been added to the experiments section. Both models were trained on a Tesla V100S-PCIE-32GB. ViVit trained at ~0.49 samples/sec (87.6M parameters, ~39 min/run) while ConvNeXt trained at ~8.6 samples/sec (27.8M parameters, ~42 min/run).
**Correction in the paper:** A computational cost table has been added to the experiments section. The conclusions have been tempered to avoid overgeneralisation.

---

**Reviewer 4, in Question 7:**
**Comment:** "No se analiza en profundidad el efecto del desbalance de clases ni se reportan métricas por clase o por subtipo de hemorragia. Tampoco se discute cómo la selección de secuencias o el muestreo de cortes puede influir en los resultados."
**Response:** First, class imbalance makes the classification task more difficult for the models; however, the focus of this research is not on class balancing, but rather on exploring the models' capabilities to process image sequences. We recognise that this may affect the results, so we have added a discussion paragraph that describes in detail how the different effective class distributions impact each model's performance.

Regarding per-class metrics: given that this is binary classification, the reported metrics (accuracy, precision, recall) together with the confusion matrices provide a complete picture of performance on both classes. Precision captures the positive prediction quality, recall captures the positive detection rate, and the confusion matrix shows all four outcomes explicitly.

Regarding hemorrhage subtypes: for the scope of this study, only binary classification was considered. The problem of an image sequence as input already imposes complexity on the models. Future studies could build upon these results and explore a multiclass classification problem.
**Correction in the paper:** A discussion paragraph has been added to the results section analysing the effect of class imbalance. The conclusions explicitly discuss this as an inherent property of the paradigm comparison.

---

**Reviewer 5, in Question 9:**
**Comment:** "Debe matizar las afirmaciones realizadas en las conclusiones, ya que aunque son razonables, deberían interpretarse con cautela debido a la comparación limitada a una sola arquitectura por tipo de modelo. Debe referirse explícitamente al uso de IA en la producción del artículo, en particular las secciones de background, related work y conclusiones."
**Response:** Agree.
**Correction in the paper:** The conclusions section has been rewritten to encourage a cautious interpretation. An "Acknowledgements" section has been added to disclose the use of AI tools.

---

**Changes made to the paper:**

*Conclusions*

- Before:

> This section outlines the key findings from our experiments, highlights the main contributions and impact of this study on both ICH classification and the use of video models in medical contexts, and provides recommendations for future research aimed at building upon these findings.
>
> ViVit's superior performance in accuracy and recall can be attributed to its ability to model temporal information inherent in video sequences. This temporal dimension likely played a significant role in ViVit's strong classification results, as it was able to capture the dynamics of ICH progression across time, which may be challenging for models that only analyze individual frames.
>
> On the other hand, ConvNeXt, trained on individual images rather than image sequences, faced challenges due to the imbalanced nature of the dataset. The model was exposed to a disproportionate number of negative cases compared to positive ones, which likely contributed to its lower performance in terms of recall and overall misclassification of ICH cases. This imbalance could be alleviated by adopting methods such as data augmentation, oversampling, or using a loss function that accounts for class imbalance.
>
> Even state-of-the-art image models like ConvNeXt can struggle with tasks requiring temporal understanding. ConvNeXt, optimized for single-image classification, is not inherently suited for image sequence analysis. This highlights a key limitation when applying standard image models to dynamic, time-dependent tasks like medical sequence classification. Future research may benefit from focusing on models designed for temporal information, such as video models or multi-view models.
>
> To overcome the challenges observed in this study, future experiments should explore alternative approaches. One direction is to use ensemble models or multi-view architectures that can better handle the temporal and spatial complexities of medical image sequences. Additionally, incorporating larger and more diverse datasets with a greater number of patients would help to improve model generalization and robustness.
>
> Given the potential of video models like ViVit, it would be worthwhile to repeat the experiment using other video-based models and compare their performance against more appropriate models designed for the task. An ensemble or multi-view approach might better suit the needs of medical image sequence classification, providing more robust results and reducing the inherent challenges posed by imbalanced data.
>
> In summary, this study demonstrates the promise of using video models for medical image sequence classification, while also highlighting some limitations of current image-based models like ConvNeXt in this context. Future work should address data imbalance, leverage the temporal dimension of medical sequences, and explore alternative model architectures, such as multi-view or ensemble approaches, to further enhance performance in medical sequence classification tasks.

- After:

> This section outlines the key findings from our experiments and provides recommendations for future research.
>
> On this dataset, ViVit achieved higher accuracy and recall than ConvNeXt. As discussed in the results section, this performance gap is likely tied to the different effective class distributions each model faced during training. The asymmetry is not a flaw in the experimental design but an inherent consequence of how each paradigm processes sequential data.
>
> These findings should be interpreted with caution. The study is limited to a single dataset of 82 patients from one hospital, and only one architecture was evaluated per model category. The results cannot be generalised to all video models or all image models. They do, however, suggest that models which natively process sequences may hold an advantage for medical image sequence classification, precisely because they avoid the class imbalance that arises when sequences are decomposed into individual frames.
>
> No class-balancing techniques (oversampling, augmentation, or weighted loss functions) were applied in this study, as the goal was to compare both paradigms under natural conditions. Future work could explore whether such techniques close the performance gap between image and video models, which would help disentangle the effect of architecture from the effect of class distribution.
>
> Other directions for future research include: evaluating additional architectures per category (such as ensemble or multi-view models), testing on larger and more diverse datasets, and exploring multiclass classification of hemorrhage subtypes. Computational cost and inference latency should also be considered in studies targeting clinical deployment.

- Reason for the change: Reviewer 5 requested that conclusions be tempered given the limited comparison to a single architecture per model category. Reviewer 4 noted the uncontrolled information asymmetry between models. The rewrite acknowledges these limitations, frames findings as specific to this dataset, and avoids causal claims. The detailed explanation of the class imbalance now lives in the results section, and the conclusions reference it rather than repeating it.

---

*Discussion (added to Results and Analysis)*

- Before: The section ended with the confusion matrix comparison, with no discussion of the factors behind the results.

- After: The following paragraph was added at the end of the Results and Analysis section:

> These results should be considered in light of the different effective class distributions each model faced during training. ViVit received patient-level sequences where the class ratio was approximately 1:1.1 (positive to negative), while ConvNeXt received individual slices where approximately 90% were negative. This imbalance is inherent to how each model consumes the data, not a flaw in the experimental design. It does, however, likely explain ConvNeXt's tendency to predict the majority class, yielding high accuracy but very low recall. The dataset is also limited to 82 patients from a single hospital, which constrains the generalisability of these findings. These factors mean that the observed performance gap may reflect differences in training conditions as much as differences in architectural capability.

- Reason for the change: Reviewers 3 and 4 noted that the effect of class imbalance was not analysed. This paragraph contextualises the results, explaining that the different class distributions are inherent to how each model paradigm processes the data.

---

*Computational Cost (added to Experiments)*

- Before: No computational cost data was reported.

- After: A table summarising computational cost was added to the experiments section:

> | Model | Parameters | Train time/run | Train samples/sec | Eval samples/sec |
> |-------|-----------|---------------|-------------------|------------------|
> | ViVit | 87.6M | ~39 min | 0.49 | 0.71 |
> | ConvNeXt | 27.8M | ~42 min | 8.6 | 12.7 |

- Reason for the change: Reviewer 4 requested analysis of computational cost and latency. The data was extracted from the Weights & Biases experiment tracker.

---

*Acknowledgements*

- Before: No acknowledgements section existed.

- After:

> GitHub Copilot and Claude Code were used to assist in the development of code and the writing of this paper. All contributions were reviewed, verified, and accepted by the authors.

- Reason for the change: Reviewer 5 requested explicit reference to the use of AI in the production of the article, in compliance with IEEE submission and peer review policies.

---

**Additional corrections:**

- Fixed British English spelling: 18 instances of -ize/-ized/-izing changed to -ise/-ised/-ising across 11 files.
- Fixed grammatical errors: "runned" → "ran", "may had" → "may have", "Notably, Meanwhile," → corrected sentence.
- Removed duplicate LaTeX packages (graphicx, xcolor, amsmath) and resolved natbib/cite conflict in configuration.tex.
