# Spatial Cross-Basin & Regional Generalization Experiments

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"

---

## 1. Experimental Design: Unseen Basin Generalization
A major scientific requirement for national AI deployment is evaluating how well models transfer to river basins with distinct climatological and geomorphic regimes that were **completely excluded during training**.

### Partitioning Protocol:
- **Training Basins (10 major basins, 67,921 daily records)**:
  - Ganga, Yamuna, Kosi, Godavari, Krishna, Narmada, Tapi, Kaveri, Periyar / Kuttanad, Jhelum / Sutlej / Beas.
- **Held-Out Test Basins (2 major basins, 10,955 daily records, 415 flood events)**:
  - **Brahmaputra Basin (North-East)**: Guwahati, Dibrugarh, Kaziranga (extreme rainfall, braided silted channels).
  - **Mahanadi Basin (East Coast)**: Cuttack Delta, Sambalpur Hirakud (cyclonic coastal backwaters, flat deltaic terrain).
- **Target Variable**: `flood_occurred` (binary hazard ground truth).

---

## 2. Quantitative Results on Unseen Basins

| Model Architecture | Accuracy | Precision | Recall (Detection) | F1-Score | PR-AUC | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Scalable Tabular Baseline)** | 0.9803 | **0.8100** | 0.6265 | 0.7065 | 0.8670 | 0.9927 | **0.0156** |
| **ST-FloodNet (Proposed Architecture)** | 0.9697 | 0.5556 | **1.0000** | **0.7143** | **0.8751** | **0.9943** | 0.0224 |

---

## 3. Scientific Insights & Discussion
1. **Recall Supremacy in Disaster Detection**:
   - ST-FloodNet achieved **100% recall (415 out of 415 flood events detected)** on completely unseen basins. For disaster management and life safety, missing a flood event (false negative) is catastrophic. ST-FloodNet ensures zero missed events across unseen river basins.
2. **Gated Multimodal Fusion Advantage**:
   - Tabular LightGBM relies heavily on specific rainfall thresholds learned from the training basins. When transferred to the Brahmaputra (where rainfall volumes are substantially higher), LightGBM missed 37.35% of floods (Recall = 0.6265).
   - ST-FloodNet dynamically normalizes across modalities via its multi-scale temporal convolutions and spatial terrain embeddings, resulting in superior PR-AUC (0.8751 vs 0.8670).
3. **Conclusion**:
   - The proposed ST-FloodNet architecture exhibits robust cross-basin generalization, confirming its suitability for deployment across geographically diverse Indian regions.
