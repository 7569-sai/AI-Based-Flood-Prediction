# 10-Tier Progressive Architectural & Modality Ablation Study

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Standard:** Rigorous Scientific Component Isolation & Contribution Analysis  

---

## 1. Objective & Methodological Design

In complex multimodal machine learning architectures, individual components (attention layers, recurrent memories, multimodal encoders, and stacking meta-learners) can introduce parameter complexity without necessarily improving out-of-distribution generalizability.

To verify that every architectural module in the FloodAI Engine contributes positively and measurably to predictive skill, a **10-Tier Progressive Ablation Protocol** is executed on the untouched 2023 out-of-time test partition.

```
                           10-TIER ABLATION HIERARCHY
                           
 Tier 1: Full HF-STFloodNet [Benchmark Architecture]
   ├── [Architecture Layer Ablations]
   │   ├── Tier 2: Remove Stacking Meta-Learner (50/50 Probability Blend)
   │   ├── Tier 3: Remove LightGBM Tabular Branch (ST-FloodNet Standalone)
   │   ├── Tier 4: Remove Deep Latent Embeddings (Probabilities-Only Stacking)
   │   ├── Tier 5: Remove Temporal Attention (Uniform Average Pooling)
   │   └── Tier 6: Remove Bi-GRU Memory (Convolutional Antecedents Only)
   │
   └── [Sensor & Modality Feature Ablations]
       ├── Tier 7: Remove Sentinel-1 SAR Radar (Zeroed VV, VH, Cross-ratio)
       ├── Tier 8: Remove Sentinel-2 Optical Remote Sensing (Zeroed NDWI, NDVI)
       ├── Tier 9: Remove Topographic Geomorphology (Zeroed DEM, Slope, TWI, River Dist)
       └── Tier 10: Remove Socioeconomic Exposure & Vulnerability (Zeroed Demographics)
```

---

## 2. Description of Experimental Tiers

### Tier 1: Full Proposed HF-STFloodNet (Reference Model)
- Complete pipeline incorporating:
  - 27 input features routed into 4 specialized physical encoders
  - Multi-scale 1D Conv $\to$ Bidirectional GRU $\to$ Temporal Self-Attention
  - Gated cross-modal fusion unit
  - 64-dimensional bottleneck latent representation
  - LightGBM gradient boosted decision tree tabular branch
  - Out-of-fold leakage-safe stacking meta-learner
  - Validation-locked Platt scaling calibration and optimal decision threshold $\theta^*$

### Tier 2: Without Meta-Learner Stacking
- **Modification:** Replaces the LightGBM meta-learner with an unweighted arithmetic mean of base probabilities: $\hat{p} = 0.5 \hat{p}_{\text{ST}} + 0.5 \hat{p}_{\text{LGB}}$.
- **Hypothesis:** Measures the marginal value of learned non-linear decision boundary fusion over naïve ensemble averaging.

### Tier 3: Without LightGBM Tabular Branch
- **Modification:** Discards the gradient boosting branch entirely, relying solely on the deep neural network ST-FloodNet.
- **Hypothesis:** Quantifies the value of structured tabular tree splitting for handling sharp physical feature thresholds (e.g. river stage exceeding 2.0m).

### Tier 4: Without Deep Bottleneck Latent Embeddings
- **Modification:** Fits the meta-learner solely on base model probability outputs $(\hat{p}_{\text{ST}}, \hat{p}_{\text{LGB}})$ and their interactions, stripping out the 8 PCA components of the 64-dim cross-modal representation.
- **Hypothesis:** Evaluates whether deep latent feature representations provide richer inductive signal to the meta-learner than scalar probabilities alone.

### Tier 5: Without Temporal Attention
- **Modification:** Replaces the dynamic self-attention mechanism with uniform average pooling over temporal GRU hidden states: $\mathbf{h}_{\text{temp}} = \frac{1}{T} \sum_{t=1}^T \mathbf{h}_t$.
- **Hypothesis:** Tests whether dynamically attending to peak precipitation bursts outperforms uniform antecedent accumulation.

### Tier 6: Without Bidirectional GRU Memory
- **Modification:** Eliminates recurrent recurrence, flattening the multi-scale 1D convolutional outputs directly into the cross-modal fusion unit.
- **Hypothesis:** Quantifies the importance of sequential hydrological memory and hydrograph shape tracking.

### Tier 7: Without Sentinel-1 SAR Radar Modality
- **Modification:** Sets radar backscatter features (`sar_vv`, `sar_vh`, `sar_cross_ratio`) to zero during inference.
- **Hypothesis:** Isolates the contribution of all-weather cloud-penetrating synthetic aperture radar in confirming active surface water inundation.

### Tier 8: Without Sentinel-2 Optical Remote Sensing
- **Modification:** Sets optical spectral indices (`ndwi_proxy`, `ndvi_proxy`) to zero.
- **Hypothesis:** Evaluates the diagnostic role of multispectral water and vegetation vigor indicators.

### Tier 9: Without Topographic Geomorphology
- **Modification:** Sets DEM elevation, terrain slope, Euclidean river distance, topographic wetness index (TWI), and drainage vulnerability to zero.
- **Hypothesis:** Measures how much predictive skill is lost when the model is blind to physical catchment lowlands and channel proximity.

### Tier 10: Without Socioeconomic Exposure & Vulnerability
- **Modification:** Sets population density, exposure index, and vulnerability index to zero.
- **Hypothesis:** Quantifies the impact of demographic vulnerability priors on disaster response risk classification.

---

## 3. Results Summary & Physical Takeaways

The exact numerical results across all 10 tiers are stored in `outputs/metrics/ablation_results.csv` and illustrated in `outputs/figures/10_tier_ablation_waterfall.png`.

**Key Physical Insights:**
1. **The Stacking Synergy:** Combining ST-FloodNet with LightGBM via meta-learning yields higher F1 and ROC-AUC than either base model alone, as LightGBM resolves localized sharp feature cutoffs while ST-FloodNet extracts rich spatiotemporal patterns.
2. **Terrain & Radar Criticality:** Zeroing out topographic geomorphology (Tier 9) or SAR radar (Tier 7) produces significant performance drops, proving that hydrometeorological forcing alone is insufficient without geomorphic depression and observational inundation evidence.
3. **Temporal Attention Advantage:** Replacing temporal attention with average pooling (Tier 5) causes a measurable recall penalty during sudden flash flood events where rain fell predominantly on a single burst day.
