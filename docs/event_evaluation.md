# Event-Level Continuous Flood Evaluation Methodology

**Project Title:** AI-Based Flood Prediction, Dynamic Risk Mapping and Resilience Planning  
**Application:** FloodAI Engine — "AI for a Safer, Resilient India"  
**Standard:** Continuous Disaster Lifecycle & Operational Incident Metrics  

---

## 1. Motivation: Beyond Pointwise Daily Metrics

Traditional machine learning benchmarks evaluate flood prediction on a pointwise, day-by-day classification basis ($y_t$ vs $\hat{y}_t$). While sample-level metrics (such as Precision, Recall, $F_1$, and ROC-AUC) quantify general statistical discriminability, they fail to capture the operational reality faced by disaster management authorities (NDRF, SDMAs, and District Magistrates).

In emergency civil protection:
1. **Disaster events are contiguous episodes:** Floods unfold over several days or weeks as atmospheric depression systems move across catchments, saturating soil profiles and cascading down river channels.
2. **Timing is paramount:** An alert issued on Day 3 of an ongoing flood after critical embankments have already breached provides zero operational value, even though it counts as a statistical "True Positive" in sample-level metrics.
3. **Advance notice enables preventative action:** Civil defense operations require sufficient lead time prior to the crest / peak inundation stage to stage rescue boats, open relief shelters, and coordinate reservoir emergency discharges.
4. **False Alarm Fatigue:** Persistent false alarms erode public trust and trigger costly, unnecessary mass evacuations.

To address these imperatives, the FloodAI Engine evaluates all models across continuous **Event-Level Metrics**.

---

## 2. Mathematical Definition of Event-Level Metrics

### 2.1 Event Segmentation
Let $\mathbf{y}^{(s)} = [y_1^{(s)}, y_2^{(s)}, \dots, y_N^{(s)}]$ be the binary ground truth time series for station $s$, and $\hat{\mathbf{y}}^{(s)} = [\hat{y}_1^{(s)}, \dots, \hat{y}_N^{(s)}]$ be the predicted warnings at decision threshold $\theta^*$.

A continuous **True Flood Event** $\mathcal{E}_k = [t_{\text{start}}^{(k)}, t_{\text{end}}^{(k)}]$ is defined as a maximal contiguous subsequence where $y_t^{(s)} = 1$, allowing a 1-day temporal tolerance for brief sensor or precipitation pauses.

Similarly, a continuous **Predicted Warning Episode** $\mathcal{P}_j = [t_{\text{start}}^{(j)}, t_{\text{end}}^{(j)}]$ is defined as a contiguous subsequence where $\hat{y}_t^{(s)} = 1$.

### 2.2 Event Recall (ER)
A true flood event $\mathcal{E}_k$ is classified as *detected* if there exists at least one predicted warning episode $\mathcal{P}_j$ overlapping its temporal lifespan:
$$\text{Overlap}(\mathcal{E}_k, \mathcal{P}_j) = \max(t_{\text{start}}^{(k)}, t_{\text{start}}^{(j)}) \le \min(t_{\text{end}}^{(k)}, t_{\text{end}}^{(j)})$$

$$\text{Event Recall (ER)} = \frac{\sum_{k=1}^{K} \mathbf{1}_{\{\exists j : \text{Overlap}(\mathcal{E}_k, \mathcal{P}_j)\}}}{K}$$
where $K$ is the total count of ground truth flood events across all monitored stations.

### 2.3 Event Detection Delay ($\Delta t_{\text{delay}}$)
For each detected event $\mathcal{E}_k$, let $t_{\text{first\_alert}}^{(k)} = \min \{ t : t \in \mathcal{P}_j \text{ and } t \ge t_{\text{start}}^{(k)} \}$.  
The detection delay represents how many days elapsed between the physical onset of inundation and the first system trigger:
$$\Delta t_{\text{delay}}^{(k)} = \max\left(0, t_{\text{first\_alert}}^{(k)} - t_{\text{start}}^{(k)}\right)$$
$$\text{Mean Detection Delay} = \frac{1}{K_{\text{detected}}} \sum_{k=1}^{K_{\text{detected}}} \Delta t_{\text{delay}}^{(k)} \quad (\text{days})$$
*Optimal Target:* $\le 0.5$ days (instantaneous detection on onset day).

### 2.4 Peak Severity Forecast Lead Time ($\tau_{\text{lead}}$)
Every flood event exhibits a day of maximum hydraulic extremity (peak river stage, maximum inundation depth, or peak severity class $s_{\text{max}} = \max_{t \in \mathcal{E}_k} s_t$):
$$t_{\text{peak}}^{(k)} = \arg\max_{t \in \mathcal{E}_k} s_t$$

The operational lead time provided to emergency responders before catastrophic cresting is:
$$\tau_{\text{lead}}^{(k)} = \max\left(0, t_{\text{peak}}^{(k)} - t_{\text{first\_alert}}^{(k)}\right)$$
$$\text{Mean Forecast Lead Time} = \frac{1}{K_{\text{detected}}} \sum_{k=1}^{K_{\text{detected}}} \tau_{\text{lead}}^{(k)} \quad (\text{days})$$
*Optimal Target:* $\ge 2.0$ days advance notice.

### 2.5 Event False Alarm Rate (FAR)
A predicted alert episode $\mathcal{P}_j$ is classified as a *false alarm* if it possesses zero temporal overlap with any ground truth flood event:
$$\text{Event FAR} = \frac{\sum_{j=1}^{J} \mathbf{1}_{\{\forall k : \neg \text{Overlap}(\mathcal{E}_k, \mathcal{P}_j)\}}}{J}$$
where $J$ is the total count of predicted warning clusters.
*Optimal Target:* $\le 0.20$ (minimizing false panic).

---

## 3. Operational Standard Alignment

| Metric | Scientific Utility | Operational Civil Defense Impact |
| :--- | :--- | :--- |
| **Event Recall** | Assesses macroscopic disaster capture rate | Ensures no catastrophic deluge goes unflagged across the district. |
| **Detection Delay** | Quantifies temporal responsiveness of temporal Bi-GRU & SAR radar | Determines whether NDRF search-and-rescue teams are mobilised prior to structural inundation. |
| **Lead Time** | Measures predictive antecedent horizon from cumulative rainfall & API | Provides the crucial window for pre-depleting upstream dam reservoirs and staging food/medical supplies. |
| **Event FAR** | Quantifies credibility and noise rejection | Prevents warning desensitization among coastal and floodplain residents. |

Empirical measurements across all benchmark models are logged in `outputs/metrics/event_level_metrics.csv` and visualized in `outputs/figures/event_detection_latency.png`.
