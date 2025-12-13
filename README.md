# 🛣️ Road Infrastructure Climate Risk Assessment Framework (RICRAF) 

**RICRAF** — the *Road Infrastructure Climate Risk Assessment Framework* — is an open-source, reproducible workflow for assessing climate-related risks to road networks under current and future global warming levels. 
It integrates geospatial data fusion, machine learning, and explainable AI to support data-driven resilience planning.

RICRAF comprises **three research and code stages**, each linked to a corresponding publication:

| Stage | Focus | Journal | Status |
|--------|--------|----------|--------|
| **1. Data Fusion** | Creation of the fused geospatial dataset linking road and climate hazard data | *Scientific Data* | Under review |
| **2. Model Development** | Development of the XGBoost–SHAP framework and Climate Risk Formula | *Climate Services* | Under review |
| **3. Model Application** | Application of the framework under multiple Global Warming Levels (GWLs) and traffic scenarios | *Transportation Research Part D: Transport and Environment* | Under review |

---

## 🌏 Overview

RICRAF operationalises the **IPCC AR5/AR6 Hazard–Exposure–Vulnerability** concept through a transparent, reproducible, and transferable Python workflow.  
The framework:

- Links **road infrastructure** (traffic, road surface, geometry) with **climate hazards** (heat, rainfall, drought, frost).
- Develops a machine learning-based **Climate Risk Formula (CRF)** using SHAP-derived weights.
- Produces **Climate Risk Scores (CRiskS)** at road link scale to support adaptation and resilience investment planning.
- Adheres to **FAIR principles** — *Findable, Accessible, Interoperable, Reusable* — through open data, open code, and clear metadata.

---

## 🔄 Workflow

### Stage 1 — Data Fusion

The data fusion workflow integrates:
- Road datasets from [DataVic](https://www.data.vic.gov.au/) — including traffic volume, road surface condition, and road geometry.
- Climate hazard indices from the [Australian Climate Service](https://www.acs.gov.au/pages/data-explorer) — including precipitation, heat, frost, and drought indicators derived from CMIP6 downscaled models (BARPA & CCAM).

Processing steps include Coordinate Reference System (CRS) standardisation, schema harmonisation, topology checks, spatial joins, and quality validation.
The resulting dataset links 7,579 road segments with climate hazard metrics across **Global Warming Levels (GWLs of 1.2°C, 1.5°C, 2.0°C, 3.0°C)**.

**Outputs:**
- `data/processed/gdf_road_clim_cln_final_withfuture.geojson`
- `data/processed/variables.csv`
- `data/processed/metadata.json`

📘 **Reference:**
> Chin, T.K., Prakash, M., Zheng, N., & Pauwels, V.R.N. (2025). *Fused Geospatial Dataset Linking Climate Hazards and Road Infrastructure for Victoria, Australia.*  
> *Scientific Data* (under review). 

---

### Stage 2 — Model Development

This stage develops the **XGBoost–SHAP framework** for multi-stressor climate risk assessment, integrating:
- **Extreme Gradient Boosting (XGBoost)** for predictive modelling of road surface distress (roughness, rutting, cracking).
- **SHapley Additive exPlanations (SHAP)** for interpretable feature attribution and driver importance quantification.
- A **Climate Risk Formula (CRF)** that computes *Climate Risk Scores (CRiskS)* using SHAP-derived weights within the IPCC risk structure.

**Key Contributions:**
- Unified modelling of climate, traffic, and vulnerability stressors.
- SHAP-derived weighting of hazard, exposure, and vulnerability components.
- Network-wide and link-level interpretability for actionable risk mapping.
- Python implementation consistent with national climate services and adaptation frameworks.

**Outputs:**
- Trained ML models for each distress type.
- SHAP-based driver contribution plots.
- CRiskS maps for current warming (GWL of 1.2°C).

📘 **Reference:**
> Chin, T.K., Prakash, M., Zheng, N., & Pauwels, V.R.N. (2025). *Explainable AI for Multi-Stressor Climate Risk Assessment of Road Networks: An XGBoost–SHAP Framework.*  
> *Climate Services* (under review).

---
### Stage 3 — Model Application

This stage applies the multi-stressor climate risk framework to quantify spatial risk patterns across 23,117 km of Victoria's road network under current climate conditions and projected warming levels up to 3.0°C. It attributes total risk to hazard, exposure, and vulnerability using interpretable Aumann–Shapley decomposition methods, assesses robustness through sensitivity analysis, and identifies emerging network hotspots where interacting stressors amplify climate risk.

**Key Contributions:**
- Quantification of risk evolution: Currently, 6.4% of the network exhibits high or extreme risk, driven primarily by vulnerability and hazard. Under 3.0°C warming, high/extreme risk expands to 24.4%, with hazard’s contribution increasing.
- Inclusion of traffic growth scenarios, showing risk exceeding 55% in metropolitan corridors.
- Sensitivity tests confirming stable results under ±20% parameter variation.
- Demonstration of framework scalability and transferability for transport resilience planning through modular, open-source design.

**Outputs:**
- Climate Risk Scores GeoJSON for multiple scenarios and warming levels.
- Confusion matrices, statistical evaluations, and sensitivity analyses.
- Static maps (PNG) and interactive HTML maps (Kepler.gl) for risk visualisation.

📘 **Reference:**
> Chin, T.K., Prakash, M., Zheng, N., & Pauwels, V.R.N. (2025). *Climate Risk Assessment of Road Infrastructure under Multi-Stressor Conditions in Victoria, Australia.*  
> *Transportation Research Part D: Transport and Environment* (under review).

---

## 🧰 Repository Structure

```
ricraf/
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── ricraf_data_fusion.ipynb
│   ├── ricraf_development.ipynb
│   └── ricraf_application.ipynb
├── outputs/
└── src/
    ├── ricraf_data_fusion.py
    └── ricraf_dev.py
```

---


## ⚙️ Installation

```bash
git clone https://github.com/teckkean/ricraf.git
cd ricraf
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Data Fusion

Download the raw data from Zenodo ([DOI 10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)) and execute the Jupyter notebook:
```bash
jupyter notebook notebooks/ricraf_data_fusion.ipynb
```

Raw data sources: 
- *Road data: [DataVic](https://www.data.vic.gov.au/)*
- *Climate data: [Australian Climate Service](https://www.acs.gov.au/pages/data-explorer)*


### 2. Model Development

Run the model training and explainability workflow:
```bash
jupyter notebook notebooks/ricraf_development.ipynb
```
This notebook reproduces the XGBoost–SHAP pipeline, generates driver attribution plots, and computes Climate Risk Scores (CRiskS).

Outputs are saved to:
```swift
data/processed/model_dev/out_crs/
```

### 3. Model Application

Run the application workflow to compute risk scores, generate confusion matrices, and produce maps:
```bash
jupyter notebook notebooks/ricraf_application.ipynb
```
This notebook applies the framework to current and future scenarios, performs statistical evaluations, and saves outputs to
```swift
data/processed/model_app/out_cra/
```

---

## 🗂️ Data Description 

| Aspect                | Details                                                                                                                    |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Spatial Coverage**  | Victoria, Australia                                                                                                        |
| **Temporal Coverage** | Road data (2020 baseline); <br/>Climate projections for GWLs 1.2°C (current), 1.5°C (~2030), 2.0°C (~2050), 3.0°C (~2090). |
| **Coordinate Reference System (CRS)** | GDA94 / VicGrid projection (EPSG:3111)                                                                                     |
| **Format**            | GeoJSON, CSV                                                                                                               |
| **Access**            | [Zenodo DOI: 10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)                                             |
| **License**           | CC BY 4.0 (data), MIT (code)                                                                                               |


---

## 📖 Citation

If you use this dataset, code, or methodology, please cite:

**Data Descriptor Paper:**
> Chin, T.K., Prakash, M., Zheng, N., & Pauwels, V.R.N. (2025). Fused Geospatial Dataset Linking Climate Hazards and Road Infrastructure for Victoria, Australia.  
> *Scientific Data* (under review).

**Dataset (Data Fusion Stage):**
> Chin, T.K. (2025). *Fused Geospatial Dataset of Road Infrastructure and Climate Hazards for Victoria, Australia.*  
> Zenodo. [https://doi.org/10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)

**Code Repository:**
> Chin, T.K. (2025). *RICRAF: Road Infrastructure Climate Risk Assessment Framework (Code Repository).*  
> Zenodo. [https://doi.org/10.5281/zenodo.17391486](https://doi.org/10.5281/zenodo.17391486)

**Model Development Paper:**
> Chin, T.K., Prakash, M., Zheng, N., & Pauwels, V.R.N. (2025). Explainable AI for Multi-Stressor Climate Risk Assessment of Road Networks: An XGBoost–SHAP Framework. \
> *Climate Services* (under review).

**Model Application Paper:**
> Chin, T.K., Prakash, M., Zheng, N., & Pauwels, V.R.N. (2025). Climate Risk Assessment of Road Infrastructure under Multi-Stressor Conditions in Victoria, Australia. \
> *Transportation Research Part D: Transport and Environment* (under review).

For machine-readable citation metadata, please refer to [`CITATION.cff`](CITATION.cff) in this repository.


---

## 🪪 License

- **Code:** MIT License — see [`LICENSE`](LICENSE)  
- **Dataset:** Creative Commons Attribution 4.0 International (CC BY 4.0)

> 🧩 This repository complies with the FAIR data principles and the open-science standards recommended by
> *Scientific Data*, *Climate Services*, and *Transportation Research Part D*.


---

## 🙏 Acknowledgements

**Data sources:**
- [DataVic](https://www.data.vic.gov.au/) — for road infrastructure and traffic datasets
- [Australian Climate Service](https://www.acs.gov.au/pages/data-explorer) — for climate hazard and projection data
- [Zenodo](https://zenodo.org/) — for open data hosting
- Supported by [Monash University](https://www.monash.edu/) and [CSIRO Data 61](https://research.csiro.au/data61/)

---

## 🔗 Related Resources

- **Dataset (Zenodo):** [10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)
- **Code Snapshot (Zenodo):** [10.5281/zenodo.17391486](https://doi.org/10.5281/zenodo.17391486)
- **GitHub Repository:** [https://github.com/teckkean/ricraf](https://github.com/teckkean/ricraf)
- **Data Article:** *(link to final publication DOI once available)*
- **Model Development Paper:** *(link to final publication DOI once available)*
- **Model Application Paper:** *(link to final publication DOI once available)*

---

### 📚 Reference

Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., Appleton, G., Axton, M., Baak, A., … Mons, B. (2016).  
*The FAIR Guiding Principles for scientific data management and stewardship.*  
**Scientific Data**, 3, 160018. https://doi.org/10.1038/sdata.2016.18


---
© 2025 Teck Kean Chin   
