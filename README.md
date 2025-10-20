# Road Infrastructure Climate Risk Assessment Framework (RICRAF)

This repository provides the **Jupyter notebooks, Python code, and processing workflow** used to create the 
fused geospatial dataset of **road infrastructure and climate hazards for Victoria, Australia**, described in 
the *Geoscience Data Journal (GDJ)* Data Article:

> **Chin, T.K. (2025).** *Fused Geospatial Dataset of Road Infrastructure and Climate Hazards for Victoria, Australia.*  
> Zenodo. [https://doi.org/10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)

---

## 🌏 Overview

The RICRAF workflow integrates **road attributes** (traffic volume, pavement condition, and road configuration) 
from **DataVic** with **climate hazard indicators** (e.g. extreme rainfall indices, multi-day precipitation summaries, 
and temperature extremes) from the **Australian Climate Service**. 

Processing and validation are implemented in **Python** using **GeoPandas** and related open-source libraries to 
ensure transparency and reproducibility under **FAIR data principles** — *Findable, Accessible, Interoperable, and Reusable* (Wilkinson *et al.*, 2016).

**Key Outputs**
- Fused geospatial dataset → `data/processed/gdf_road_clim_cln_final_withfuture.geojson`
- Field dictionary → `data/processed/variables.csv`
- Metadata file → `data/processed/metadata.json`

The dataset archive is permanently hosted on **Zenodo** and the codebase is mirrored on **GitHub** for reproducibility.

---

## 🗂️ Repository Structure

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
│   └── ricraf_data_fusion.ipynb
├── outputs/
└── src/
    └── ricraf_data_fusion.py
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

1. **Download raw data**
   - Official data archive: [Zenodo DOI 10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)
     
      *Note: raw datasets sourced from:*
      - *Road data: [DataVic](https://www.data.vic.gov.au/)*
      - *Climate data: [Australian Climate Service](https://www.acs.gov.au/pages/data-explorer)*


2. **Run the Jupyter notebook**
   ```bash
   jupyter notebook notebooks/ricraf_data_fusion.ipynb
   ```

3. **Reproducibility**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧭 Data Description

| Aspect | Details                                                                                                                     |
|--------|-----------------------------------------------------------------------------------------------------------------------------|
| **Spatial Coverage** | Victoria, Australia                                                                                                         |
| **Temporal Coverage** | Road data (2020 baseline); <br/>Climate projections for GWLs: 1.2°C (current), 1.5°C (~2030), 2.0°C (~2050), 3.0°C (~2090). |
| **Coordinate System** | GDA94 / VicGrid projection (EPSG:3111)                                                                                      |
| **Format** | GeoJSON, CSV                                                                                                                |
| **Variables** | See `data/processed/variables.csv`                                                                                          |
| **Access** | [Zenodo DOI: 10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)                                              |

---

## 📖 Citation

If you use this dataset or code, please cite both the **data** and **code** archives:

**Dataset:**
> Chin, T.K. (2025). *Fused Geospatial Dataset of Road Infrastructure and Climate Hazards for Victoria, Australia.*  
> Zenodo. [https://doi.org/10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)

**Code:**
> Chin, T.K. (2025). *RICRAF: Road Infrastructure Climate Risk Assessment Framework (Code Repository).*  
> Zenodo. [https://doi.org/10.5281/zenodo.17391486](https://doi.org/10.5281/zenodo.17391486)

For machine-readable citation metadata, please refer to [`CITATION.cff`](CITATION.cff) in this repository.

---

## 🪪 License

- **Code:** MIT License — see [`LICENSE`](LICENSE)  
- **Dataset:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 🙏 Acknowledgements

**Data sources:**
- [DataVic](https://www.data.vic.gov.au/) — Road infrastructure and traffic datasets
- [Australian Climate Service](https://www.acs.gov.au/pages/data-explorer) — Climate hazard and projection data
- [Zenodo](https://zenodo.org/) — Long-term repository for the fused dataset archive

---

## 🔗 Related Resources

- **Dataset (Zenodo):** [10.5281/zenodo.17379391](https://doi.org/10.5281/zenodo.17379391)
- **Code Archive (Zenodo Snapshot):** [10.5281/zenodo.17391486](https://doi.org/10.5281/zenodo.17391486)
- **GDJ Data Article:** *(link to final publication DOI once available)*
- **GitHub Repository:** [https://github.com/teckkean/ricraf](https://github.com/teckkean/ricraf)

---

### 📚 Reference

Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., Appleton, G., Axton, M., Baak, A., … Mons, B. (2016).  
*The FAIR Guiding Principles for scientific data management and stewardship.*  
**Scientific Data, 3**, 160018. https://doi.org/10.1038/sdata.2016.18

---
