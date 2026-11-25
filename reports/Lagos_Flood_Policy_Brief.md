# EXECUTIVE POLICY BRIEF: NEIGHBOURHOOD-LEVEL FLOOD RISK & MULTI-SECTOR EXPOSURE IN LAGOS STATE
**Prepared for**: Lagos State Ministry of the Environment and Water Resources, LASEMA, LASBCA, and Infrastructure Development Partners  
**Lead Author**: Geospatial Data Science & Risk Analytics Initiative  
**Date**: November 2026 | **Resolution**: 100-Meter Spatial Susceptibility Grid  

---

## 1. Executive Summary
Urban flooding in Lagos State causes recurring economic destruction, displaced populations, and paralysis of critical logistics arteries. Traditional regional outlooks lack the spatial granularity required for street-level interventions. This brief summarizes findings from an open-source, satellite-driven machine learning framework mapping flood susceptibility and multi-sector exposure across all Local Government Areas (LGAs).

### Key Empirical Findings:
- **Disproportionate Population Exposure**: Over **117,000 residents** live directly inside high-risk flood zones across the top 5 most vulnerable LGAs: **Amuwo-Odofin (27,792), Ikeja (23,587), Alimosho (23,442), Oshodi/Isolo (21,852), and Eti-Osa (21,346)**.
- **Critical Arterial Paralysis**: Over **8.8 kilometers** of primary road corridors are at high risk of seasonal submergence, led by **Apapa Port Corridor (2.66 km)**, **Mushin (2.10 km)**, and **Eti-Osa Lekki-Epe spurs (0.93 km)**, severing port freight logistics and commuter transit.
- **Healthcare Accessibility at Risk**: Key frontline clinics—including the General Hospital corridor in Eti-Osa—are located within flash flood accumulation corridors lacking elevated drainage access.

---

## 2. Priority Risk Hotspots & Drivers

| Local Government Area (LGA) | Total Population | Exposed Population | % Pop. Exposed | High-Risk Road KM | Primary Risk Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Apapa** | 176,497 | 14,946 | **8.47%** | **2.66 km** | Low elevation ($<3\text{m}$), lagoon storm surge, inadequate port drainage |
| **Amuwo-Odofin** | 612,288 | 27,792 | **4.54%** | $0.00\text{ km}$ | High Topographic Wetness Index (TWI), canal backflow |
| **Eti-Osa (Lekki / VI)** | 413,934 | 21,346 | **5.16%** | **0.93 km** | Rapid loss of natural wetlands to concrete/impervious surfaces |
| **Mushin** | 635,908 | 17,959 | **2.82%** | **2.10 km** | Dense urban runoff, primary arterial transit choke points |
| **Ikeja** | 739,425 | 23,587 | **3.19%** | **0.74 km** | Surface pluvial flooding along major transit spurs |

---

## 3. Targeted Policy Recommendations

### Recommendation 1: Prioritize Strategic Canal Desilting in Apapa & Oworonshoki Basin
State maintenance funds should prioritize preemptive desilting in Apapa port canals and the Kosofe/Oworonshoki drainage basin starting in Q1 (February) before the May–July peak deluge. Field-clearing contracts should be linked directly to neighborhood-level TWI hotspot maps.

### Recommendation 2: Mandate On-Site Stormwater Infiltration in Lekki & Ibeju-Lekki
Rapid real estate expansion along the Lekki-Epe corridor has replaced natural retention wetlands with impermeable concrete. The Lagos State Physical Planning Permit Authority (LASPPPA) should mandate permeable pavements and mandatory on-site retention cisterns for all new developments exceeding 500 square meters.

### Recommendation 3: Construct Elevated "Blue-Green" Healthcare Access Spurs
Emergency response fails when access roads to frontline clinics are submerged. State works should prioritize raised culverts and elevated pedestrian walkways for healthcare facilities situated in low-lying corridors (e.g., Eti-Osa and Ajeromi-Ifelodun).

---

## 4. Methodology & Data Sources
- **Terrain**: Copernicus GLO-30 Digital Elevation Model (derived Slope and Topographic Wetness Index at 100m).
- **Satellite Earth Observation**: Sentinel-1 Synthetic Aperture Radar (SAR) backscatter change detection for historical flood extents.
- **Exposure Datasets**: WorldPop 100m constrained population raster and OpenStreetMap arterial road networks and healthcare facilities.
- **Machine Learning**: LightGBM Classifier validated through **5-Fold Spatial Block Cross-Validation (GroupKFold by LGA)** to guarantee zero spatial autocorrelation leakage across district boundaries (**0.981 ROC-AUC, 91.5% Recall**).
