# Green-mirage
Code for The Green Mirage of Carbon Uptake in a Drying World. Four machine learning models were used to estimate monthly GPP using climate and remote sensing data.

This repository contains the Python code used to reconstruct global gross primary productivity (GPP) from 1982 to 2022 using four machine learning algorithms: Random Forest (RF), Extra Trees (ET), Extreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine (LightGBM).

The workflow integrates FLUXNET-based GPP observations with climatic, soil moisture, atmospheric CO2 and satellite-derived vegetation data. Model predictors include potential evapotranspiration, precipitation, air temperature, vapour pressure, vapour pressure deficit, soil water, surface solar radiation, NDVI and atmospheric CO2 concentration. Missing NDVI values are handled using a missingness indicator and zero-filled NDVI values.

The dataset is randomly divided into training (70%) and testing (30%) subsets, with model performance evaluated using the coefficient of determination (R2) and root mean square error (RMSE). The trained models are subsequently applied to monthly global environmental raster datasets to reconstruct GPP at 0.5° spatial resolution. Predictions are restricted to vegetated lands using a land-cover mask, and monthly GPP estimates are aggregated to annual totals.

The code includes data preprocessing, model training, performance evaluation, spatial prediction and annual GPP aggregation.
