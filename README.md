# Urban Grid Climate Storage

This repository contains a research workflow for climate-driven electricity load forecasting and future energy storage capacity assessment for urban power systems.

The current case study focuses on the Tokyo Electric Power Company service area.

## Research Objective

The project aims to estimate future electricity demand, renewable supply-demand mismatch, and energy storage requirements under climate change scenarios.

The long-term goal is to connect historical load forecasting, future climate scenarios, renewable generation simulation, supply-demand mismatch analysis, and storage capacity estimation into one reproducible workflow.

## Current Progress

The current stage focuses on historical load forecasting under observed meteorological conditions.

Completed components include:

- Historical TEPCO hourly load data processing.
- Tokyo hourly temperature and relative humidity data processing.
- Exploratory analysis of electricity load patterns.
- Weekday and weekend load profile comparison.
- Monthly and hourly load pattern analysis.
- Baseline forecasting model implementation.
- LSTM load forecasting model implementation.
- Model evaluation using MAE, RMSE, sMAPE, and Top1pct_MAE.

## Repository Structure

```text
configs/        Configuration files for different cities and experiments
scripts/        Executable workflow scripts
src/            Reusable Python source code
data/           Local data directory, not tracked by Git
outputs/        Generated figures, metrics, predictions, and models
docs/           Methodology notes and project documentation
tests/          Unit tests

How to Run

Install dependencies:
pip install -r requirements.txt

Run the full workflow:
python scripts/run_all.py --config configs/tokyo.yaml

Data

Raw and processed data are not included in this repository due to file size and data licensing considerations.

Expected local data structure:

data/
├── raw/
│   └── tokyo/
│       ├── TEPCO_Load_2016_2022_hourly_with_units.csv
│       └── Tokyo_TempRH_2016_2022_hourly_wide_clean.csv
├── interim/
└── processed/

Evaluation Metrics

The current model evaluation uses the following metrics:

MAE: Mean Absolute Error.
RMSE: Root Mean Squared Error.
sMAPE: Symmetric Mean Absolute Percentage Error.
Top1pct_MAE: Mean absolute error during the highest 1% load periods.

Top1pct_MAE is especially important for this research because peak load periods are closely related to power system stress and future storage capacity requirements.

Future Work

Planned next steps:

Refactor the current Tokyo workflow into a configuration-driven pipeline.
Add CMIP6 future climate scenario inputs.
Implement bias correction and downscaling workflow.
Predict future hourly electricity demand under SSP scenarios.
Implement PV and wind power generation models.
Calculate hourly supply-demand mismatch.
Estimate energy storage capacity requirements under reliability constraints.
Research Context

This project supports a master's thesis on energy storage capacity assessment for Tokyo's power grid under future climate scenarios.