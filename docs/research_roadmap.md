# Research Roadmap

## Stage 1: Historical Load Forecasting

Status: implemented as the current repository workflow.

- Build an hourly load-weather dataset.
- Train seasonal naive, Random Forest, and LSTM forecasting models.
- Evaluate performance, including peak-load error.

## Stage 2: Climate Scenario Processing

Status: planned.

- Select CMIP6 models and scenarios: SSP1-2.6, SSP2-4.5, and SSP5-8.5.
- Process climate variables relevant to load, PV, and wind generation.
- Apply bias correction and downscaling.

## Stage 3: Supply-Demand Simulation

Status: planned.

- Forecast future hourly demand.
- Estimate PV and wind generation from meteorological drivers.
- Calculate hourly supply-demand mismatch.

## Stage 4: Storage Capacity Estimation

Status: planned.

- Estimate storage capacity under closed-system assumptions.
- Evaluate sensitivity to climate scenarios and reliability targets.
- Discuss limitations from ignoring interregional power exchange.

## 中文摘要

当前仓库对应第一阶段：历史负荷预测。后续研究将扩展到 CMIP6 气候情景处理、可再生能源供给模拟、供需缺口计算，以及可靠性约束下的储能容量估计。
