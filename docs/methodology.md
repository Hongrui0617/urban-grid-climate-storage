# Methodology Notes

## Research Aim

This project studies the energy storage capacity required for Tokyo's power grid under future climate scenarios.

The planned thesis workflow links:

1. Historical electricity load and meteorological observations.
2. Climate-driven load forecasting.
3. CMIP6 scenario processing for SSP1-2.6, SSP2-4.5, and SSP5-8.5.
4. PV and wind generation simulation.
5. Hourly supply-demand mismatch analysis.
6. Storage capacity estimation under reliability constraints.

## Current Implementation

The current codebase implements the historical load forecasting stage. It combines hourly electricity load with Tokyo temperature and relative humidity observations, adds calendar and lag features, trains baseline models and an LSTM model, and evaluates prediction error with MAE, RMSE, sMAPE, and Top1pct_MAE.

This stage is a foundation for later scenario-based demand projection. It does not yet include CMIP6 data, renewable generation modeling, or storage optimization.

## Planned Extensions

- Add CMIP6 download and preprocessing notes.
- Implement bias correction and downscaling for future weather variables.
- Simulate PV and wind generation from meteorological drivers.
- Calculate hourly supply-demand mismatch under closed-system assumptions.
- Estimate storage capacity for target reliability levels.

## 中文摘要

本项目的论文目标是评估未来气候情景下东京电网所需的储能容量。当前代码实现的是第一阶段：历史观测气象条件下的逐小时负荷预测。后续将加入 CMIP6 情景、可再生能源出力模拟、供需缺口分析和储能容量优化。
