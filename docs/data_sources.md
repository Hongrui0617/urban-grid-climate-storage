# Data Sources

Raw datasets are not distributed with this repository. Users should obtain data from official or authorized sources and place prepared CSV files in `data_raw/`.

## Historical Load Data

Expected local file:

```text
data_raw/TEPCO_Load_2016_2022_hourly_with_units.csv
```

Expected source category:

- TEPCO/OCCTO-type hourly electricity demand data for the Tokyo Electric Power Company service area.

Expected columns used by the current workflow:

- `datetime_jst`
- `load_MW`

## Historical Weather Data

Expected local file:

```text
data_raw/Tokyo_TempRH_2016_2022_hourly_wide_clean.csv
```

Expected source category:

- JMA-type hourly meteorological observations for Tokyo.

Expected columns used by the current workflow:

- `datetime`
- `temp_c`
- `rh_percent`

## Future Climate Data

Future thesis stages are expected to use CMIP6 scenario data for SSP1-2.6, SSP2-4.5, and SSP5-8.5. These data are not used by the current codebase yet.

## 中文说明

本仓库不分发原始数据。请从官方或授权来源获取数据，并将整理后的 CSV 文件放入 `data_raw/`。当前工作流需要东京电力负荷数据和东京逐小时气象观测数据；未来阶段将加入 CMIP6 情景数据。
