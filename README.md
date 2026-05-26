# Urban Grid Climate Storage

## English

This repository supports a master's thesis project on the energy storage capacity needed for Tokyo's power grid under future climate scenarios.

The long-term research goal is to connect historical electricity load forecasting, CMIP6 climate scenarios, renewable generation simulation, hourly supply-demand mismatch analysis, and storage capacity estimation under reliability constraints.

The current repository implements the first stage: historical hourly load forecasting for the Tokyo Electric Power Company service area using observed weather conditions.

### Current Stage

Implemented components:

- Hourly TEPCO load and Tokyo weather data preparation.
- Feature engineering with weather, calendar, and lagged load variables.
- Exploratory analysis for historical load and meteorological patterns.
- Baseline forecasting models, including seasonal naive and Random Forest.
- LSTM load forecasting workflow.
- Model evaluation with MAE, RMSE, sMAPE, and Top1pct_MAE.

Future thesis stages will add CMIP6 scenarios, bias correction/downscaling, PV and wind generation modeling, hourly supply-demand mismatch analysis, and storage capacity estimation under reliability metrics such as loss-of-load probability.

### Repository Structure

```text
configs/          Experiment and city configuration files
data_raw/         Local raw data files; not tracked by Git
data_processed/   Generated processed datasets; not tracked by Git
docs/             Research notes and methodology documentation
outputs/          Generated figures, tables, and model files; not tracked by Git
scripts/          Utility scripts
src/              Python source code for the workflow
tests/            Future tests and validation scripts
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation

Raw data are not included because of file size and data licensing considerations.

Place the following local files in `data_raw/`:

```text
data_raw/TEPCO_Load_2016_2022_hourly_with_units.csv
data_raw/Tokyo_TempRH_2016_2022_hourly_wide_clean.csv
```

The expected source categories are TEPCO/OCCTO-type hourly electricity demand data and JMA-type hourly meteorological observations for Tokyo. See `docs/data_sources.md` for details.

### Run the Workflow

Check the configuration:

```bash
python scripts/check_config.py --config configs/tokyo.yaml
```

Run the current historical load forecasting workflow:

```bash
python src/build_dataset.py
python src/explore_dataset.py
python src/train_baselines.py
python src/train_lstm.py
python src/evaluate_models.py
```

Generated datasets and outputs are written to `data_processed/` and `outputs/`. These directories are intentionally ignored by Git.

### License

This project is released under the MIT License.

## 中文

本仓库用于支持硕士论文研究：未来气候情景下东京电网所需储能容量评估。

长期研究目标是建立一条可复现的分析链条：历史电力负荷预测、CMIP6 未来气候情景、可再生能源发电模拟、逐小时供需缺口分析，以及在可靠性约束下的储能容量估计。

当前仓库实现的是第一阶段：基于历史观测气象条件，对东京电力公司服务区域的逐小时电力负荷进行预测。

### 当前阶段

已完成内容包括：

- TEPCO 逐小时负荷数据和东京气象数据整理。
- 温度、相对湿度、日历变量和滞后负荷特征构建。
- 历史负荷与气象变量的探索性分析。
- 季节性朴素模型和 Random Forest 等 baseline 模型。
- LSTM 负荷预测流程。
- 使用 MAE、RMSE、sMAPE 和 Top1pct_MAE 进行模型评估。

后续论文阶段将继续加入 CMIP6 情景、偏差订正与降尺度、光伏和风电出力建模、逐小时供需缺口计算，以及基于失负荷概率等可靠性指标的储能容量估计。

### 仓库结构

```text
configs/          城市和实验配置文件
data_raw/         本地原始数据，不纳入 Git
data_processed/   生成的处理后数据，不纳入 Git
docs/             研究方法和资料说明
outputs/          生成的图表、表格和模型文件，不纳入 Git
scripts/          工具脚本
src/              Python 工作流源代码
tests/            后续测试与验证脚本
```

### 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 数据准备

由于文件大小和数据许可原因，原始数据不包含在本仓库中。

请将以下本地文件放入 `data_raw/`：

```text
data_raw/TEPCO_Load_2016_2022_hourly_with_units.csv
data_raw/Tokyo_TempRH_2016_2022_hourly_wide_clean.csv
```

数据来源类型为 TEPCO/OCCTO 类逐小时电力需求数据，以及 JMA 类东京逐小时气象观测数据。更多说明见 `docs/data_sources.md`。

### 运行流程

检查配置：

```bash
python scripts/check_config.py --config configs/tokyo.yaml
```

运行当前历史负荷预测流程：

```bash
python src/build_dataset.py
python src/explore_dataset.py
python src/train_baselines.py
python src/train_lstm.py
python src/evaluate_models.py
```

生成的数据集和输出会写入 `data_processed/` 和 `outputs/`。这些目录已被 Git 忽略。

### 许可证

本项目使用 MIT License。
