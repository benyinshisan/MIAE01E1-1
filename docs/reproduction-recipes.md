# PowerShell 復跑指引

整理：陈博文（2260031814）｜2026-11-22｜Issue #29

以下命令在倉庫根目錄執行，每個程式區塊為一條命令。環境建立後，依序完成冒煙與產物檢查，再按實驗安排啟動完整訓練。

## 一、安裝與環境確認

由 `environment.yml` 建立環境；已有同名環境時可直接啟用。

```powershell
conda env create -f environment.yml
```

```powershell
conda activate miae01-dqn
```

CPU 復跑採用項目固定的 PyTorch 版本：

```powershell
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

```powershell
python -m pip install -e .
```

```powershell
python -m pip install -r requirements-test.txt
```

```powershell
python -c "import sys,torch,numpy,gymnasium,miae01_dqn; print(sys.version); print('torch',torch.__version__,'numpy',numpy.__version__,'gymnasium',gymnasium.__version__); print(miae01_dqn.__file__)"
```

項目版本包括 Python 3.12、torch 2.4.1、NumPy 2.5.4、Gymnasium 1.0.0、ALE 0.10.1；依賴清單以倉庫固定檔案為準。套件路徑應指向當前倉庫的 `miae01_dqn`。

使用已配置的 CUDA 環境時，先檢查實際張量運算，再選 `--device cuda`：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name()); print((torch.ones(1,device='cuda')+1).cpu())"
```

此檢查同時執行裝置查詢與 CUDA 核心運算。硬體適配問題可循上述 CPU 路徑完成復跑。

## 二、5k 冒煙與評測設定

預設冒煙：5,000 決策步、250 預熱步、5,000 回放容量、2 局評測。

```powershell
python -m miae01_dqn.runner --smoke --seed 1 --device cpu
```

三局評測範例沿用相同訓練檔位，每局上限 18,000 原始幀：

```powershell
python -m miae01_dqn.runner --smoke --config configs/smoke-evaluation.yaml --seed 1 --device cpu
```

現有 CLI 的局數開關為 `--episodes`；單局幀數上限透過 YAML 的 `eval.max_episode_frames` 設定。命令列明確指定的局數優先於 YAML，例如：

```powershell
python -m miae01_dqn.runner --smoke --config configs/smoke-evaluation.yaml --episodes 2 --seed 1 --device cpu
```

`--smoke` 固定訓練步數與預熱檔位。完整實驗使用下一節命令；三局範例用於冒煙驗收。

## 三、完整實驗入口

依團隊種子安排調整 `--seed`。本命令使用預設 30 局評測、ε=0.05，每 1M 名義訓練幀評測一次；checkpoint 隨 `--save-checkpoints` 保存。

```powershell
python -m miae01_dqn.runner --seed 1 --device cuda --replay-capacity 200000 --total-agent-steps 2500000 --save-checkpoints
```

CPU 執行時將裝置參數改為 `--device cpu`。本機首種子完整 GPU 紀錄為 11,688.96 秒，含定期評測；詳細數據見 [驗收紀錄](validation.md)。

## 四、輸出與檢查

預設目錄為 `runs/ALE-Pong-v5__seed1__cap5000__<UTC時間戳>/`（冒煙）；完整實驗名稱中的容量為 `cap200000`。`--log-dir` 可指定輸出父目錄；自訂 `--run-name` 時每次採用新名稱。

| 檔案 | 用途 |
|---|---|
| `config.json`、`environment.json` | 最終設定與執行環境 |
| `result.json` | `status=complete`、實際步數、更新次數及最終評測 |
| `evaluations.json` | 各輪逐局得分、均值與總體標準差 |
| `metrics.jsonl`、`events.out.tfevents.*` | 訓練指標與 TensorBoard 事件 |
| `curves.png`、`progress.json` | 回報／損失曲線與最近進度 |
| `agent.pt` | 指定保存時產生的本機 checkpoint，Git 已排除 |

列出本次產生的目錄後，將下一條命令中的目錄名稱替換為實際名稱：

```powershell
Get-ChildItem runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name
```

```powershell
Get-Content -Raw 'runs/<實際目錄名稱>/result.json' | ConvertFrom-Json | Select-Object status,train,evaluation
```

三局冒煙應完成 5,000 步、1,187 次更新，且 `evaluation.scores` 包含 3 個分數。既有 Runner 回歸涵蓋設定覆蓋、輸出產物、同名目錄保護及 Windows 寫檔重試：

```powershell
python -m pytest -q tests/test_config_network_runner.py
```

## 五、與後續交付銜接

本指引完成 #29 的安裝、冒煙、主實驗與輸出檢查部分。失敗用例回放入口由 #25 提供，跨種子統計入口由 #28 提供；其命令隨對應功能整合至本指引。#13 的非實作者復跑紀錄與 PR #32 的交叉驗證相互對照。
