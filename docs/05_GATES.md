# 05｜検証ゲート・現在地・未決事項

> **管理方針：** 期限なし。2026-11-18のユーザー依頼を受け、**CPU小型合成Decoderでの参照正確性テスト**を実施した（[結果](12_CPU_CORRECTNESS_PILOT.md)）。**11月25日：実Qwen3-0.6Bの短いFP32 CPU初回検証はPASS、GPU/BF16を含むG1の全条件は未完了**（[実測](../results/qwen3_real_cpu_fp32_2026-11-25.json)）。研究目的は[01](01_FOUNDATIONS.md)、[11](11_SHARED_ACTIVATION_ROUTING.md)が正本。

## 現在の状態

| 項目 | 状態 | 根拠 |
|---|---|---|
| GitHub公開リポジトリ | **作成済み** | Publicで`main`に文書を配置 |
| 研究目的・基本構造 | **文書化済み** | [01 研究目的とモデル構造](01_FOUNDATIONS.md) |
| 学習・KV・GPUの評価手順 | **計画文書として整理済み** | [02](02_ROUTER_TRAINING.md) / [03](03_KV_CACHE.md) / [04](04_EVALUATION.md) |
| CPU合成モデルの参照実行器 | **実装・初回テストPASS** | [12 CPU Pilot](12_CPU_CORRECTNESS_PILOT.md)。Qwen3重みは未使用 |
| Qwen3の参照実行器 | **小型公式Qwen3構造でCPU 13テストPASS** | [実測](../results/qwen3_synthetic_cpu_2026-11-25.json)／[13 実験Runbook](13_EXPERIMENT_RUNBOOK.md) |
| Qwen3-0.6B実重みによるG1 | **CPU短系列Pilot PASS／GPU・長系列は未完了** | [実測](../results/qwen3_real_cpu_fp32_2026-11-25.json)。1プロンプト、FP32、2 Decode位置、6マスク列の最大Logits差0 |
| 合成モデルのHead更新・Eager/Lazy KV | **参照テストPASS** | 1回の合成REINFORCE更新、29種類固定Replay。ただしタスク正解0件 |
| 実モデルのHead学習 | **実装済み・未検証** | [一括実行](../scripts/run_pilot.py)。実Qwen3の学習更新は未確認 |
| GPU速度・品質改善 | **未測定** | ベンチマーク結果なし |
| Nanbeigeへの移植 | **未着手** | 検証ログなし |

「文書化済み」は「実験で合格した」を意味しない。

## 期限を置かない段階移行

| Gate | 何をするか | 必須の通過条件 |
|---|---|---|
| **G0: 計画の公開** | 研究目的・固定仕様・比較方針・未決事項を整理 | 文書が整合し、公開・参照できる |
| **G1: 全ONの再現** | Qwen3-0.6B Baselineと任意26ビットマスク実行器 | 全ONで元モデルと数値整合、確定層を必ず実行 |
| **G2: Eager KV** | スキップ層の入力からKVを生成 | 位置とAttention、同一経路再現性が正しい |
| **G3: Router学習** | 複数プロンプト・複数マスクで自己回帰生成・報酬学習 | Routerのみ勾配更新、学習データを再現可能に記録 |
| **G4: 品質判定** | 未見入力で全層・固定Skip・学習済みRouterを比較 | 誤→正と正→誤、品質・費用トレードオフを報告 |
| **G5: 遅延KV** | On-demand/Batched/Shared/Adaptiveを追加 | **同一マスク・トークン列のEagerとの差**が許容範囲内、メモリ不変条件を守る |
| **G6: 共同最適化** | GPU費用表、単純結合D1と再学習D2を比較 | 追加利益／失敗が公平な実測で判断できる |
| **G7: 33B検証** | Nanbeigeに移植、Routerを再学習 | 52層構造・KV/VRAMを検証し、品質・速度を独立評価 |
| **G8: 結果の公開** | 再現コードと実測結果・論文原稿 | HW、Backend、Seed、統計、否定的結果を含む |

G1〜G8は**検証作業の順序**であり、日付や作業時間の約束ではない。GPU資源の確保に応じて実験の順序変更は可能だが、比較に必要な先行条件は記録する。

## 次の検証：G1

**実Qwen3-0.6Bの短いFP32 CPU PilotはPASS。以下のGPU/BF16・長系列の検証は未完了：**

1. Qwen3-0.6Bの固定revision・Tokenizer・BF16・Backendの記録と通常全層Baseline。
2. 第1・最終層が常時ONで、任意の26ビットマスクを適用できる実行器。
3. 全ONと元モデルのLogitsの一致テスト。
4. 層入力・位置・実行マスク・ログを記録する仕組み。

ルータの学習、Eager/LazyのGPU性能、33Bへの拡張を、G1の成功より先に「完成」として記録しない。

## 未決の技術事項

| ID | 未決事項 | どこで確定するか | 判定資料 |
|---|---|---|---|
| O-01 | 実GPU機種・枚数・GPU時間 | G1前 | VRAM・メモリ帯域・Clock |
| O-02 | Qwen3の正確性実装Backend、Attention仕様 | G1 | Model revision・Logits差 |
| O-03 | 報酬のタスク・正解判定器・`λ`・探索率 | G3 | Pilot学習曲線・正解率・費用 |
| O-04 | 数値一致許容幅 | G1・G5前 | BF16丸め・比較Kernel |
| O-05 | Quality非劣性幅、Test件数、主要検定 | G4前 | Pilot分布・必要検出力 |
| O-06 | KV補完のメモリ予算・期限・切替ルール | G5前 | Eager/Lazy Microbenchmark |
| O-07 | Serving BackendとGPU費用モデル | G6前 | Nsight、Batch 1/8 |
| O-08 | Nanbeige BF16/量子化・複数GPU構成 | G7前 | 利用可能VRAM・品質比較 |
| O-09 | 先行研究との新規性境界 | 成果の主張前 | 一次論文・再現条件 |

## Gate記録テンプレート

```markdown
## Gx: [検証名]
- 状態：未開始 / 実施中 / PASS / FAIL / UNCERTAIN
- 目的・仮説：
- 環境：モデルrevision、GPU、Clock、CUDA、Backend、Precision、Batch、Seed
- 入力：データ版、プロンプト数、入力/Decode長
- 実行方法と比較対象：
- 期待される合否・数値許容幅：
- 測定結果と信頼区間：（未測定なら明記）
- 不確かさ・失敗例：
- 判断と次に試すこと：
- 関連Commit / 実験成果物：
```

## 運用方法

- 方針変更は[設計判断ログ](DECISIONS.md)へ、実験の判断は上記のGate記録へ。
- 説明を重複させず、数式は[02](02_ROUTER_TRAINING.md)、KV不変条件は[03](03_KV_CACHE.md)、測定方法は[04](04_EVALUATION.md)を参照する。
- **期日で進行判定しない。** 数学的な正確性、再現可能な記録、実測証拠の順に進める。

[← 04 実験設計](04_EVALUATION.md)｜[文書ガイド](README.md)
