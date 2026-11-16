# 04｜比較実験・GPU測定・統計判定

> **状態：** 実験設計案。GPU速度、精度、品質改善は未測定。数値閾値と標本数は**Pilotで最終確定する提案値**である。

## 研究の主たる比較と支援的な比較

**主たる研究の評価対象は、第1層Attentionの共有活性から単一Linear Headがトークン別の実行領域を学習する方式。** 同じ計算・品質条件で固定Skipとの差を検討することが先で、D1/D2のKV費用込み共同最適化は**派生的な比較**である。[01](01_FOUNDATIONS.md)・[11](11_SHARED_ACTIVATION_ROUTING.md)を参照。現時点では実験を実施しない。

## 研究で最も重要な区別

1. **層を減らした** ≠ **GPU時間が短くなった**。
2. **探索Oracleが正解した** ≠ **学習済みRouterが未知問題で一度に正解した**。
3. **EagerとLazyの速度差**を調べる時は、**トークン列・マスク列を完全固定**しないと比較が壊れる。
4. **Router＋Schedulerの組合せ** ≠ **共同最適化の追加利益**。単純結合との比較が必要。

## 仮説と必要な証拠

| 仮説 | 確かめたいこと | 主要な対照 |
|---|---|---|
| **H1** 学習 | 学習済みLinear Routerは、同等の平均実行層数・費用の固定Skipより良い品質・費用トレードオフを示す | A1 vs B2/B3 |
| **H2** KV | 同一経路で、EagerよりLazy/Batched/Adaptiveが実測遅延またはVRAMで有利 | B3 vs C1/C2/C3（固定Replay） |
| **H3** 共同最適化 | GPU/KV費用を含めたRouter再学習が、単純にSchedulerを後付けした方式より良い | D1 vs D2 |
| **H4** 品質改善 | 元モデルより正解率が高い経路を**未知プロンプト**で学習済みRouterが選べる | A0-native vs D2 |

H4の改善例を示すときは、元モデルの「誤→正」と「正→誤」を**両方**集計する。学習中の最良軌跡（探索Oracle）は性能上限の参考値とし、**公開モデルの一回の推論性能に混ぜない**。

## 比較実験の一覧

| ID | 実行経路 | KV方式 | 主要目的 |
|---|---|---|---|
| **A0-native** | 元モデルの全層実行 | 通常 | 品質・速度の基準 |
| **A0-all-on** | マスク実行器の全ON | 通常 | 実行器・Router呼出しによる純粋な費用 |
| **A1** | 固定スキップ | Eager | 固定経路との比較 |
| **A2** | 未学習ランダム | Eager | 探索のみ |
| **B1** | 単純REINFORCE | Eager | 方策勾配の効果 |
| **B2** | REINFORCE＋Leave-one-out | Eager | 基準化の効果 |
| **B3** | B2＋段階的探索 | Eager | Curriculumの効果 |
| **C1** | **B3と同一マスク列** | On-demand | 遅延による効果 |
| **C2** | **B3と同一マスク列** | Batched | 一括補完による効果 |
| **C3** | **B3と同一マスク列** | Adaptive Eager/Lazy | メモリ・費用切替の効果 |
| **D1** | 学習済みB3 Routerは固定 | C3 Scheduler | 単純結合 |
| **D2** | 同じSchedulerでRouterを再学習 | C3 Scheduler | **共同最適化** |
| **MLP-control** | 小型MLP Router（対照） | Eager | Linearの能力限界の参考 |

**静的経路への退化を監査：** 同一の層群をすべての入力で常にSkipするなら、動的Routerに固有の有効性は実証できない。A1には、**学習・開発データから選んだ最良の固定経路**を含め、学習済みRouterと同等の総GPU費用（または品質・費用のPareto曲線）で比較する。トークン間のマスクの一致率や変化率は**診断量のみ**とし、変化を強制する追加報酬も、同一マスク維持の追加報酬も設定しない。

**公平性：** C1〜C3は、B3のマスクと生成トークンを保存してReplayする。速度差をSchedulerのみに帰属できるよう、**同一Backend・GPU・精度・バッチ・文脈長**を使う。生成品質の比較は別途、通常の自己回帰生成で行う。

**追加の統制：** D2には再学習の計算予算が追加されるため、D1との公平な比較では追加の探索費用も報告し、単純に計算量の多さの利益を「共同最適化の新規効果」と呼ばない。

## 測定条件（Pilot候補）

| 項目 | 初期条件と未決事項 |
|---|---|
| モデル | Qwen3-0.6B → Nanbeige4.3-33B、revision固定 |
| 推論の範囲 | Prefill全層／Decode動的スキップ、Thinking設定を固定 |
| 精度 | BF16基準。必要な量子化は別実験 |
| Batch | **1・8**を必須候補 |
| 入力長 | 128・1,024 tokenを候補 |
| 固定Decode長 | 128・512 tokenを候補 |
| Hardware | GPU型番・枚数・VRAM・Clock・電力制限・ドライバ・CUDA |
| Software | PyTorch・Transformers・Tokenizer・Attention Backend・実行器・Seed |
| 測定 | Warm-up後のCUDA Eventsと壁時計、Nsight Systems/Compute |
| 反復 | Pilotで必要反復数を設計。平均だけでなく中央値・p95を報告 |
| データ分割 | Train/Validation/Pilot/Testは分離し、重複・リークを確認 |

**測定時の禁止事項：** 単にCPUでKernelをenqueueした時間だけをGPU処理時間とみなすこと、自然EOSによって出力が短くなった時間短縮を計算高速化と呼ぶこと、異なる量子化精度を同条件として比較すること。

### 必須指標

| 分野 | 主指標 | 補助情報 |
|---|---|---|
| 品質 | Held-out NLL、正解率 | 生成形式・早期EOS・繰返し・元モデルとのLogits差 |
| 品質改善 | 誤→正、正→誤の対応付き件数 | McNemar・対応付きBootstrap |
| GPU速度 | **固定Decode長の実時間** | TPOT、Tokens/s、End-to-end、p95 |
| GPU効率 | 実行層集合・重みアクセス・Kernel時間 | Gather/Scatter、CPU同期、Stream待機 |
| KV | 補完トークン数、待機、未補完負債 | Eagerと同一経路の数値整合性 |
| メモリ | ピークVRAM、保存入力量 | Allocator・共存中のKV容量 |
| 学習 | 未見データでの品質・費用 | Rollout数、報酬分散、Seed、Entropy、学習計算量 |

## GPU実効時間の判断

**変数表**

| 記号 | 意味（日本語・定義） | SI単位 | 定義域・条件 | 型 |
|---|---|---|---|---|
| \(T_{\rm full}\) | 全層実行のDecode時間 | s | 正 | スカラー |
| \(T_{\rm new}\) | 提案方式のDecode時間 | s | 正 | スカラー |
| \(T_{\rm saved}\) | スキップによりクリティカルパスで節約した実効時間 | s | 非負 | スカラー |
| \(T_{\rm overhead}\) | Router・補完・同期で増えた実効時間 | s | 非負 | スカラー |
| \(S\) | 全層に対するDecode時間改善率 | 1 | 実数 | スカラー |

```math
T_{\rm new}=T_{\rm full}-T_{\rm saved}+T_{\rm overhead}
```

```math
S=1-\frac{T_{\rm new}}{T_{\rm full}}
```

Decode時間を5%以上減らすには、

```math
T_{\rm saved}-T_{\rm overhead}\ge0.05T_{\rm full}
```

が必要。**次元チェック：** 全時間量はs。Sは時間比s/sで無次元、0.05も無次元。

この式はクリティカルパス上の**実効**時間の分解であり、Stream上で重なる個々のKernel時間を足し引きしてよいという意味ではない。

## 統計的な判定（H/T/D/C/U）

| 区分 | 事前登録する事項 |
|---|---|
| **H：仮説** | H1〜H4を個別に検証。H3はD1対D2、H4は未見テストで評価。 |
| **T：最小検証** | Pilot（暫定128〜256）、複数学習Seed（最低3を候補）、独立Test。Pilot結果から評価件数と停止ルールを確定し、Testで恣意的に変更しない。 |
| **D：合否** | **暫定PASS**：品質非劣性を満たし、速度改善点推定が5%以上、対応付き95%CI下限>0。**強いPASS**：速度改善95%CI下限>5%。十分な精度で否定ならFAIL、精度不足ならUNCERTAIN。 |
| **C：対立仮説** | Router表現不足、探索分散・過学習、KV同期待ち、メモリ帯域、Schedulerの費用超過など。 |
| **U：不確かさ** | プロンプト、学習Seed、サンプリング、GPU Clock、Backend・Kernel選択。対応関係を崩さないBootstrap等を用いる。 |

**暫定の品質非劣性幅（未決）：** NLL増分0.02 nat/token、主要タスク正解率低下2パーセントポイント。これらは**実験結果ではなく例示の閾値**。Test前にPilotから根拠を記録して固定する。主要仮説・多重比較の扱いもTest前に定義する。

**ERROR CHECK：** 速度改善の「点推定5%以上・CI下限>0」は「真の改善が5%以上」という強い主張を意味しない。「CI下限>5%」を別の強い合格条件にする。品質が非劣性かどうかも推定値だけでなく区間推定で確認する。

## 参照先と最終判断

- Routerの学習条件：[02 Router学習](02_ROUTER_TRAINING.md)
- KVの正確性条件：[03 KV補完](03_KV_CACHE.md)
- Gateの進行条件：[05 検証ゲート](05_GATES.md)
- GPU Servingと既存の動的Layer Skippingの比較文献：[先行研究・技術的出典台帳](PRIOR_ART_AND_ATTRIBUTION.md)
- 未確定の最適化候補とパイプライン：[06 最適化対象](06_OPTIMIZATION_TARGETS.md)／[07 学習方策](07_POLICY_LEARNING_PIPELINE.md)。**比較実験IDは正式設計が確定するまで変更しない。**

[← 03 KV補完](03_KV_CACHE.md)｜[05 検証ゲート →](05_GATES.md)｜[文書ガイド](README.md)
