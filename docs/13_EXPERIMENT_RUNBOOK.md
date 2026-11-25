# 13｜Qwen3-0.6B 実験Runbook（実行手順と合否条件）

> **状態：実験実行に必要な最小コードとCLIをリポジトリへ追加済み。**
> **2026-11-25のCPU統合検証：** 接続先Linux aarch64、Python 3.12.3、PyTorch 2.8.0+cpu、Transformers 4.57.3、FP32、1スレッドで、公式Qwen3クラスの小型ランダム初期化モデルを使い**13テストPASS**。pre/post-residualの両方で6ケースのEager/Lazy一致、全ON差0.0。結果は[実測JSON](../results/qwen3_synthetic_cpu_2026-11-25.json)。**事前学習済み重みによる品質・GPU速度の証拠ではない。**
>
> **事前学習済みQwen3-0.6Bの初回CPU検証もPASS。** 固定revision、FP32、Batch1、1プロンプト・2 Decode位置、post-residual、6マスク列で全ONおよびEager/Lazyの最大Logits差0。 [実測JSONと実行コマンド](../results/qwen3_real_cpu_fp32_2026-11-25.json)。長系列、BF16/CUDA、品質は未検証。自動的な有料GPUジョブは起動していない。

## 0. 最短の実験起動

```bash
python -m pip install -e '.[test]'
python scripts/run_pilot.py --stage smoke
# 実モデルの数値正確性だけを検査（GPU環境）
python scripts/run_pilot.py --stage real --device cuda
# 正確性がPASSしてからHead学習・開発評価・壁時計計測
python scripts/run_pilot.py --stage pilot --device cuda
```

[一括実行スクリプト](../scripts/run_pilot.py)は各段階のコマンド・ログ・終了コードを新しい `runs/` ディレクトリへ保存する。モデルRevision既定値は Hugging Face APIで2026-11-25に確認した `c1899de289a04d12100db370d81485cdf75e47ca`。`--revision`で別の40桁SHAを指定可能。実モデル段階には明示的な `--device` が必要。GPUジョブ契約や課金は行わない。

学習が更新ゼロなら終了コード3と `NO_EFFECTIVE_UPDATE` を保存し、checkpoint作成および後続評価を停止する。`--stage pilot`は**小規模デバッグ試行**であり、研究の最終検証ではない。

## 1. 目的・確定した研究骨格

[01_FOUNDATIONS.md](01_FOUNDATIONS.md)および[11_SHARED_ACTIVATION_ROUTING.md](11_SHARED_ACTIVATION_ROUTING.md)を先に読む。

- **第1Attentionの計算済み活性を共有。** Transformer本体の第1層Self-Attention終了後に、出力射影後の活性を再利用する。別のAttentionを追加しない。
- **第1層MLPと最終層は必ず実行。** Layer Executorの入力は元の第1Block完了後のHidden State。
- **唯一の学習対象はLinear Head。** Decoder全層、Embedding、LM Headは凍結。
- **Token-wise all-middle mask。** 各Decodeトークンで中間26層のON/OFFを一括選択。固定SkipはBaseline。
- **KV再構成は支援機構。** 正しい層入力からK/Vを射影し、次回のAttention実行前に補完する。
- 第2段階はNanbeige4.3-33B。**本RunbookのQwen3実行器をそのまま33Bへ適用できるとはしない**（アーキテクチャ差）。

## 2. バージョン・Backendと既知の制約

| 要素 | 実験コードの前提 |
|---|---|
| モデル | Qwen/Qwen3-0.6B、**実行時に完全なCommit SHAへ固定** |
| Transformers | **4.57.3** をピン留め |
| PyTorch | **CPU統合検証は2.8.0+cpu**、環境に合ったCUDA Wheelを別途用意。GPUでは未検証 |
| Attention | **eager** を強制。FlashAttention、SDPA、torch.compileは初回G1対象外 |
| Cache | Transformers 4.57.3の **DynamicCache(config=model.config)**。Static/Sliding/Offloadを未サポート |
| Batch | **1のみ**。異なるMaskを持つBatch=8は実装・検証対象外 |
| Decode長 | 1 Stepで1 Token。Prefillは**全層通常実行** |
| Tap | コマンドでpre_residual/post_residualを選択。**post_residualは試験用デフォルトであり研究上の確定ではない** |
| Precision | CPU smokeはFP32、GPU候補はBF16（実装上の前提であり速度の証拠ではない） |
| CPU同期 | MaskをPython側へ読む`tolist`/`item`がある。**本コードは性能最適化済みではない** |
| モデルライセンス | Qwen3モデルの利用条件を別に尊重。リポジトリは第三者重みを含まない |

[Qwen3のDecoderLayerとAttentionの一次コード](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py)で`input_layernorm`, `self_attn.k_norm`, `rotary_emb`, `past_key_values.update`の契約を確認した。ただし、**コードを読んだだけでは統合動作は証明されない**。

### 初期セットアップ（ローカルGPU計算機等で実行）

```bash
git clone https://github.com/Unjuno/frozen-llm-layer-skip-kv.git
cd frozen-llm-layer-skip-kv
python -m venv .venv
source .venv/bin/activate

# CUDA版PyTorchを使用する場合はGPU/Driverに合う公式インストール方法で先に導入
python -m pip install -U pip
python -m pip install -e '.[test]'
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

**GPUが必要な検査とGPUなしでもできる検査を混同しない。** 環境はOS、GPU名、Driver、CUDA/Backend、GPU Clock、dtype、Batch、Input/Output長、Seed、Model SHAとともに記録する。

## 3. まずネットワーク不要の小型Qwen3構造でテスト

Transformersをインストールした後に、事前学習済み重みを使わず**小さなQwen3モデル構造**をランダム初期化する。これは独立した通常ForwardとMasked-All-Onを比較できる最小の統合試験。

```bash
pytest -q tests/test_qwen3_reference.py

python experiments/qwen3/run.py verify \
  --model synthetic --device cpu --dtype float32 \
  --decode-steps 6 --tap post_residual \
  --output results/qwen3_synthetic_g1.json
```

**PASS条件（暫定）：** 同じ入力・固定トークン列に対してnativeとmasked all-onのLogits最大絶対差がFP32で2e-4以下、第1Attentionの呼出しはDecodeトークン当たりちょうど1回、1st MLPはPrefill1回＋Decode回数、EagerとLazyが同一マスク列で一致。各層の末尾未補完位置数がMask履歴から計算した数に一致。

SyntheticでPASSしても**Qwen3-0.6Bの事前学習済み重みに対するG1の合格にはならない**。

## 4. 実Qwen3のチェックポイントを固定してG1を実施

事前学習済みモデルの完全Commit SHAを取得する。

```bash
export QWEN_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen3-0.6B").sha)')"
echo "$QWEN_REVISION"
```

そのSHAでモデルとTokenizerを固定して検査する。例：

```bash
python experiments/qwen3/run.py verify \
  --model Qwen/Qwen3-0.6B --revision "$QWEN_REVISION" \
  --device cuda --dtype bfloat16 --tap post_residual \
  --decode-steps 6 \
  --output results/qwen3_real_g1_bf16.json
```

**全ONは元のHF Forwardをそのまま通るため、その一致は自明化する部分がある。** 追加の独立検証として、Tapの形状・残差位置、KVの位置変換、実行経路の意味論、修正を入れないBaselineの同時比較を確認する。

BF16のLogits最大絶対誤差の暫定値は5e-2（**不確かな仮設定**）。値がこれを超えたらFAILであり、数値誤差・RoPE・正規化・KVレイアウトを調べる。正しい閾値はPilotから再設定して独立Test前に固定する。

**Error caseを無視して次工程へ進まない。** `verify`はFAIL時に終了コード2を返す。Qwen3の実行が失敗したときも「測定なし」と記録する。

## 5. KVモード別の壁時計比較（参照実装のみ）

```bash
python experiments/qwen3/run.py bench \
  --model Qwen/Qwen3-0.6B --revision "$QWEN_REVISION" \
  --device cuda --dtype bfloat16 \
  --decode-steps 32 --warmup 3 --repeat 10 \
  --output results/qwen3_reference_wall_b1.json
```

Native／All-On／固定Skip／動的Eager／動的Lazyを実行する。**Batch1、固定トークンReplay、CPUからの逐次Dispatchを含む端から端のDecode壁時計時間**のみ比較する。

この方式は`item`と`tolist`によるCPU同期などがあり、**本格GPU高速化を示すものではない**。小さいトークン数や初回コンパイル等の影響を除外し、ウォームアップ・複数反復・クロック条件を記録する。NsightやCUDA Eventは後続の最適化段階で導入する。

## 6. 生成中のLinear Head学習（RLOO Pilot）

Qwen3実モデルのG1とKVの条件を満たした後のみ、**元のモデルを凍結**してLinear Headをオンポリシー学習する。デバッグ用入力：[train_multistep.jsonl](../data/pilot/train_multistep.jsonl)。

```bash
python experiments/qwen3/run.py train \
  --model Qwen/Qwen3-0.6B --revision "$QWEN_REVISION" \
  --device cuda --dtype bfloat16 --tap post_residual \
  --dataset data/pilot/train_multistep.jsonl \
  --epochs 1 --rollouts 4 --max-new-tokens 16 \
  --exploration 0.05 --cost-weight 0.02 \
  --checkpoint checkpoints/qwen3_router_pilot.pt \
  --output results/qwen3_router_train_pilot.json
```

- **実生成**：最初のトークンは全層Prefillから出力、以後は毎Step共有Attention活性に基づき全中間層マスクをサンプリングして生成する。
- **Headのみ学習**：LLMの全重みを凍結。Mask系列の実際の`log_prob`と同一Promptの独立RolloutでLeave-One-Out REINFORCEを計算する。
- **学習信号の注意**：正解一致と文字列類似度の組は**暫定品質代理値**、実費用でなく平均ON率は**費用代理値**。両者の学術的妥当性は未保証。すべてのRolloutで品質差がない場合はデフォルトで更新をSkipする。
- **学習結果は品質保証ではない**：このデバッグデータだけで未知問題の正解率向上や高速化を主張しない。将来のHoldout評価と最良固定Skipとの同一費用での比較を別に構築する。
- 複数整数の回答を求める `train_multistep.jsonl` / `dev_multistep.jsonl` を使用する。最初の出力トークンは全層Prefillで決まるため、単一整数問題ではHeadが品質に影響できる区間が不足する。開発データ評価は `evaluate.py` に分離。
- **初期ON確率0.95（探索前）は暫定値**。同品質かつ全不正解のRolloutを除く品質ゲートはバイアスを生じ得るヒューリスティックで、厳密な不偏REINFORCEと区別する。更新が有効でなければ結果のみ保存して停止する。

### 6.1 開発データでの学習済みHeadの比較（デバッグ用）

学習済みCheckPointがある場合、**実際の自己回帰生成**を使い、元の全層LLM・単純な固定Skip・Headが各Tokenで選ぶ経路を開発データ上で比較する。

```bash
python experiments/qwen3/evaluate.py \
  --model Qwen/Qwen3-0.6B --revision "$QWEN_REVISION" \
  --device cuda --dtype bfloat16 --tap post_residual \
  --checkpoint checkpoints/qwen3_router_pilot.pt \
  --dataset data/pilot/dev_multistep.jsonl \
  --max-new-tokens 16 \
  --output results/qwen3_pilot_dev.json
```

この`evaluate.py`は**平均ON率・完全一致率・Decode壁時計**を出力する。既存の固定マスク候補は`every_other`と`every_fourth`に限られ、**開発データで最適化した「最良固定マスク」ではない**。答えの長さも各方式で異なるため、時間比較は探索用の説明指標であり、研究の最終判定には用いない。

**品質の改善や非劣性は、独立したHoldoutセット・複数Seed・同じ費用予算・適切な統計検定を実施するまで未実証。**

## 7. 11月24日の行列乗算論文をどう評価するか

**`ω=9/4`の論文にあるアルゴリズムを使う実装は、現在の実行器に含まれていない。** 理論的な平方行列乗算の指数を、狭いK/V射影GEMMの高速化の保証と解釈しない。

K/V補完の射影の中で、**1つずつのGEMV vs 複数入力のGEMM vs K/V重みを結合したGEMM**を、まず同じdtype・M・形状で比較する（GPU必要）。

```bash
python experiments/qwen3/bench_kv_projection.py \
  --model-shape qwen3-0.6b --device cuda --dtype bfloat16 \
  --m-values 1,2,4,8,16,32,64,128 \
  --warmups 5 --repeats 15 \
  --output results/qwen3_kv_projection_bf16.json

# 33B側は行列形状のみを模したマイクロベンチ（モデル重みは未使用）
python experiments/qwen3/bench_kv_projection.py \
  --model-shape nanbeige4.3-33b --device cuda --dtype bfloat16 \
  --output results/nanbeige_shape_kv_projection.json
```

あくまで**射影だけのMicrobenchmark**であり、RMSNorm、RoPE、KV追加、GPU依存待機、全体Decode時間、品質は含まれない。正方行列乗算の`ω=9/4`理論だけでは結果を予想できない。[一次論文](https://github.com/openai/math/blob/main/preprints/The-matrix-multiplication-exponent-is-nine-fourths-November-24-2026/README.md)／[Lean形式化範囲](https://github.com/openai/math/blob/main/lean/docs/379.md)。

## 8. H/T/D/C/U と ERROR CHECK

| 分類 | 事前に指定する条件・現在の状態 |
|---|---|
| **H** | H-G1: All-OnがNativeと一致。H-G5: 同じMask/トークンのEagerとLazyでLogits差が許容以下。H-G3: Linear Headだけ更新でき、凍結重みが変わらない |
| **T** | 小型Qwen3合成構造→実Qwen3固定Revision→GPU Batch1→複数Seed（学習）。**結果がない部分は未実施** |
| **D** | PASS: 各数値・マスク・KV不変条件を満たす。FAIL: 差分超過・欠損KV・Frozen weight更新。UNCERTAIN: GPU非提供、モデル未取得、条件/標本不足 |
| **C** | RoPE位置ずれ、Qwen3のK正規化誤り、Cache更新順、マスク間相互作用、Routerの低表現力、CPU同期、品質報酬の全ゼロ |
| **U** | FP32/BF16丸め、GEMMの順序、Backend、GPU Clock、Kernel選択、Rollout Seed、評価データ。Pilotの誤差を無条件に他環境へ流用しない |

**ERROR CHECK：** 小型モデルと実重みの短いCPU数値検証の実測は上記JSONで確認できる。本資料のGPU・学習・品質評価のコマンドは準備済みだが未実行。短い固定ReplayのPASSを学習成功・速度向上・長系列の一般的な正確性と混同しない。

[研究の骨格](01_FOUNDATIONS.md)｜[共有活性](11_SHARED_ACTIVATION_ROUTING.md)｜[実行コード](../experiments/qwen3/run.py)｜[最適化の評価](04_EVALUATION.md)
