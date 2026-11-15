# Repository Policy — research notebook and publication workflow

> **Purpose:** Public, inspectable record of *why* and *how* the research is conducted. The repository is not a claim that the algorithm is implemented or that the benchmarks succeeded.

## 1. 正本と守備範囲

- `README.md`: 研究目的、固定仕様、進捗の意味、詳細文書への入口。
- `docs/RESEARCH_PROTOCOL.md`: 研究手法・数学的前提・実験の公平性・完了条件の**正本**。
- `docs/REPOSITORY_POLICY.md`: 公開・変更・再現性・コミット方法の規約。
- `LICENSE`: リポジトリの文書・将来追加するオリジナルコードの利用許諾。

現段階では**研究方法と作成方針のみ**を公開する。コード、テスト、Benchmark、実験ログは、実際に着手した段階で追加する。未着手工程用の空ディレクトリは必要ない。

## 2. 変更方法：`main`を直接更新

- **既定ブランチは`main`**。原則として`main`へ直接コミットする。
- PR、Feature Branch、Merge Commitを通常の研究ノート更新には要求しない。
- 変更は小さく分け、コミットメッセージに意図を記載する。
- 他者の貢献や大規模変更が必要になった場合だけ、レビュー手続きを別途検討する。既定の運用を黙って変更しない。
- GitHub上の記録を正本とし、ローカルの草稿と相違があればGitHub側に反映する前に差分を確認する。
- 履歴は上書き（force-push）より追記コミットを優先し、研究方針を変更した根拠を残す。

推奨コミット例：

- `docs: clarify fixed first and last layer execution`
- `docs: specify on-policy routing and leave-one-out baseline`
- `docs: define KV consistency invariants`
- `docs: revise H3 comparison to control training budget`

## 3. 研究上の変更を記録する際の必須項目

設計変更のたびに、該当文書へ次を記載する。

1. **変更の対象**：Router、報酬、KV状態、GPU実行、品質評価など。
2. **旧仕様と新仕様**：何をどこまで変更したか。
3. **変更理由**：論文、数学的根拠、GPU測定、実装制約などの裏付け。
4. **比較可能性**：以前の実験との比較が有効か、再測定が必要か。
5. **ステータス**：確定仕様／仮説／未検証／実測による確認／棄却。

新しい実測値を書く場合は、GPU型番、実行Backend、精度、Batch、入力長、出力長、モデルrevision、乱数Seed、測定手順を併記する。

## 4. 研究結果の品質管理

- 未測定の速度改善率、実測していないBenchmark、未実装の機能を実施済みと記載しない。
- 仮説と結果を明確に分け、**否定的結果も保存**する。
- Native Full、Masked All-ON、固定/ランダムRouter、学習済みRouter、同一経路KV Replayを分けて比較する。
- 学習・開発・最終評価データを分離し、Testデータを学習・モデル選択に再利用しない。
- 数式には変数表と単位チェックを付け、測定にはGPU/Clock/Batch/Backend等の条件を付ける。
- 新規性は一次文献との比較で主張し、単なる先行研究名の列挙で確定しない。

## 5. 公開・セキュリティ

- APIキー、アクセストークン、秘密情報、非公開ログ、機密プロンプトをコミットしない。
- モデル重み、データセット、論文本文、第三者コードは**それぞれのライセンス・利用規約**を確認する。
- 巨大な重み、データセット、キャッシュを安易にGit履歴へ格納しない。
- 結果を公開する際、必要な依存バージョン、再現手順、権利・出典を記録する。
- 著者・寄与者のクレジット、第三者素材の表示義務を尊重する。

## 6. 進行管理（カレンダー期限なし）

期間ではなく、`docs/RESEARCH_PROTOCOL.md`のGate G0〜G8で次段階への移行を判定する。Gateには少なくとも次を記録する。

- 検証条件・成功/失敗/不確定の基準
- 実行環境と数値誤差の許容範囲
- 比較した実装とマスク・トークンの固定条件
- 測定値（未実施の場合は「未測定」とする）
- 次に進む／同じ段階を続ける判断の根拠

**「リポジトリを完成させる」ことと「研究仮説が成立する」ことは独立した状態として扱う。**
