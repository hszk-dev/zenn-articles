---
title: "LLMの「一発書き」を卒業する。AI Scientistに学ぶ「探索的生成(Tree Search)」の実装戦略"
emoji: "🌳"
type: "tech"
topics: ["LLM", "AI", "TreeSearch", "Python", "機械学習"]
published: true
---

## はじめに

LLMアプリケーションの開発において、単発生成による出力品質のばらつきは深刻な課題です。「今回は良い結果が出たが、次回は期待外れ」といった不安定性は、プロダクション環境での信頼性を大きく損ないます。

本記事では、AI Scientist-v2で採用されている「探索的アプローチ」を**記事生成タスクに応用**する手法を解説します。AI Scientist-v2は本来、科学実験の自動化（コード生成・実行・改善）にTree Searchを活用していますが、この探索アーキテクチャの本質である「候補展開→評価→選択→剪定」のサイクルは、テキスト生成タスクにも有効に適用できます。

本記事で紹介する実装のソースコードは [GitHub](https://github.com/hszk-dev/ai-tech-writer) で公開しています。
　
## なぜLLMに「探索」が必要なのか - 単発生成の限界

![ツリーサーチ](/images/ツリーサーチ.jpg)

LLMの出力は確率的な性質を持つため、同じプロンプトでも毎回異なる結果が生成されます。特に温度パラメータ（temperature）を高く設定するとクリエイティブな出力が得られる一方で、一貫性が失われがちです。逆に温度を低くすると安定性は向上しますが、創造性に欠ける平凡な出力になりやすいという根本的なトレードオフが存在します。

プロダクション環境では、この品質のばらつきが大きな問題となります。ユーザーに提供する記事やレポートの品質が実行ごとに大きく変動するのは、ビジネス上受け入れられません。従来の単発生成アプローチでは、一度の生成で最高品質の出力を得ることは困難であり、複数の候補から最適解を選択する「探索的アプローチ」が必要となるのです。

## AI Scientist-v2のTree Searchアルゴリズム解説

![探索サイクル](/images/探索サイクル.jpg)

AI Scientist-v2で採用されているBest-First Tree Search（最良優先木探索）は、**科学実験プロセスの自動探索**を実現するアルゴリズムです。重要な点として、このTree Searchは論文の「執筆」ではなく、**実験の設計・実行・改善**に適用されています。

本家の実装では、各ノードは以下の要素を保持します：

- `code`: 実験を実行するPythonコード
- `plan`: 実験計画の説明
- `metric`: 実験結果（validation lossなど）
- `is_buggy`: コードエラーの有無

探索プロセスは4つのステージで構成されます：
1. initial_implementation（基本実装）
2. baseline_tuning（ハイパーパラメータ調整）
3. creative_research（新規手法探索）
4. ablation_studies（アブレーション実験）
各ステージで複数の実験バリエーションを生成・評価し、最も有望な結果を示したノードを優先的に展開します。

この「候補展開→評価→選択→剪定」というサイクルの本質は、**探索対象を変えても有効**です。本記事では、実験コードの代わりに記事コンテンツを、実験metricの代わりに品質スコアを使用することで、同様の探索的改善を記事生成に応用します。

## 本家AI Scientist-v2との実装上の違い

本記事で紹介する実装は、AI Scientist-v2の探索アーキテクチャを**記事生成タスク向けにアダプテーション**したものです。両者の違いを明確にしておきましょう。

| 項目 | AI Scientist-v2（実験探索） | 本実装（記事生成） |
|------|---------------------------|-------------------|
| **探索対象** | 実験コードと結果 | 記事コンテンツ |
| **ノードの中身** | `code`, `plan`, `metric`, `is_buggy` | `idea`, `outline`, `draft`, `stage` |
| **評価基準** | 実験metric（validation lossなど） | 文章品質スコア（構造・正確性・可読性・実用性） |
| **並行処理** | `multiprocessing`（コード実行のため） | `asyncio`（API呼び出しはIOバウンド） |
| **ノード展開** | debug（バグ修正）/ improve（改善） | ステージ進行（INITIAL→EXPANDED→ENHANCED→POLISHED） |
| **選択戦略** | Best-First（スコア順） | Best-First（スコア順）+ 枝刈り |

**なぜこのアダプテーションが有効か？**

探索的生成の本質は「複数の候補を生成し、評価に基づいて最適解を選択する」ことにあります。AI Scientist-v2では実験結果（数値metric）で評価しますが、記事生成ではLLMによる品質評価で同様のフィードバックループを構築できます。

また、本家では`multiprocessing`を使用していますが、これは実験コードの実行がCPUバウンドだからです。一方、記事生成ではLLM APIの呼び出しがボトルネックとなるため、`asyncio`による非同期処理がより適切です。

### テキスト生成特有の課題

**1. 「デバッグ」に相当するフィードバックの不在**

AI Scientist-v2では、生成したコードを実行し、エラー（Traceback）が出れば修正するという**客観的な失敗信号**に基づくフィードバックループが存在します。一方、テキスト生成では「動く/動かない」という明確な境界線がありません。

本実装では、コードブロックの構文検証（`CodeValidator`）や実行検証（`DockerSandbox`）を探索ループに組み込むことで、部分的にこの課題に対処できます。

**2. 評価の客観性（LLMがLLMを評価する問題）**

AI Scientist-v2の評価指標はValidation Lossなどの**数値的Ground Truth**です。本実装ではLLMによる品質評価を使用しており、「LLMが好みやすい文章」が高評価される可能性（Reward Hacking）があります。

対策として、人間評価との相関検証や、外部指標（構文エラー率、実行成功率）の併用が有効です。

## 実装アーキテクチャ設計 - データ構造アルゴリズム

探索的生成の実装について、**データ構造**と**探索ループ**の2つに絞って説明します。`dataclass`を活用してボイラープレートを削減し、アルゴリズムの本質が見えるようにしました。

```python
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import uuid

class SearchStage(str, Enum):
    """探索の進行ステージ"""
    INITIAL = "initial"      # アイデア・アウトライン生成
    EXPANDED = "expanded"    # セクション詳細化
    ENHANCED = "enhanced"    # コード例の強化
    POLISHED = "polished"    # 最終仕上げ

@dataclass
class ArticleState:
    """記事の状態（3段階で進化）"""
    idea: Optional[str] = None       # 記事のアイデア・方向性
    outline: Optional[str] = None    # アウトライン（構成）
    draft: Optional[str] = None      # 完成した記事本文

    def get_current_content(self) -> Optional[str]:
        """現在の状態に応じたコンテンツを返す"""
        return self.draft or self.outline or self.idea

@dataclass
class TreeNode:
    """探索木のノード（IDベースで親子を参照）"""
    id: str
    stage: SearchStage
    article_state: ArticleState
    score: float = 0.0
    depth: int = 0
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    is_pruned: bool = False  # 枝刈りフラグ

    @classmethod
    def create_root(cls, topic: str) -> "TreeNode":
        """ルートノードを生成"""
        return cls(
            id=str(uuid.uuid4())[:8],
            stage=SearchStage.INITIAL,
            article_state=ArticleState(idea=topic),
        )

    def can_expand(self) -> bool:
        """展開可能かどうか"""
        return self.stage != SearchStage.POLISHED and not self.is_pruned

@dataclass
class SearchTree:
    """探索木の管理"""
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    root_id: Optional[str] = None

    def add_node(self, node: TreeNode) -> None:
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.root_id = node.id

    def get_expandable_nodes(self) -> list[TreeNode]:
        """展開可能なノードを取得"""
        return [n for n in self.nodes.values() if n.can_expand()]

    def get_best_node(self) -> Optional[TreeNode]:
        """最高スコアのノードを取得"""
        valid = [n for n in self.nodes.values() if not n.is_pruned]
        return max(valid, key=lambda n: n.score) if valid else None

    def prune(self, keep_top: int = 20) -> None:
        """低スコアノードを枝刈り"""
        sorted_nodes = sorted(
            self.nodes.values(), key=lambda n: n.score, reverse=True
        )
        for node in sorted_nodes[keep_top:]:
            node.is_pruned = True

def best_first_search(
    topic: str,
    expand_fn: Callable[[TreeNode], list[ArticleState]],  # LLMで候補生成
    evaluate_fn: Callable[[ArticleState], float],          # LLMで品質評価
    max_iterations: int = 10,
    beam_width: int = 3,
    prune_keep_top: int = 20,
) -> ArticleState:
    """Best-First Tree Searchのコアループ"""
    tree = SearchTree()
    root = TreeNode.create_root(topic)
    tree.add_node(root)
    root.score = evaluate_fn(root.article_state)

    for iteration in range(max_iterations):
        # 1. 展開可能なノードを取得し、スコア順でソート
        expandable = tree.get_expandable_nodes()
        if not expandable:
            break

        # 2. 上位beam_width個のノードを選択（Best-First）
        best_to_expand = sorted(
            expandable, key=lambda n: n.score, reverse=True
        )[:beam_width]

        # 3. 各ノードを展開して子ノードを生成
        for node in best_to_expand:
            child_states = expand_fn(node)
            for state in child_states:
                child = TreeNode(
                    id=str(uuid.uuid4())[:8],
                    stage=_next_stage(node.stage),
                    article_state=state,
                    depth=node.depth + 1,
                    parent_id=node.id,
                )
                child.score = evaluate_fn(state)
                tree.add_node(child)
                node.children.append(child.id)

        # 4. 低スコアノードを枝刈り
        tree.prune(keep_top=prune_keep_top)

    best = tree.get_best_node()
    return best.article_state if best else root.article_state

def _next_stage(current: SearchStage) -> SearchStage:
    """次のステージを返す"""
    stages = list(SearchStage)
    idx = stages.index(current)
    return stages[min(idx + 1, len(stages) - 1)]
```

このコードの要点：
- **`SearchStage`**: 記事生成の4段階（INITIAL→EXPANDED→ENHANCED→POLISHED）
- **`ArticleState`**: アイデア→アウトライン→ドラフトの3層構造
- **`SearchTree`**: IDベースでノードを管理、枝刈り機能を内蔵
- **Best-First選択**: スコア順にソートして上位`beam_width`個を展開

## 並列評価システム - レート制限付きWorkerPool

LLM APIの呼び出しはI/Oバウンドなので、`asyncio`による非同期処理が効果的です。本実装では、並行数制限に加えて**レート制限**（1分あたりのAPI呼び出し数制限）も組み込んだWorkerPoolを使用しています。

```python
import asyncio
import time
from typing import Any, Coroutine
import anthropic

class WorkerPool:
    """レート制限付きの並列タスク実行"""

    def __init__(self, max_workers: int = 3, rate_limit_per_minute: int = 20):
        self.semaphore = asyncio.Semaphore(max_workers)
        self.rate_limit = rate_limit_per_minute
        self._lock = asyncio.Lock()
        self._call_count = 0
        self._window_start = time.time()

    async def _apply_rate_limit(self) -> None:
        """レート制限を適用"""
        async with self._lock:
            now = time.time()
            if now - self._window_start >= 60:
                self._call_count = 0
                self._window_start = now

            if self._call_count >= self.rate_limit:
                wait_time = 60 - (now - self._window_start)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self._call_count = 0
                self._window_start = time.time()

            self._call_count += 1

    async def execute(self, tasks: list[Coroutine[Any, Any, Any]]) -> list[Any]:
        """タスクを並列実行"""
        async def run_with_limits(task: Coroutine) -> Any:
            await self._apply_rate_limit()
            async with self.semaphore:
                return await task

        return await asyncio.gather(
            *[run_with_limits(t) for t in tasks],
            return_exceptions=True
        )

# 使用例
async def generate_variants(prompts: list[str], client: anthropic.AsyncAnthropic) -> list[str]:
    pool = WorkerPool(max_workers=3, rate_limit_per_minute=20)

    async def call_llm(prompt: str) -> str:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    tasks = [call_llm(p) for p in prompts]
    return await pool.execute(tasks)
```

この実装のポイント：
- **`asyncio.Semaphore`**: 同時実行数を制限（`max_workers`）
- **レート制限**: 1分間のAPI呼び出し数を制限（`rate_limit_per_minute`）
- **`return_exceptions=True`**: 一部のタスクが失敗しても全体を継続

## 品質評価器 - Pydanticによる堅牢な評価設計

探索的生成の核心は「どのように候補を評価するか」です。本実装では、Pydanticを使用してスキーマを定義し、LLMの出力を確実にバリデーションします。評価軸は4つ（構造性・正確性・可読性・実用性）で、加重平均によって総合スコアを算出します。

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
import anthropic

class EvaluationOutput(BaseModel):
    """LLM出力のスキーマ定義"""
    structure: float = Field(ge=1, le=10, description="構造性スコア")
    accuracy: float = Field(ge=1, le=10, description="正確性スコア")
    readability: float = Field(ge=1, le=10, description="可読性スコア")
    practicality: float = Field(ge=1, le=10, description="実用性スコア")
    feedback: str = Field(description="改善フィードバック")

@dataclass
class EvaluationResult:
    """評価結果（加重平均スコア付き）"""
    structure_score: float
    accuracy_score: float
    readability_score: float
    practicality_score: float
    feedback: str
    overall_score: float = 0.0

    def __post_init__(self):
        # 加重平均: 構造20%, 正確性30%, 可読性25%, 実用性25%
        self.overall_score = (
            self.structure_score * 0.20 +
            self.accuracy_score * 0.30 +
            self.readability_score * 0.25 +
            self.practicality_score * 0.25
        )

EVALUATION_PROMPT = """
以下の記事を4つの軸で評価し、JSON形式で回答してください。

## 評価基準
1. structure (1-10): 見出し構成、論理的な流れ、セクション間の整合性
2. accuracy (1-10): 技術情報の正確性、コードの妥当性、最新性
3. readability (1-10): 文章の読みやすさ、説明の明確さ、専門用語の適切な解説
4. practicality (1-10): 実務での活用しやすさ、具体例の充実度

## 評価対象
{content}

## 出力形式（JSONのみ）
{{"structure": <数値>, "accuracy": <数値>, "readability": <数値>, "practicality": <数値>, "feedback": "<改善点>"}}
"""

class ArticleEvaluator:
    """記事品質の評価器"""

    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    async def evaluate(self, content: str) -> EvaluationResult:
        """記事を評価してスコアを返す"""
        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": EVALUATION_PROMPT.format(content=content)
            }],
            temperature=0.1,  # 評価の一貫性を保つため低温度
        )

        # Pydanticでバリデーション
        raw_text = response.content[0].text
        validated = EvaluationOutput.model_validate_json(raw_text)

        return EvaluationResult(
            structure_score=validated.structure,
            accuracy_score=validated.accuracy,
            readability_score=validated.readability,
            practicality_score=validated.practicality,
            feedback=validated.feedback,
        )
```

評価のポイント：
- **Pydanticスキーマ**: `model_validate_json()`で出力を確実にパース・検証
- **4軸評価**: 構造性・正確性・可読性・実用性の多角的な品質判断
- **加重平均**: 正確性を重視（30%）した総合スコア算出
- **フィードバック**: 次の改善イテレーションに活用可能

## 検証における評価設計

探索的生成の効果を検証する際は、単発生成との比較実験を設計します。以下の指標で評価することを推奨します。

| 指標 | 観点 |
|-----|------|
| 平均スコア・中央値 | 品質の中央傾向 |
| 標準偏差・最小値 | 品質の安定性（ばらつきと下限） |
| API呼び出し数 | コスト効率 |

**注意点**: LLMによる品質評価は主観的な側面があります。また、効果はタスクの性質（構造化度、創造性の要求度など）により大きく異なる可能性があります。

## トレードオフ

探索的生成には以下のトレードオフが存在します。

**期待されるメリット:**
- 複数候補から選択することで、極端に低品質な出力を回避できる
- 評価・選択プロセスにより、出力品質のばらつきが削減される
- 探索深度を増やすほど、より良い候補が見つかる可能性が高まる

**予測されるコスト:**
- 探索深度・幅に応じて、単発生成の数倍〜数十倍のAPI呼び出しが発生
- 状態管理・エラーハンドリングの実装複雑性が増加
- 深度を増やしても品質向上が頭打ちになるポイントが存在する

**検証時の推奨アプローチ:**
- まず小規模な探索（depth=2-3、beam_width=3）から開始
- 品質向上とコスト増加の比率を計測し、最適な深度を特定
- 評価関数を段階的に洗練させる

## まとめ

本記事では、AI Scientist-v2の**科学実験探索アーキテクチャ**を記事生成タスクに応用する手法を解説しました。本家AI Scientist-v2ではTree Searchを実験コードの探索に使用していますが、その本質である「候補展開→評価→選択→剪定」のサイクルは、テキスト生成にも有効に適用できます。

探索的生成の主な利点は、**品質の安定化**（ばらつきの削減と最低品質の底上げ）にあります。複数候補から評価に基づいて選択することで、単発生成で発生しうる極端に低品質な出力を回避できます。ただし、探索深度・幅に応じてAPI呼び出し数が増加するため、品質向上とコストのトレードオフを自身の環境で検証することを推奨します。

実装を始める際は、まずは小規模な探索（depth=2-3、beam_width=3）から着手し、評価関数を段階的に洗練させることをお勧めします。AI Scientist-v2の原論文や、関連研究であるTree of Thoughts、Self-Refineなども参考になるでしょう。

**参考リンク:**
- [AI Scientist-v2 GitHub](https://github.com/SakanaAI/AI-Scientist-v2)
- [AI Scientist 論文 (arXiv)](https://arxiv.org/abs/2408.06292)
