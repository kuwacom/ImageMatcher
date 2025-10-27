# ImageMatcher

ImageMatcher は、SIFT (Scale-Invariant Feature Transform) 特徴量と平均色情報を組み合わせて画像の類似性を比較し、類似画像を見つける Python ベースの画像類似検索アプリケーションです。  
2つの画像を直接比較するコマンドラインツールと、事前に構築された特徴量データベースから類似画像を検索する REST API の両方を提供します。

## 機能

- **SIFT 特徴量抽出**: OpenCV の SIFT アルゴリズムを使用して画像から堅牢な特徴量を抽出します。
- **カラー類似度計算**: 画像の平均色に基づいてカラー類似度を計算します。
- **総合類似度**: SIFT 類似度とカラー類似度を重み付けして総合類似度を算出します（SIFT 70%、カラー 30%）。
- **画像類似性比較**: 2 つの画像を比較し、マッチした特徴量とカラー情報に基づいて類似性パーセンテージを計算します。
- **特徴量データベース構築**: 画像ディレクトリから画像の特徴量とカラー情報を含むデータベースを構築します。
- **REST API**: FastAPI ベースの REST API で、画像をアップロードしてデータベース内の類似画像を検索します。
- **パフォーマンス指標**: API には特徴量抽出、マッチング、ソート操作のタイミング情報が含まれます。

## 要件

- Python 3.10 以上
- `pyproject.toml` に記載された依存関係:
  - fastapi >= 0.120.0
  - opencv-contrib-python >= 4.12.0.88
  - python-multipart >= 0.0.20
  - uvicorn >= 0.38.0

## インストール

1. リポジトリをクローンします:
   ```bash
   git clone <repository-url>
   cd imagematcher
   ```

2. uv を使用して依存関係をインストールします (推奨):
   ```bash
   uv sync
   ```

   または pip を使用:
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 特徴量データベースの構築

検索 API を使用する前に、画像から特徴量データベースを構築する必要があります:

```bash
python buildFeatures.py ./images
```

これにより、`./images` ディレクトリ内のすべての `.jpg`、`.png`、`.jpeg`、`.bmp` ファイルを処理し、特徴量と平均色情報を `features.pkl` に保存します。

### コマンドライン画像比較

2 つの画像を直接比較するには:

```bash
python main.py path/to/image1.jpg path/to/image2.jpg
```

これにより、マッチした特徴量の数、SIFT 類似度、カラー類似度、総合類似度が表示され、マッチしたキーポイントの視覚比較が表示されます。

### Web API

FastAPI サーバーを起動します:

```bash
python app.py
```

またはカスタムポートを指定:

```bash
python app.py 8080
```

#### API エンドポイント

FastAPI で実装をしているため、http://localhost:8000/docs にアクセスすれば、OpenAPI の確認ができます。

**POST /search**

データベース内の類似画像を検索するために画像ファイルをアップロードします。

- **リクエスト**: `file` という名前のファイルアップロードを含むマルチパートフォームデータ
- **レスポンス**: 類似性スコアとタイミング情報を含む上位 5 つの類似画像の JSON

curl を使用した例:

```bash
curl -X POST "http://localhost:8000/search" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

レスポンス形式:

```json
{
  "results": [
    {
      "file": "category1-image1.jpg",
      "tag": "category1",
      "matchCount": 45,
      "siftSimilarity": 85.2,
      "colorSimilarity": 92.5,
      "totalSimilarity": 87.8,
      "matchingTime_ms": 12.5
    },
    ...
  ],
  "timings_ms": {
    "featureExtraction_ms": 150.3,
    "matchingTotal_ms": 234.7,
    "sorting_ms": 0.1,
    "total_ms": 385.1
  }
}
```

## プロジェクト構造

- `app.py`: 画像検索用の FastAPI Web アプリケーション
- `main.py`: 直接画像比較用のコマンドラインツール
- `buildFeatures.py`: 画像から特徴量データベースを構築するスクリプト
- `siftFeatures.py`: SIFT 特徴量抽出、マッチング、類似度計算、カラー類似度計算、データベース保存/読み込みのユーティリティ関数
- `pyproject.toml`: プロジェクト構成と依存関係
- `README.md`: このファイル
- `images/`: 入力画像用のディレクトリ (git で無視)
- `features.pkl`: 生成された特徴量データベース (git で無視)

## 仕組み

1. **特徴量抽出**: SIFT が画像内のキーポイントを検出し、デスクリプタを計算します。同時に平均色を抽出します。
2. **マッチング**: Brute-Force マッチャーと Lowe の比率テストを使用して、デスクリプタ間の良好なマッチを見つけます。
3. **類似性計算**: SIFT 類似性は、2 つの画像のキーポイント数の平均に対する良好なマッチの比率として計算されます。カラー類似性は平均色の距離に基づいて計算され、SIFT とカラー類似度を重み付けして総合類似度を算出します。
4. **データベース検索**: API 検索の場合、クエリ画像の特徴量と平均色を事前に構築されたデータベース内のすべての画像と比較します。
