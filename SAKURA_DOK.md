# さくらインターネット 高火力Dokでの実行方法

このドキュメントは、LEAPSをさくらインターネットの高火力Dokサービスで実行するためのガイドです。

## 概要

高火力Dokは、さくらインターネットが提供するGPUコンテナサービスです。このサービスを使用してLEAPSを実行できます。

公式ドキュメント: https://manual.sakura.ad.jp/koukaryoku-dok-api/spec.html

## 前提条件

- さくらインターネットのアカウント
- 高火力Dokの利用申し込みが完了していること
- API トークンを取得していること

## Dockerイメージのビルド

### 1. イメージのビルド

```bash
# リポジトリのルートディレクトリで実行
docker build -t leaps-worker:latest .
```

### 2. イメージのタグ付け

高火力Dokで使用するために、適切なタグを付けます：

```bash
# さくらのコンテナレジストリを使用する場合
docker tag leaps-worker:latest [YOUR_REGISTRY]/leaps-worker:latest
```

### 3. イメージのプッシュ

```bash
# コンテナレジストリにプッシュ
docker push [YOUR_REGISTRY]/leaps-worker:latest
```

## 高火力Dokでの実行

### APIを使用したコンテナの起動

高火力DokのAPIを使用してコンテナを起動します：

```bash
curl -X POST https://api.koukaryoku-dok.sakura.ad.jp/v1/containers \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "leaps-worker",
    "image": "[YOUR_REGISTRY]/leaps-worker:latest",
    "gpu": {
      "type": "A100",
      "count": 1
    },
    "env": {
      "DATABASE_URL": "postgresql://user:password@host:port/database",
      "STORAGE_URL": "https://your-blob-storage.public.blob.vercel-storage.com",
      "BLOB_READ_WRITE_TOKEN": "vercel_blob_rw_XXXXXXXXXXXXX",
      "USE_TORCH": "ON"
    },
    "resources": {
      "cpu": "8",
      "memory": "32Gi"
    }
  }'
```

### 環境変数の設定

以下の環境変数を設定する必要があります：

| 変数名 | 説明 | 例 |
|--------|------|-----|
| `DATABASE_URL` | PostgreSQL接続文字列 | `postgresql://user:pass@host:5432/db` |
| `STORAGE_URL` | Vercel Blob StorageのベースURL | `https://xxxxx.public.blob.vercel-storage.com` |
| `BLOB_READ_WRITE_TOKEN` | Vercel Blobアクセストークン | `vercel_blob_rw_XXXXX` |
| `USE_TORCH` | PyTorch使用フラグ | `ON` |

## コンテナの管理

### ステータスの確認

```bash
curl -X GET https://api.koukaryoku-dok.sakura.ad.jp/v1/containers/leaps-worker \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

### ログの確認

```bash
curl -X GET https://api.koukaryoku-dok.sakura.ad.jp/v1/containers/leaps-worker/logs \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

### コンテナの停止

```bash
curl -X DELETE https://api.koukaryoku-dok.sakura.ad.jp/v1/containers/leaps-worker \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

## リソース設定

### 推奨スペック

- **GPU**: NVIDIA A100 または V100 (1基以上)
- **CPU**: 8コア以上
- **メモリ**: 32GB以上
- **ストレージ**: 100GB以上

### GPU タイプ

高火力Dokで利用可能なGPUタイプ：

- `A100`: NVIDIA A100 (40GB/80GB)
- `V100`: NVIDIA V100 (16GB/32GB)
- `T4`: NVIDIA T4 (16GB)

## トラブルシューティング

### コンテナが起動しない

1. イメージが正しくプッシュされているか確認
2. 環境変数が正しく設定されているか確認
3. GPUリソースが利用可能か確認

### メモリ不足エラー

`config.json`の`batch_size`を小さくしてください：

```json
{
  "predictor": {
    "ex": {
      "batch_size": 8
    }
  },
  "evaluator": {
    "likelihood": {
      "batch_size": 16
    }
  }
}
```

### データベース接続エラー

- データベースのホスト名とポートが正しいか確認
- ネットワーク接続が許可されているか確認
- 認証情報が正しいか確認

## スケーリング

複数のワーカーを並列実行する場合：

```bash
# ワーカー1
curl -X POST https://api.koukaryoku-dok.sakura.ad.jp/v1/containers \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "leaps-worker-1", ...}'

# ワーカー2
curl -X POST https://api.koukaryoku-dok.sakura.ad.jp/v1/containers \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "leaps-worker-2", ...}'
```

### 並行実行の安全性

複数のワーカーが同時に動作する場合でも、以下の仕組みで安全に処理されます：

1. **データベースロック**: `FOR UPDATE SKIP LOCKED` により、同一タスクが複数のワーカーで処理されることを防止
2. **一意な出力ディレクトリ**: 各タスクは `{project_id}_{task_id}` という一意のディレクトリで実行
3. **一意なファイル名**: 結果ファイルは `{project_id}/result_{task_id}.csv` として保存され、上書きを防止

これにより、同じ `project_id` を持つ複数のタスクでも、互いに干渉することなく並列実行できます。

## コスト管理

- 使用していないコンテナは停止する
- バッチサイズを調整してGPU使用率を最適化
- タスクキューが空の場合はコンテナをスケールダウン

## セキュリティ

- API トークンは安全に管理する
- データベース接続は暗号化する (SSL/TLS)
- 環境変数に機密情報を直接含めない場合は、シークレット管理サービスを使用

## 参考リンク

- [高火力Dok API仕様書](https://manual.sakura.ad.jp/koukaryoku-dok-api/spec.html)
- [さくらのクラウド コントロールパネル](https://secure.sakura.ad.jp/cloud/)
- [LEAPSリポジトリ](https://github.com/Tomoro0726/leaps)
