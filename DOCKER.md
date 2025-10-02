# Docker Setup for LEAPS

このドキュメントは、LEAPSをDockerコンテナで実行するためのガイドです。

## 概要

Dockerコンテナは以下のワークフローを実行します：

1. データベースから `status=pending` のタスクを取得
2. Vercel Blob から `config.json` と `input.csv` をダウンロード
3. `main.py` を実行してタンパク質配列を生成
4. 結果の `result.csv` を Vercel Blob にアップロード
5. タスクのステータスを `succeeded` に更新

## 前提条件

- Docker と Docker Compose がインストールされていること
- NVIDIA GPU を使用する場合、NVIDIA Container Toolkit がインストールされていること
- PostgreSQL データベースが稼働していること
- Vercel Blob Storage のアクセストークンを持っていること

## セットアップ

### 1. 環境変数の設定

`.env.example` をコピーして `.env` ファイルを作成し、必要な値を設定してください：

```bash
cp .env.example .env
```

`.env` ファイルを編集して以下の値を設定：

```env
DATABASE_URL=postgresql://user:password@host:port/database
STORAGE_URL=https://your-blob-storage.public.blob.vercel-storage.com
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_XXXXXXXXXXXXX
```

### 2. データベーステーブルの作成

PostgreSQL データベースに以下のテーブルを作成してください：

```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_created_at ON tasks(created_at);
```

## 実行方法

### Docker Compose を使用する場合（推奨）

```bash
# コンテナをビルドして起動
docker-compose up -d

# ログを確認
docker-compose logs -f leaps-worker

# コンテナを停止
docker-compose down
```

### Docker コマンドを直接使用する場合

```bash
# イメージをビルド
docker build -t leaps-worker .

# コンテナを実行
docker run -d \
  --name leaps-worker \
  --gpus all \
  -e DATABASE_URL="postgresql://user:password@host:port/database" \
  -e STORAGE_URL="https://your-blob-storage.public.blob.vercel-storage.com" \
  -e BLOB_READ_WRITE_TOKEN="vercel_blob_rw_XXXXXXXXXXXXX" \
  -v $(pwd)/runs:/app/runs \
  leaps-worker

# ログを確認
docker logs -f leaps-worker

# コンテナを停止
docker stop leaps-worker
docker rm leaps-worker
```

## タスクの作成

タスクを作成するには、データベースに新しいレコードを挿入します：

```sql
INSERT INTO tasks (project_id, status)
VALUES ('my-project-id', 'pending');
```

また、Vercel Blob Storage に以下のファイルをアップロードしてください：

- `{project_id}/input.csv` - 入力データ
- `{project_id}/config.json` - 設定ファイル

### config.json の例

```json
{
  "project": "my-project-id",
  "debug": true,
  "device": "cuda",
  "seed": 42,
  "sampler": {
    "num_shuffles": 100000,
    "shuffle_rate": 0.04,
    "window_sizes": [1, 3, 5]
  },
  "predictor": {
    "ex": {
      "batch_size": 16,
      "model_name_or_path": "facebook/esm2_t30_150M_UR50D",
      "num_epochs": 200
    }
  },
  "evaluator": {
    "hamiltonian": {
      "threshold": 5.0,
      "mode": "min"
    }
  },
  "generator": {
    "batch_size": 32,
    "model_name_or_path": "hugohrban/progen2-small",
    "num_epochs": 6
  },
  "early_stopper": {
    "batch_size": 32,
    "model_name_or_path": "facebook/esm2_t33_650M_UR50D",
    "num_samples": 1000,
    "patience": 5
  },
  "runner": {
    "num_iterations": 30,
    "num_sequences": 30000
  }
}
```

## トラブルシューティング

### GPU が認識されない

NVIDIA Container Toolkit がインストールされているか確認してください：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### データベース接続エラー

- `DATABASE_URL` が正しく設定されているか確認
- データベースが起動しているか確認
- ネットワーク接続を確認（コンテナからデータベースホストにアクセスできるか）

### Vercel Blob Storage エラー

- `STORAGE_URL` と `BLOB_READ_WRITE_TOKEN` が正しく設定されているか確認
- トークンに読み書き権限があるか確認
- ファイルが正しいパスにアップロードされているか確認

## ログとデバッグ

コンテナ内でシェルを起動してデバッグ：

```bash
docker exec -it leaps-worker /bin/bash
```

特定のコマンドを実行：

```bash
docker exec leaps-worker ls -la /app/runs
docker exec leaps-worker python -c "import torch; print(torch.cuda.is_available())"
```

## スケーリング

複数のワーカーを起動して並列処理：

```bash
docker-compose up -d --scale leaps-worker=3
```

### 並行実行の安全性

複数のワーカーが同時に動作する場合でも、以下の仕組みで安全に処理されます：

1. **データベースロック**: `FOR UPDATE SKIP LOCKED` により、同一タスクが複数のワーカーで処理されることを防止
2. **一意な出力ディレクトリ**: 各タスクは `{project_id}_{task_id}` という一意のディレクトリで実行
3. **一意なファイル名**: 結果ファイルは `{project_id}/result_{task_id}.csv` として保存され、上書きを防止

これにより、同じ `project_id` を持つ複数のタスクでも、互いに干渉することなく並列実行できます。

## 注意事項

- GPU メモリが不足する場合は、`config.json` の `batch_size` を小さくしてください
- 大規模なタスクを実行する場合は、十分なディスク容量を確保してください
- 定期的に `runs` ディレクトリをクリーンアップしてください
