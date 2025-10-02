# Docker クイックスタートガイド

このガイドは、ローカル環境またはDocker対応サーバーでの実行方法を説明します。

**さくらインターネット 高火力Dokでの実行については [SAKURA_DOK.md](./SAKURA_DOK.md) を参照してください。**

## 最小限のセットアップ

### 1. 環境変数の設定

```bash
cp .env.example .env
# .envファイルを編集して以下の値を設定:
# DATABASE_URL, STORAGE_URL, BLOB_READ_WRITE_TOKEN
```

### 2. コンテナの起動

```bash
docker-compose up -d
```

### 3. ログの確認

```bash
docker-compose logs -f leaps-worker
```

## タスクの作成方法

### SQLでタスクを挿入

```sql
INSERT INTO tasks (project_id, status)
VALUES ('my-project-id', 'pending');
```

### 必要なファイルをアップロード

Vercel Blob Storage に以下をアップロード:
- `{project_id}/input.csv`
- `{project_id}/config.json`

## 結果の確認

タスクが完了すると:
- Vercel Blob Storage: `{project_id}/result.csv`
- タスクのステータスが `succeeded` に更新

## トラブルシューティング

```bash
# コンテナの状態確認
docker-compose ps

# コンテナ内でシェルを起動
docker exec -it leaps-worker /bin/bash

# ログの確認
docker-compose logs -f

# コンテナの再起動
docker-compose restart

# コンテナの停止と削除
docker-compose down
```

詳細は [DOCKER.md](./DOCKER.md) を参照してください。
