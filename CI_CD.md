# CI/CD Setup for LEAPS

このドキュメントでは、LEAPSのCI/CDパイプラインについて説明します。

## 概要

GitHub Actionsを使用して、`release`ブランチへのプッシュまたはバージョンタグの作成時に、DockerイメージをDocker Hubに自動的にビルド・プッシュします。

## ワークフロー

### 自動ビルド・プッシュ

`.github/workflows/docker-publish.yaml` により、以下のトリガーで自動的にDockerイメージがビルド・プッシュされます：

1. **`release`ブランチへのプッシュ**
   ```bash
   git push origin release
   ```

2. **バージョンタグの作成**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

### ビルドされるイメージ

2つのDockerイメージが自動的にビルドされます：

1. **標準版** (`Dockerfile`)
   - イメージ名: `{username}/leaps:latest`
   - 用途: 一般的なDocker環境、Docker Compose

2. **Sakura Dok最適化版** (`Dockerfile.sakura`)
   - イメージ名: `{username}/leaps-sakura:latest`
   - 用途: さくらインターネット 高火力Dok

## セットアップ

### 1. Docker Hub アカウントの準備

Docker Hubアカウントを作成し、アクセストークンを生成します：

1. [Docker Hub](https://hub.docker.com/)にログイン
2. Account Settings → Security → New Access Token
3. トークン名を入力（例: `github-actions-leaps`）
4. 権限を選択（`Read, Write, Delete`推奨）
5. トークンをコピー

### 2. GitHubシークレットの設定

GitHubリポジトリにDocker Hub認証情報を設定します：

1. GitHubリポジトリの Settings → Secrets and variables → Actions
2. 以下のシークレットを追加：

| シークレット名 | 説明 | 例 |
|---------------|------|-----|
| `DOCKERHUB_USERNAME` | Docker Hubユーザー名 | `your-username` |
| `DOCKERHUB_TOKEN` | Docker Hubアクセストークン | `dckr_pat_xxxxx...` |

### 3. リポジトリ名の設定

デフォルトでは、GitHubリポジトリ名（`{owner}/{repo}`）がDockerイメージ名として使用されます。

カスタム名を使用したい場合は、ワークフローファイル (`.github/workflows/docker-publish.yaml`) の`IMAGE_NAME`を変更してください：

```yaml
env:
  REGISTRY: docker.io
  IMAGE_NAME: your-dockerhub-username/leaps  # カスタマイズ
```

## 使用方法

### リリースフロー

1. **開発ブランチで作業**
   ```bash
   git checkout -b feature/new-feature
   # 開発作業...
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```

2. **Pull Requestの作成とマージ**
   - PRを作成し、レビュー後に`main`ブランチにマージ

3. **releaseブランチへのマージ**
   ```bash
   git checkout release
   git merge main
   git push origin release
   ```
   → 自動的にDockerイメージがビルド・プッシュされます

### バージョンタグによるリリース

セマンティックバージョニングを使用したリリース：

```bash
# バージョンタグを作成
git checkout release
git tag v1.0.0
git push origin v1.0.0
```

自動的に以下のタグでイメージがプッシュされます：
- `your-username/leaps:v1.0.0`
- `your-username/leaps:1.0`
- `your-username/leaps:1`
- `your-username/leaps:latest`

### イメージの使用

ビルドされたイメージを使用する：

```bash
# 標準版
docker pull your-username/leaps:latest
docker run --gpus all \
  -e DATABASE_URL="..." \
  -e STORAGE_URL="..." \
  -e BLOB_READ_WRITE_TOKEN="..." \
  your-username/leaps:latest

# Sakura Dok版
docker pull your-username/leaps-sakura:latest
```

## ワークフローの詳細

### トリガー条件

```yaml
on:
  push:
    branches:
      - release        # releaseブランチへのプッシュ
    tags:
      - 'v*.*.*'       # v1.0.0形式のタグ
```

### ビルドマトリクス

2つのDockerfileを並列ビルド：

| Dockerfile | イメージサフィックス | プラットフォーム |
|------------|---------------------|-----------------|
| `Dockerfile` | なし | `linux/amd64` |
| `Dockerfile.sakura` | `-sakura` | `linux/amd64` |

### タグ戦略

| トリガー | 生成されるタグ |
|---------|---------------|
| `release`ブランチ | `release` |
| タグ`v1.2.3` | `v1.2.3`, `1.2`, `1`, `latest` |

### キャッシュ

GitHub Actionsキャッシュを使用してビルド時間を短縮：
- 依存関係のレイヤーをキャッシュ
- 2回目以降のビルドが高速化

## トラブルシューティング

### ビルドが失敗する

1. **Docker Hub認証エラー**
   - `DOCKERHUB_USERNAME`と`DOCKERHUB_TOKEN`が正しく設定されているか確認
   - トークンの権限が十分か確認

2. **Dockerfileのエラー**
   - ローカルで`docker build`が成功するか確認
   - 依存関係が正しくインストールされるか確認

3. **プッシュ権限エラー**
   - Docker Hubリポジトリが存在するか確認
   - リポジトリが公開設定になっているか確認

### ワークフローログの確認

GitHubリポジトリの Actions タブでワークフローの実行状況とログを確認できます：

1. リポジトリ → Actions → "Build and Push Docker Images"
2. 失敗したワークフローをクリック
3. 各ステップのログを確認

### ローカルテスト

ワークフローをローカルでテストする（[act](https://github.com/nektos/act)使用）：

```bash
# actのインストール
brew install act  # macOS
# または
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# ワークフローのテスト
act push -s DOCKERHUB_USERNAME=your-username -s DOCKERHUB_TOKEN=your-token
```

## ベストプラクティス

1. **セキュリティ**
   - Docker Hubトークンは定期的にローテーション
   - 最小権限の原則に従う
   - シークレットをコードにコミットしない

2. **バージョニング**
   - セマンティックバージョニング（v1.0.0形式）を使用
   - 破壊的変更はメジャーバージョンを上げる
   - 後方互換性のある変更はマイナーバージョンを上げる

3. **タグ管理**
   - `latest`タグは常に最新の安定版を指す
   - 特定バージョンのタグは変更しない（イミュータブル）

4. **テスト**
   - `release`ブランチにマージする前に十分なテストを実施
   - PRでコードレビューを行う

## 関連リンク

- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Docker Hub](https://hub.docker.com/)
- [Semantic Versioning](https://semver.org/)
