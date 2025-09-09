# LEAPS

![version](https://img.shields.io/badge/version-2.0.0-red.svg)
![stars](https://img.shields.io/github/stars/igem-tsukuba/leaps?color=yellow)
![commit-activity](https://img.shields.io/github/commit-activity/t/igem-tsukuba/leaps)
![license](https://img.shields.io/badge/license-MIT-green)

LEAPSは、入力された配列から多様な配列を生成し、その機能を予測する過程を繰り返すことで、広大な配列空間を効率的に探索することを可能にした、タンパク質配列設計のためのフレームワークです。

<br/>
<br/>

## 🚀 Features

> [!CAUTION]
> 本プロジェクトは進行中のため、このセクションは随時更新されます。

<br/>
<br/>

## 📚 Background

> [!CAUTION]
> 本プロジェクトは進行中のため、このセクションは随時更新されます。

<br/>
<br/>

## 🚀　Usage

1. リポジトリをクローンする

```bash
git clone https://github.com/igem-tsukuba/leaps.git
```

<br/>

2. リポジトリに移動する

```bash
cd leaps
```

<br/>

3. 仮装環境を作成する

```bash
python -m venv .venv
source .venv/bin/activate
```

<br/>

3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

<br/>

> [!NOTE]
> `bin`に[foldseek](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view?usp=sharing)を配置してください。

<br/>
<br/>

## ⚡️ Quick Start

```bash
$ python main.py
```

<br/>
<br/>

## 📂 Structure

```
leaps/
├── bin/
│   └── foldseek
├── notebooks/                # ノートブック
│   └── example.ipynb
├── runs/                     # ログ
├── src/
│   ├── config/               # 設定を管理するクラス
│   ├── early_stopper/        # 早期終了を行うクラス
│   ├── evaluator/            # タンパク質を評価するクラス
│   ├── generator/            # タンパク質を生成するクラス
│   ├── predictor/            # タンパク質を予測するクラス
│   ├── runner/               # 実行を管理するクラス
│   └── sampler/              # サンプリングするクラス
├── .gitattributes
├── .gitignore
├── .python-version
├── config.yaml               # 設定ファイル 
├── CONTRIBUTING.md
├── LICENSE     
├── main.py                   # メイン関数
├── pyproject.toml
├── README.md                 # 本ファイル
├── requirements.txt          # 依存関係
└── uv.lock
```

<br/>
<br/>

## 🤝 Contributer

<a href="https://github.com/yushin-ito">
  <img  src="https://avatars.githubusercontent.com/u/75526539?s=48&v=4" width="64px">
</a>

<br/>
<br/>

## 📜 LICENSE

[MIT LICENSE](LICENSE)
