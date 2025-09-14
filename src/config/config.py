import shutil
from pathlib import Path
from typing import Any, Dict

import yaml
from easydict import EasyDict as edict


class Config(edict):
    """
    設定を管理するクラス
    """

    def __init__(self, config_path: str | Path = "config.yaml") -> None:  # noqa: D401
        """
        Args:
            path (str | Path): 設定ファイルのパス
        """
        config_path = Path(config_path)

        with config_path.open("r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        project_dir: Path = Path("runs") / config["project"]
        project_dir.mkdir(parents=True, exist_ok=True)

        src = config_path
        dst = project_dir / "config.yaml"
        shutil.copy2(src, dst)

        self._inject(config, key="project", value=config["project"])
        self._inject(config, key="debug", value=config["debug"])
        self._inject(config, key="device", value=config["device"])
        self._inject(config, key="seed", value=config["seed"])

        super().__init__(config)

    def _inject(self, node: Any, *, key: str, value: Any) -> None:
        """
        Args:
            node (Any): ノード
            key (str): キー
            value (Any): 値
        """
        if isinstance(node, dict):
            node.setdefault(key, value)
            for v in node.values():
                self._inject(v, key=key, value=value)
        elif isinstance(node, list):
            for item in node:
                self._inject(item, key=key, value=value)
