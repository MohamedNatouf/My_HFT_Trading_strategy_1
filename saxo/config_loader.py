import json
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, path: str = "config/saxo_config.json"):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        with self.path.open() as f:
            return json.load(f)

    def generate_fix_cfg(self, cfg_path: str = "config/saxo_fix.cfg") -> str:
        cfg = self.load()["fix"]
        lines = [
            "[DEFAULT]",
            f"ConnectionType=initiator",
            f"HeartBtInt=30",
            f"ReconnectInterval=10",
            f"BeginString=FIX.4.4",
            f"UseDataDictionary=Y",
            f"DataDictionary={cfg.get('dictionary','FIX44')}",
            f"SocketConnectHost={cfg['host']}",
            f"SocketConnectPort={cfg['port']}",
            "FileStorePath=store",
            "FileLogPath=log",
            "ResetOnLogon=Y",
            "ResetOnLogout=Y",
            "ResetOnDisconnect=Y",
            "ValidateUserDefinedFields=N",
            "PersistMessages=Y",
            "[SESSION]",
            "StartTime=00:00:00",
            "EndTime=23:59:59",
            f"SenderCompID={cfg['senderCompId']}",
            f"TargetCompID={cfg['targetCompId']}"
        ]
        content = "\n".join(lines)
        Path(cfg_path).write_text(content)
        return content
