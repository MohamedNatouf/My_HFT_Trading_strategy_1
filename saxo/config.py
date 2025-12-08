from dataclasses import dataclass

@dataclass
class SaxoConfig:
    fix_host: str
    fix_port: int
    fix_sender_comp_id: str
    fix_target_comp_id: str
    fix_username: str
    fix_password: str
    websocket_url: str
    websocket_token: str

    venue: str = "SAXO"
