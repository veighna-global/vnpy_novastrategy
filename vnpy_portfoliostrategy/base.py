from enum import Enum


APP_NAME = "NovaStrategy"


class EngineType(Enum):
    LIVE = "实盘"
    BACKTESTING = "回测"


EVENT_NOVA_LOG = "eNovaLog"
EVENT_NOVA_STRATEGY = "eNovaStrategy"
