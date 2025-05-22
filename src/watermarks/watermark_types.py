from strenum import StrEnum
from src.watermarks.scheme_config import (
    KGWWatermarkConfiguration,
    KGWWatermarkTokenConfiguration,
    KGWWatermarkTokenBoundaryConfiguration
)


class WatermarkType(StrEnum):
    KGW = "kgw"
    KGW_WMTOKEN = "kgw_wmtoken"
    KGW_WMTOKEN_BOUNDARY = "kgw_wmtoken_boundary"

    def get_config(self):
        if self.value == "kgw":
            return KGWWatermarkConfiguration
        elif self.value == "kgw_wmtoken":
            return KGWWatermarkTokenConfiguration
        elif self.value == "kgw_wmtoken_boundary":
            return KGWWatermarkTokenBoundaryConfiguration
        else:
            raise NotImplementedError("Watermark type not implemented")
