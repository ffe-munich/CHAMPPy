"""CHAMPPy package."""

# TODO: Challangen. Ist das der rcihtige platz, um die Module zu importieren?
from champpy.core.mobility.mobility_components import Logbooks, Vehicles, Clusters, Locations  # noqa: F401
from champpy.core.mobility.mobility_data import MobData  # noqa: F401
from champpy.core.mobility.mobility_cleaning import MobDataCleaner, UserParamsCleaning, LimitConfig  # noqa: F401
from champpy.core.mobility.parameterization import Parameterizer, UserParamsParameterizer, ParamsLoader  # noqa: F401
from champpy.core.mobility.mobility_model import MobModel, UserParamsMobModel  # noqa: F401
from champpy.core.mobility.mobility_validation import (
    MobPlotter,
    UserParamsMobPlotter,
    MobilityCharacteristics,
)  # noqa: F401
from champpy.core.charging.charging_model import (
    ChargingModel,
    ChargingArray,
    UserParamsChargingModel,
    ChargingData,
)  # noqa: F401
from champpy.core.charging.charging_validation import (  # noqa: F401
    ChargingPlotter,
    UserParamsChargingPlotter,
    ChargingCharacteristics,
)
from champpy.utils.time_utils import TypeDays  # noqa: F401
from champpy.utils.logging import setup_logging

setup_logging()  # TODO: Wo packe ich das am besten hin?
