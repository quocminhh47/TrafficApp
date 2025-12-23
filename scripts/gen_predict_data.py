# scripts/gen_predict_data.py
from modules.predict_cache import build_forecast_cache


def main():
    build_forecast_cache(
        city="Minneapolis",
        zone="I94",
        route_id="I-94-WB",
        file_name="i94_main",
    )

    build_forecast_cache(
        city="Seattle",
        zone="FremontBridge",
        route_id="Fremont-East",
        file_name="Fremont_East",
    )

    build_forecast_cache(
        city="Seattle",
        zone="FremontBridge",
        route_id="Fremont-Total",
        file_name="Fremont_Total",
    )

    build_forecast_cache(
        city="Seattle",
        zone="FremontBridge",
        route_id="Fremont-West",
        file_name="Fremont_West",
    )


if __name__ == "__main__":
    main()
