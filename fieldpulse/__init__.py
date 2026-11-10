"""FieldPulse: a data pipeline for machine field-test telemetry."""
SIGNALS = ["engine_rpm", "engine_temp_c", "ground_speed_kph", "fuel_rate_lph", "hyd_pressure_bar"]

# Physical limits for each signal. Anything outside is treated as a bad reading.
LIMITS = {
    "engine_rpm": (0, 3000),
    "engine_temp_c": (-30, 125),
    "ground_speed_kph": (0, 50),
    "fuel_rate_lph": (0, 60),
    "hyd_pressure_bar": (0, 250),
}

# Signals that always carry some noise, so a perfectly flat stretch means a stuck sensor.
# Ground speed is left out because a parked machine really does read 0 for minutes.
NOISY_SIGNALS = ["engine_rpm", "engine_temp_c", "fuel_rate_lph", "hyd_pressure_bar"]
