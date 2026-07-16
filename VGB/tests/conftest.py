"""Configuración global de Hypothesis para tests de propiedad VGB."""

from hypothesis import HealthCheck, settings

# Registrar perfil "vsn" con configuración apropiada para tests con PyTorch
settings.register_profile(
    "vsn",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,  # PyTorch ops pueden ser lentas
)

# Cargar el perfil por defecto
settings.load_profile("vsn")
