"""Validadores de invariantes arquitectónicos para la red VSN.

Implementa verificaciones programáticas de los invariantes formales que
deben cumplirse en toda instanciación y ejecución válida del modelo:

- I3: Parámetros independientes por plano (θ_x ≠ θ_x' para x≠x', θ_x ≠ θ^dec_x)
- I6: Input Cache siempre precede al encoder en la cadena de ejecución
- I7: H es la única interfaz encoder-decoder (sin skip connections ni caminos auxiliares)

Cada validador lanza InvariantError si la condición es violada, con contexto
suficiente para diagnosticar la causa sin necesidad de debugger.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------


class InvariantError(Exception):
    """Error lanzado cuando un invariante arquitectónico de VSN es violado.

    Incluye contexto descriptivo sobre cuál invariante fue violado, qué
    componentes están involucrados y qué se esperaba para facilitar el
    diagnóstico sin necesidad de debugger.
    """

    pass


# ---------------------------------------------------------------------------
# I3: Independencia de parámetros entre planos
# ---------------------------------------------------------------------------


def validate_no_weight_sharing(encoder: nn.Module, decoder: nn.Module) -> None:
    """Verifica que no hay parámetros compartidos entre planos del encoder y decoder.

    Invariante I3: Cada VGB en posición x tiene tensores de parámetros con
    data_ptr() distinto de cualquier otro VGB en posición x'≠x, tanto dentro
    del encoder como del decoder, y entre encoder y decoder.

    La función recolecta todos los data_ptr() de parámetros de cada bloque VGB
    y verifica que no existe intersección entre conjuntos de punteros de bloques
    distintos.

    Args:
        encoder: Módulo encoder que contiene un atributo ``vgb_blocks``
            (nn.ModuleList de bloques VGB).
        decoder: Módulo decoder que contiene un atributo ``vgb_blocks``
            (nn.ModuleList de bloques VGB).

    Raises:
        InvariantError: Si se detectan parámetros compartidos (mismo data_ptr())
            entre bloques VGB de planos distintos, con detalle de qué bloques
            y parámetros están involucrados.
    """
    enc_blocks = _get_vgb_blocks(encoder, "encoder")
    dec_blocks = _get_vgb_blocks(decoder, "decoder")

    # Mapear: data_ptr → (componente, índice_bloque, nombre_param)
    ptr_registry: Dict[int, Tuple[str, int, str]] = {}

    # Registrar punteros del encoder
    for block_idx, block in enumerate(enc_blocks):
        for param_name, param in block.named_parameters():
            ptr = param.data_ptr()
            if ptr in ptr_registry:
                existing = ptr_registry[ptr]
                raise InvariantError(
                    f"Invariante I3 violado: parámetro compartido detectado.\n"
                    f"  - encoder.vgb_blocks[{block_idx}].{param_name} "
                    f"(data_ptr={ptr})\n"
                    f"  - {existing[0]}.vgb_blocks[{existing[1]}].{existing[2]} "
                    f"(data_ptr={ptr})\n"
                    f"Cada bloque VGB debe tener parámetros independientes."
                )
            ptr_registry[ptr] = ("encoder", block_idx, param_name)

    # Registrar punteros del decoder — verificar contra encoder y otros bloques decoder
    for block_idx, block in enumerate(dec_blocks):
        for param_name, param in block.named_parameters():
            ptr = param.data_ptr()
            if ptr in ptr_registry:
                existing = ptr_registry[ptr]
                raise InvariantError(
                    f"Invariante I3 violado: parámetro compartido detectado.\n"
                    f"  - decoder.vgb_blocks[{block_idx}].{param_name} "
                    f"(data_ptr={ptr})\n"
                    f"  - {existing[0]}.vgb_blocks[{existing[1]}].{existing[2]} "
                    f"(data_ptr={ptr})\n"
                    f"Cada bloque VGB debe tener parámetros independientes "
                    f"entre encoder y decoder."
                )
            ptr_registry[ptr] = ("decoder", block_idx, param_name)


# ---------------------------------------------------------------------------
# I6: Input Cache precede al Encoder
# ---------------------------------------------------------------------------


def validate_input_cache_precedes_encoder(model: nn.Module) -> None:
    """Verifica que el Input Cache precede al encoder en el orden de ejecución.

    Invariante I6: En la cadena forward del modelo, el Input Cache SIEMPRE se
    ejecuta antes que el encoder. Esta función verifica la estructura del modelo
    para confirmar que:
    1. El modelo posee un componente input_cache (o el encoder lo contiene).
    2. El input_cache aparece antes del encoder en el grafo de módulos.

    La verificación es estructural (no requiere ejecutar forward). Se basa en
    la presencia de los atributos esperados y su posición relativa en la
    jerarquía del modelo.

    Args:
        model: Módulo VSNModel (o compatible) que debe contener ``encoder``
            y opcionalmente ``input_cache`` como atributos directos, o el
            encoder debe contener ``input_cache`` internamente.

    Raises:
        InvariantError: Si no se encuentra input_cache, o si la estructura
            indica que no precede al encoder.
    """
    # Buscar input_cache en el modelo directamente o dentro del encoder
    input_cache = _find_submodule(model, "input_cache")
    encoder = _find_submodule(model, "encoder")

    if encoder is None:
        raise InvariantError(
            "Invariante I6: No se encontró el módulo 'encoder' en el modelo. "
            "Se requiere un encoder para validar el orden de ejecución."
        )

    if input_cache is None:
        # Verificar si el encoder contiene el input_cache internamente
        input_cache = _find_submodule(encoder, "input_cache")
        if input_cache is None:
            raise InvariantError(
                "Invariante I6 violado: No se encontró 'input_cache' ni en el "
                "modelo ni dentro del encoder. El Input Cache es un componente "
                "obligatorio que debe preceder al procesamiento del encoder."
            )

    # Verificar orden estructural: input_cache debe estar registrado ANTES
    # que los bloques VGB del encoder en la lista de hijos.
    # Si input_cache está dentro del encoder, verificar que es el primer
    # hijo o que precede a vgb_blocks.
    container = model if hasattr(model, "input_cache") else encoder
    child_names = [name for name, _ in container.named_children()]

    ic_name = None
    vgb_name = None

    for name in child_names:
        child = getattr(container, name)
        if child is input_cache:
            ic_name = name
        # Detectar vgb_blocks por nombre o tipo ModuleList con VGBs
        if name == "vgb_blocks" or (
            isinstance(child, nn.ModuleList) and name != "input_cache"
        ):
            if vgb_name is None:
                vgb_name = name

    if ic_name is None:
        # input_cache encontrado pero no como hijo directo del contenedor —
        # puede estar anidado de forma válida
        return

    if vgb_name is None:
        # No hay vgb_blocks en el contenedor — podría ser una estructura
        # diferente; la validación no puede confirmar violación
        return

    # Verificar que input_cache aparece antes que vgb_blocks en el orden
    # de registro de children (que refleja el orden de construcción en __init__)
    ic_idx = child_names.index(ic_name) if ic_name in child_names else -1
    vgb_idx = child_names.index(vgb_name) if vgb_name in child_names else -1

    if ic_idx >= 0 and vgb_idx >= 0 and ic_idx > vgb_idx:
        raise InvariantError(
            f"Invariante I6 violado: 'input_cache' (posición {ic_idx}) está "
            f"registrado DESPUÉS de 'vgb_blocks' (posición {vgb_idx}) en "
            f"'{type(container).__name__}'. El Input Cache DEBE preceder al "
            f"encoder en la cadena de ejecución."
        )


# ---------------------------------------------------------------------------
# I7: H es la única interfaz encoder-decoder
# ---------------------------------------------------------------------------


def validate_h_sole_interface(model: nn.Module) -> None:
    """Verifica que H (plano latente) es la única interfaz entre encoder y decoder.

    Invariante I7: No existen skip connections, caminos auxiliares ni ningún
    otro mecanismo de paso de información entre encoder y decoder que no sea
    a través del plano latente H (producido por P, consumido por Q).

    La verificación se realiza en dos niveles:
    1. Estructural: Confirma que existen los operadores P y Q como intermediarios.
    2. Parámetros: Confirma que no hay parámetros compartidos directamente
       entre encoder y decoder (lo cual implicaría un camino implícito).
    3. Conexiones: Verifica que no existen módulos con referencias cruzadas
       encoder↔decoder fuera de P y Q.

    Args:
        model: Módulo VSNModel (o compatible) que debe contener ``encoder``,
            ``decoder``, ``P`` (o ``projector_p``) y ``Q`` (o ``transition_q``).

    Raises:
        InvariantError: Si se detectan caminos alternativos entre encoder y
            decoder fuera de H, con detalle de qué componentes violan el
            invariante.
    """
    encoder = _find_submodule(model, "encoder")
    decoder = _find_submodule(model, "decoder")

    if encoder is None:
        raise InvariantError(
            "Invariante I7: No se encontró el módulo 'encoder' en el modelo."
        )
    if decoder is None:
        raise InvariantError(
            "Invariante I7: No se encontró el módulo 'decoder' en el modelo."
        )

    # Verificar existencia de operadores P y Q
    projector_p = _find_submodule_multi(model, ["P", "projector_p", "projector"])
    transition_q = _find_submodule_multi(model, ["Q", "transition_q", "transition"])

    if projector_p is None:
        raise InvariantError(
            "Invariante I7 violado: No se encontró el Operador P (proyección "
            "a H) en el modelo. H debe ser la interfaz encoder-decoder, lo cual "
            "requiere un operador P que produzca H desde V_{X-1}."
        )

    if transition_q is None:
        raise InvariantError(
            "Invariante I7 violado: No se encontró el Operador Q (transición "
            "desde H) en el modelo. H debe ser la interfaz encoder-decoder, "
            "lo cual requiere un operador Q que consuma H para producir V^dec_0."
        )

    # Verificar que no hay parámetros compartidos entre encoder y decoder
    # (compartir parámetros implicaría un canal de información implícito)
    encoder_ptrs: Set[int] = set()
    for param in encoder.parameters():
        encoder_ptrs.add(param.data_ptr())

    shared_params: List[str] = []
    for name, param in decoder.named_parameters():
        if param.data_ptr() in encoder_ptrs:
            shared_params.append(name)

    if shared_params:
        params_str = ", ".join(shared_params[:5])
        extra = (
            f" (y {len(shared_params) - 5} más)"
            if len(shared_params) > 5
            else ""
        )
        raise InvariantError(
            f"Invariante I7 violado: Se detectaron {len(shared_params)} "
            f"parámetro(s) compartidos entre encoder y decoder: "
            f"[{params_str}{extra}]. Esto implica un canal de información "
            f"implícito fuera de H. El plano latente H (P→H→Q) debe ser la "
            f"ÚNICA interfaz encoder-decoder."
        )

    # Verificar que no existen referencias directas (skip connections) entre
    # encoder y decoder fuera de P/Q. Revisamos atributos del modelo que
    # apunten tanto a encoder como a decoder.
    model_children = dict(model.named_children())

    # Los únicos módulos que conectan encoder y decoder deben ser P y Q.
    # Cualquier otro módulo que tenga referencias a ambos sería sospechoso.
    # En la arquitectura VSN, la cadena es: encoder → P → Q → decoder
    # No debe haber otros módulos que tengan como submódulo al encoder o decoder.
    p_names = {"P", "projector_p", "projector"}
    q_names = {"Q", "transition_q", "transition"}
    expected_connectors = p_names | q_names | {"encoder", "decoder", "input_cache", "head"}

    for name, child in model_children.items():
        if name in expected_connectors:
            continue
        # Verificar si este módulo contiene referencias al encoder o decoder
        child_modules = set(child.modules())
        if encoder in child_modules and decoder in child_modules:
            raise InvariantError(
                f"Invariante I7 violado: El módulo '{name}' "
                f"({type(child).__name__}) contiene referencias tanto al "
                f"encoder como al decoder. Solo los operadores P y Q deben "
                f"conectar encoder y decoder a través de H."
            )


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _get_vgb_blocks(module: nn.Module, component_name: str) -> nn.ModuleList:
    """Extrae la lista de bloques VGB de un módulo encoder o decoder.

    Args:
        module: Módulo que debe contener ``vgb_blocks`` como atributo.
        component_name: Nombre del componente para mensajes de error.

    Returns:
        nn.ModuleList de bloques VGB.

    Raises:
        InvariantError: Si no se encuentra ``vgb_blocks``.
    """
    if not hasattr(module, "vgb_blocks"):
        raise InvariantError(
            f"No se encontró 'vgb_blocks' en el {component_name} "
            f"({type(module).__name__}). Se espera un atributo nn.ModuleList "
            f"que contenga los bloques VGB del {component_name}."
        )
    blocks = getattr(module, "vgb_blocks")
    if not isinstance(blocks, nn.ModuleList):
        raise InvariantError(
            f"'{component_name}.vgb_blocks' debe ser nn.ModuleList, "
            f"pero es {type(blocks).__name__}."
        )
    return blocks


def _find_submodule(module: nn.Module, name: str) -> Optional[nn.Module]:
    """Busca un submódulo por nombre como atributo directo.

    Args:
        module: Módulo padre donde buscar.
        name: Nombre del atributo a buscar.

    Returns:
        El submódulo si existe, None en caso contrario.
    """
    if hasattr(module, name):
        attr = getattr(module, name)
        if isinstance(attr, nn.Module):
            return attr
    return None


def _find_submodule_multi(
    module: nn.Module, names: List[str]
) -> Optional[nn.Module]:
    """Busca un submódulo por múltiples nombres posibles (primer match).

    Args:
        module: Módulo padre donde buscar.
        names: Lista de nombres candidatos a buscar.

    Returns:
        El primer submódulo encontrado, o None si ninguno existe.
    """
    for name in names:
        result = _find_submodule(module, name)
        if result is not None:
            return result
    return None
