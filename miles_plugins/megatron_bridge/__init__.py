"""Miles plugin package for ``megatron.bridge`` integration.

Importing this package is enough to:

* register miles' bridge subclasses (e.g.
  :class:`~miles_plugins.megatron_bridge.nemotron_h.MilesNemotronHBridge`) via
  ``@MegatronModelBridge.register_bridge`` so ``AutoBridge`` picks them up
  instead of the upstream defaults;
* install general-purpose shims that make ``megatron.bridge`` cooperate with
  miles infrastructure (e.g. ``ReloadableProcessGroup``).

Every shim / registration is wrapped in try/except so an import-time failure of
one model does not prevent other models from working.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _install_bridge_pp_group_unwrap() -> None:
    """Let ``MegatronParamMapping.broadcast_obj_from_pp_rank`` work with
    miles' :class:`~miles.utils.reloadable_process_group.ReloadableProcessGroup`.

    ``broadcast_obj_from_pp_rank`` calls ``broadcast_object_list`` on
    ``self.pp_group``, which goes through ``_world.pg_group_ranks``. Miles wraps
    every ``ProcessGroup`` in ``ReloadableProcessGroup`` for reload-safety; that
    wrapper is not in ``pg_group_ranks`` so ``get_group_rank`` raises
    ``"Group ... is not registered"``. Temporarily swap in the inner group for
    the duration of the broadcast.
    """
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    from miles.utils.reloadable_process_group import ReloadableProcessGroup

    if getattr(MegatronParamMapping, "_miles_pp_group_unwrap_installed", False):
        return

    _orig = MegatronParamMapping.broadcast_obj_from_pp_rank

    def broadcast_obj_from_pp_rank(self, obj, name=None):
        if not isinstance(self.pp_group, ReloadableProcessGroup):
            return _orig(self, obj, name)
        saved = self.pp_group
        self.pp_group = saved.group
        try:
            return _orig(self, obj, name)
        finally:
            self.pp_group = saved

    MegatronParamMapping.broadcast_obj_from_pp_rank = broadcast_obj_from_pp_rank
    MegatronParamMapping._miles_pp_group_unwrap_installed = True


try:
    _install_bridge_pp_group_unwrap()
except Exception as _e:  # best-effort
    logger.warning("miles bridge shim _install_bridge_pp_group_unwrap not applied: %s", _e)


def _install_remove_non_pickleables_guard() -> None:
    """Make ``megatron.bridge``'s ``remove_non_pickleables`` robust to objects
    that cannot be shallow-copied.

    During megatron->HF weight conversion (colocate + ``--megatron-to-hf-mode
    bridge``), ``MegatronParamMapping`` calls ``remove_non_pickleables(config)``
    to strip non-pickleable attributes before broadcasting the HF config. The
    helper does ``copy.copy(obj)`` on every object exposing ``__dict__``,
    assuming it is shallow-copyable. Miles attaches a ``ReloadableProcessGroup``
    (a ``torch.distributed.ProcessGroup`` subclass wrapping a c10d process group)
    to the config; ``copy.copy`` on it raises
    ``TypeError: cannot pickle 'ReloadableProcessGroup' object``, aborting
    ``update_weights``.

    Wrap the helper so process groups are dropped outright and any other
    un-copyable object is treated as non-pickleable (returned as ``None``),
    matching the helper's stated intent.
    """
    import torch

    from megatron.bridge.models.conversion import param_mapping as _pm
    from megatron.bridge.models.conversion import utils as _utils

    if getattr(_utils, "_miles_remove_non_pickleables_guarded", False):
        return

    _orig = _utils.remove_non_pickleables

    def remove_non_pickleables(obj, max_depth=3, current_depth=0):
        # Process groups (incl. miles' ReloadableProcessGroup) are inherently
        # non-pickleable -> drop them instead of trying to copy/recurse.
        if isinstance(obj, torch.distributed.ProcessGroup):
            return None
        try:
            return _orig(obj, max_depth, current_depth)
        except (TypeError, ValueError, RuntimeError):
            # Could not shallow-copy / clean this object -> treat as non-pickleable.
            return None

    # Patch the canonical name and param_mapping's already-imported alias. The
    # original body recurses via the module-global name, so this wrapper is hit
    # at every depth.
    _utils.remove_non_pickleables = remove_non_pickleables
    _pm.remove_non_pickleables = remove_non_pickleables
    _utils._miles_remove_non_pickleables_guarded = True


try:
    _install_remove_non_pickleables_guard()
except Exception as _e:  # best-effort
    logger.warning("miles bridge shim _install_remove_non_pickleables_guard not applied: %s", _e)


# Model-specific bridge subclasses. Each submodule self-installs on import.
# Keep imports here so merely importing ``miles_plugins.megatron_bridge`` is
# enough to pick up every miles bridge (mirrors ``miles_plugins.mbridge``).
try:
    from . import nemotron_h  # noqa: F401
except Exception as _e:  # pragma: no cover - defensive
    logger.warning("miles nemotron_h plugin failed to load: %s", _e)
