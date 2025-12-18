# VACL wrapper
#
# # vacl_wrapper.py
# # ============================================================
# # [DROP-IN] VACLWrapper
# #
# # Purpose:
# #   - Wraps your VACL block (VACLVA) in a DDP/Lightning-friendly way
# #   - Ensures return_intermediates defaults to False (memory-safe)
# #   - Standardizes returned keys:
# #       "loss_vacl" (and optionally "loss" as alias)
# #   - Optionally strips bulky intermediates (B,S,S matrices etc.)
# #
# # Expected inputs:
# #   X_v: Tensor (B, D_v, S)  OR (B, S, D_v)  (we support both)
# #   X_a: Tensor (B, D_a, S)  OR (B, S, D_a)
# #
# # NOTE:
# #   - This wrapper does NOT move tensors to device.
# #   - This wrapper does NOT log (keep logging in Lightning system).
# # ============================================================
#
# from __future__ import annotations
#
# from dataclasses import dataclass
# from typing import Any, Dict, Optional
#
# import torch
# import torch.nn as nn
#
# # ============================================================
# # [EXISTING] Import your actual VACL block
# # ============================================================
# from vacl_block import VACLVA
#
#
# # ============================================================
# # [ADDED] Optional config container
# # ============================================================
# @dataclass
# class VACLWrapperConfig:
#     # [ADDED] safety defaults
#     return_intermediates: bool = False          # memory-safe default
#     strip_intermediates: bool = True            # drop big tensors from output
#     # [ADDED] expected token dimension ordering
#     expect_bds: bool = True                     # True expects (B,D,S); False expects (B,S,D)
#
#
# class VACLWrapper(nn.Module):
#     def __init__(
#         self,
#         # ============================================================
#         # [MODIFIED] Either pass a ready VACLVA instance OR its kwargs
#         # ============================================================
#         vacl: Optional[VACLVA] = None,
#         vacl_kwargs: Optional[Dict[str, Any]] = None,
#         # ============================================================
#         # [ADDED] Wrapper behaviour
#         # ============================================================
#         cfg: Optional[VACLWrapperConfig] = None,
#         return_intermediates: Optional[bool] = None,
#         strip_intermediates: Optional[bool] = None,
#         expect_bds: Optional[bool] = None,
#     ) -> None:
#         super().__init__()
#
#         self.cfg = cfg or VACLWrapperConfig()
#
#         # [ADDED] Allow override without forcing you to build a dataclass
#         if return_intermediates is not None:
#             self.cfg.return_intermediates = bool(return_intermediates)
#         if strip_intermediates is not None:
#             self.cfg.strip_intermediates = bool(strip_intermediates)
#         if expect_bds is not None:
#             self.cfg.expect_bds = bool(expect_bds)
#
#         # [MODIFIED] Build underlying VACLVA once (DDP-safe)
#         if vacl is not None:
#             self.vacl = vacl
#         else:
#             vacl_kwargs = vacl_kwargs or {}
#             self.vacl = VACLVA(**vacl_kwargs)
#
#     # ============================================================
#     # [ADDED] Shape adapter: normalize to what your VACLVA expects
#     # ============================================================
#     @staticmethod
#     def _to_expected_layout(x: torch.Tensor, expect_bds: bool) -> torch.Tensor:
#         """
#         If expect_bds=True, ensures shape (B,D,S).
#         If expect_bds=False, ensures shape (B,S,D).
#         """
#         if x.ndim != 3:
#             raise ValueError(f"Expected 3D tensor (B,*,*), got shape {tuple(x.shape)}")
#
#         B, A, C = x.shape
#
#         # Heuristic: if expect_bds and second dim is "token count" (S) not "feature dim" (D),
#         # user may have passed (B,S,D). We transpose.
#         # We do NOT guess aggressively; we follow expect_bds.
#         if expect_bds:
#             # want (B,D,S)
#             # if likely (B,S,D), transpose last two dims
#             # (B,S,D) -> (B,D,S)
#             return x.transpose(1, 2).contiguous() if A < C else x
#         else:
#             # want (B,S,D)
#             # if likely (B,D,S), transpose last two dims
#             # (B,D,S) -> (B,S,D)
#             return x.transpose(1, 2).contiguous() if A > C else x
#
#     # ============================================================
#     # [ADDED] Forward
#     # ============================================================
#     def forward(
#         self,
#         *,
#         X_v: torch.Tensor,
#         X_a: torch.Tensor,
#         compute_infonce: bool = True,
#         return_intermediates: Optional[bool] = None,
#         **kwargs,
#     ) -> Dict[str, Any]:
#         """
#         Returns a dict with at least:
#           - loss_vacl : Tensor scalar
#           - (optionally) other diagnostic tensors if not stripped
#         """
#         # [ADDED] Resolve flags
#         if return_intermediates is None:
#             return_intermediates = bool(self.cfg.return_intermediates)
#
#         # [ADDED] Normalize tensor layout (keeps caller flexible)
#         X_v = self._to_expected_layout(X_v, expect_bds=self.cfg.expect_bds)
#         X_a = self._to_expected_layout(X_a, expect_bds=self.cfg.expect_bds)
#
#         # [ADDED] Basic integrity check: ensure same S
#         if self.cfg.expect_bds:
#             # (B,D,S)
#             if X_v.shape[2] != X_a.shape[2]:
#                 raise ValueError(f"VACL expects same S for both modalities. Got Sv={X_v.shape[2]} Sa={X_a.shape[2]}")
#         else:
#             # (B,S,D)
#             if X_v.shape[1] != X_a.shape[1]:
#                 raise ValueError(f"VACL expects same S for both modalities. Got Sv={X_v.shape[1]} Sa={X_a.shape[1]}")
#
#         # ============================================================
#         # [KEPT] Call underlying VACLVA
#         # ============================================================
#         vacl_out = self.vacl(
#             X_v=X_v,
#             X_a=X_a,
#             # compute_infonce=bool(compute_infonce),
#             return_intermediates=bool(return_intermediates),
#             **kwargs,
#         )
#
#         # ============================================================
#         # [ADDED] Standardize output dict
#         # ============================================================
#         if not isinstance(vacl_out, dict):
#             # If your VACLVA returns just a loss tensor, wrap it
#             if isinstance(vacl_out, torch.Tensor):
#                 return {"loss_vacl": vacl_out, "loss": vacl_out}
#             raise TypeError(f"VACLVA returned unexpected type: {type(vacl_out)}")
#
#         out: Dict[str, Any] = dict(vacl_out)
#
#         # [ADDED] Normalize loss key
#         loss = out.get("L_cor", out.get("loss", None))
#         if loss is None:
#             # Some implementations use "loss_total"
#             loss = out.get("loss_total", None)
#         if loss is None:
#             raise KeyError(
#                 "VACLVA output dict did not contain a loss key among "
#                 "['loss_vacl','loss','loss_total']"
#             )
#
#         out["loss_vacl"] = loss
#         out["loss"] = loss  # alias, useful for generic callers
#
#         # ============================================================
#         # [ADDED] Strip heavy intermediates by default (memory-friendly)
#         # ============================================================
#         if self.cfg.strip_intermediates and not return_intermediates:
#             # Common heavy keys seen in VACL-style implementations
#             for k in [
#                 "M_v", "M_a", "J_va", "J_av",
#                 "attn", "attn_weights",
#                 "pos_mask", "neg_mask",
#                 "logits", "logits_va", "logits_av",
#                 "intermediates",
#             ]:
#                 out.pop(k, None)
#
#         return out
