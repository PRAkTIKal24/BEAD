#!/bin/bash
uv run bead -m plot -p csf_results convvae -o roc_per_signal -v
uv run bead -m plot -p csf_results convvae_planar -o roc_per_signal -v
uv run bead -m plot -p csf_results convvae_house -o roc_per_signal -v
uv run bead -m plot -p csf_results ntx_convvae -o roc_per_signal -v
uv run bead -m plot -p csf_results dvae -o roc_per_signal -v
uv run bead -m plot -p csf_results convvae_sc -o roc_per_signal -v
uv run bead -m plot -p csf_results convvae_house_sc_anneal -o roc_per_signal -v
uv run bead -m plot -p 2class hp_convvae -o roc_per_signal -v
uv run bead -m plot -p 2class hs_convvae -o roc_per_signal -v
uv run bead -m plot -p 2class ps_convvae -o roc_per_signal -v
uv run bead -m plot -p 2class hp_sc_convvae -o roc_per_signal -v
uv run bead -m plot -p 2class hs_sc_convvae -o roc_per_signal -v
uv run bead -m plot -p 2class ps_sc_convvae -o roc_per_signal -v
uv run bead -m plot -p 1class convvae_h -o roc_per_signal -v
uv run bead -m plot -p 1class convvae_p -o roc_per_signal -v
uv run bead -m plot -p 1class convvae_s -o roc_per_signal -v
