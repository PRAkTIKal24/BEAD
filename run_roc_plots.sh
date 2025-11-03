#!/bin/bash
uv run bead -m plot -p zprime convvae -o roc_per_signal -v
uv run bead -m plot -p zprime convvae_planar -o roc_per_signal -v
uv run bead -m plot -p zprime convvae_house -o roc_per_signal -v
uv run bead -m plot -p zprime ntx_convvae -o roc_per_signal -v
uv run bead -m plot -p zprime dvae -o roc_per_signal -v
uv run bead -m plot -p zprime convvae_sc -o roc_per_signal -v
uv run bead -m plot -p zprime convvae_house_sc_anneal -o roc_per_signal -v
uv run bead -m plot -p 2classzp hp_convvae -o roc_per_signal -v
uv run bead -m plot -p 2classzp hs_convvae -o roc_per_signal -v
uv run bead -m plot -p 2classzp ps_convvae -o roc_per_signal -v
uv run bead -m plot -p 2classzp hp_sc_convvae -o roc_per_signal -v
uv run bead -m plot -p 2classzp hs_sc_convvae -o roc_per_signal -v
uv run bead -m plot -p 2classzp ps_sc_convvae -o roc_per_signal -v
# uv run bead -m plot -p 1class convvae_h -o roc_per_signal -v
# uv run bead -m plot -p 1class convvae_p -o roc_per_signal -v
# uv run bead -m plot -p 1class convvae_s -o roc_per_signal -v
