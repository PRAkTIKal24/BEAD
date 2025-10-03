#!/bin/bash
uv run bead -m plot -p csf_results convvae -v
uv run bead -m plot -p csf_results convvae_planar -v
uv run bead -m plot -p csf_results convvae_house -v
uv run bead -m plot -p csf_results ntx_convvae -v
uv run bead -m plot -p csf_results dvae -v
uv run bead -m plot -p csf_results convvae_sc -v
uv run bead -m plot -p csf_results convvae_house_sc_anneal -v
uv run bead -m plot -p 2class hp_convvae -v
uv run bead -m plot -p 2class hs_convvae -v
uv run bead -m plot -p 2class ps_convvae -v
uv run bead -m plot -p 2class hp_sc_convvae -v
uv run bead -m plot -p 2class hs_sc_convvae -v
uv run bead -m plot -p 2class ps_sc_convvae -v
uv run bead -m plot -p 1class convvae_h -v
uv run bead -m plot -p 1class convvae_p -v
uv run bead -m plot -p 1class convvae_s -v
