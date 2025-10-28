## Implementation of AZ-NAS on NAS-Bench-201 (NATS-Bench-TSS)
- Prepare the API file from [NATS-Bench](https://github.com/D-X-Y/NATS-Bench/tree/1d4a304ad1906aa5866563438fcbf0d624b7eda2) (e.g., `./api_data/NATS-tss-v1_0-3ffb9-simple`).
- Run `tss_general.ipynb` for experiments.

## Credit
- `nats_bench` taken from [NATS-Bench](https://github.com/D-X-Y/NATS-Bench/tree/1d4a304ad1906aa5866563438fcbf0d624b7eda2)
- `xautodl` and `configs` taken from [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects/tree/f46486e21b71ae6459a700be720d7648b5429569)

## Change Notes
- `./custom/tss_model.py`: Modified from `./xaudodl/models/cell_infers/tiny_network.py`
> *  Used for constructing individual networks (`tss_general.ipynb`)
> *  Add a function to extract cell features

- `./custom/tss_model_supernet.py`: Modified from `./xaudodl/models/cell_searchs/search_model_random.py`
> *  Used for constructing a supernet (`tss_supernet(gradsign).ipynb`)
> *  Add a function to extract cell features
