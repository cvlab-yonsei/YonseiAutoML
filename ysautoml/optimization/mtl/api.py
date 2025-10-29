from .examples.nyusp.api import train_mtl_nyusp
# from .examples.office.api import train_mtl_office  # 추후 추가


def train_mtl(dataset="nyusp", **kwargs):
    if dataset.lower() == "nyusp":
        return train_mtl_nyusp(**kwargs)
    elif dataset.lower() == "office":
        raise NotImplementedError("Office MTL API not yet implemented.")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
