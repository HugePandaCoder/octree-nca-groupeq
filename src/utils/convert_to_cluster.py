
def convert_paths_to_cluster_paths(config: dict) -> dict:
    out = {}
    for k,v in config.items():
        if isinstance(v, str):
            out[k] = v.replace(r"/local/scratch/", r"/gris/scratch-gris-filesrv/")
        else:
            out[k] = v
    return out
