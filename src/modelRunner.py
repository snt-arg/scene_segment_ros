from .fastsam import FastSAM


def fastSamInit(name: str):
    """
    Initializes Fast SAM (Semantic Anything Model) and returns the registered model

    Returns
    -------
    model: dict
        A registered model of Fast SAM
    """
    # Initialization
    model = FastSAM(name)
    return model
