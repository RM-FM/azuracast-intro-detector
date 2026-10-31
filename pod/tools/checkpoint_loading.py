"""Use PyTorch's restricted loader for PANNs, including its library-internal load."""
import os

def load_weights_only(factory,**kwargs):
    # Supported by pinned PyTorch 2.5.1; scope to the PANNs constructor.
    # Never fall back to unrestricted pickle.
    key='TORCH_FORCE_WEIGHTS_ONLY_LOAD';previous=os.environ.get(key)
    os.environ[key]='1'
    try:return factory(**kwargs)
    finally:
        if previous is None:os.environ.pop(key,None)
        else:os.environ[key]=previous
