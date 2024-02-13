from torch.hub import load_state_dict_from_url
import torch.distributed as dist

PRETRAINED_MODEL_URLS = {

}

def load_pretrained_model(pretrained_model_name):
    if pretrained_model_name not in PRETRAINED_MODEL_URLS:
        raise ValueError("Unknown pretrained model name %s", pretrained_model_name)
    if not dist.is_initialized():
        return load_state_dict_from_url(PRETRAINED_MODEL_URLS[pretrained_model_name], progress=True)["model"]
    else:
        pretrained_model = load_state_dict_from_url(PRETRAINED_MODEL_URLS[pretrained_model_name], progress=True, file_name=f"{pretrained_model_name}_{dist.get_rank()}")["model"]
        dist.barrier()
        return pretrained_model
