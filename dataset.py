from torch.utils.data import Dataset

class BioDataset(Dataset):
    def __init__(self, config):
        self.config = config
        
        return