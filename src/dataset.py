import torch as pt
from torch.utils.data import Dataset

class DigitPictures(Dataset):
    """Pictures with labels dataset."""

    def __init__(self, labels, images, transform=None):
        """
        Args:
            labels: 2xN array with 1 field is picID, 2nd is max label.
            images: the training images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):  # returns the length of dataset
        return len(self.labels)

    def __getitem__(self, idx):
        if pt.is_tensor(idx):
            idx = idx.tolist()

        image = pt.tensor(self.images[idx])
        image = pt.unsqueeze(image, 0)
        label = pt.tensor(self.labels[idx][1])
        sample = (image, label)

        if self.transform:
            sample = self.transform(sample)

        return sample
