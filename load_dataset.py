import os
import re
import shutil
import zipfile
from pathlib import Path
from warnings import warn

import cv2
import numpy as np
from loguru import logger
import torch
import torch.utils.data as data

class BaseDataModule:
    """
    A base class for a DataModule.
    DataModules perform all the steps needed for a dataset, from downloading the data to creating dataloaders.
    Specific DataModules should inherit from this class.
    Inspired by pytorch-lightning LightningDataModule
    """

    def train_dataloader(self):
        raise NotImplementedError

    def valid_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


def pannuke_multiclass_mask_to_nucleus_mask(multiclass_mask):
    """
    Convert multiclass mask from PanNuke to a single channel nucleus mask.
    Assumes each pixel is assigned to one and only one class. Sums across channels, except the last mask channel
    which indicates background pixels in PanNuke.
    Operates on a single mask.
    Args:
        multiclass_mask (torch.Tensor): Mask from PanNuke, in classification setting. (i.e. ``nucleus_type_labels=True``).
            Tensor of shape (6, 256, 256).
    Returns:
        Tensor of shape (256, 256).
    """
    # verify shape of input
    # ignore last channel
    out = np.sum(multiclass_mask, axis=-1)
    return out

class PanNukeDataset(data.Dataset):
    """
    Dataset object for PanNuke dataset

    Tissue types: Breast, Colon, Bile-duct, Esophagus, Uterus, Lung, Cervix, Head&Neck, Skin, Adrenal-Gland, Kidney,
    Stomach, Prostate, Testis, Liver, Thyroid, Pancreas, Ovary, Bladder

    masks are arrays of 6 channel instance-wise masks
    (0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 5: Background)
    If ``classification is False`` then only a single channel mask will be returned, which is the inverse of the
    'Background' mask (i.e. nucleus pixels are 1.). Otherwise, the full 6-channel masks will be returned.

    If using transforms for data augmentation, the transform must accept two arguments (image and mask) and return a
    dict with "image" and "mask" keys.
    See example here https://albumentations.ai/docs/getting_started/mask_augmentation

    Args:
        data_dir: Path to PanNuke data. Should contain an 'images' directory and a 'masks' directory.
            Images should be 256x256 RGB in a format that can be read by `cv2.imread()` (e.g. png).
            Masks should be .npy files of shape (6, 256, 256).
            Image and mask files should be named in 'fold<fold_ix>_<i>_<tissue>' format.
        fold_ix: Index of which fold of PanNuke data to use. One of 1, 2, or 3. If ``None``, ignores the folds and uses
            the entire PanNuke dataset. Defaults to ``None``.
        transforms: Transforms to use for data augmentation. Must accept two arguments (image and mask) and return a
            dict with "image" and "mask" keys. If ``None``, no transforms are applied. Defaults to ``None``.
        nucleus_type_labels (bool, optional): Whether to provide nucleus type labels, or binary nucleus labels.
            If ``True``, then masks will be returned with six channels, corresponding to

                0. Neoplastic cells
                1. Inflammatory
                2. Connective/Soft tissue cells
                3. Dead Cells
                4. Epithelial
                5. Background

            If ``False``, then the returned mask will have a single channel, with zeros for background pixels and ones
            for nucleus pixels (i.e. the inverse of the Background mask). Defaults to ``False``.
        hovernet_preprocess (bool): Whether to perform preprocessing specific to HoVer-Net architecture. If ``True``,
            the center of mass of each nucleus will be computed, and an additional mask will be returned with the
            distance of each nuclear pixel to its center of mass in the horizontal and vertical dimensions.
            This corresponds to Gamma(I) from the HoVer-Net paper. Defaults to ``False``.
    """

    def __init__(
        self,
        data_dir,
        fold_ix=None,
        transforms=None,
        nucleus_type_labels=False,
        hovernet_preprocess=False,
    ):
        self.data_dir = data_dir
        self.fold_ix = fold_ix
        self.transforms = transforms
        self.nucleus_type_labels = nucleus_type_labels
        self.hovernet_preprocess = hovernet_preprocess

        data_dir = Path(data_dir)

        # dirs for images, masks
        imdir = os.path.join(data_dir, "images")
        maskdir = os.path.join(data_dir, "masks")

        # stop if the images and masks directories don't already exist
        #assert imdir.is_dir(), f"Error: 'images' directory not found: {imdir}"
        #assert maskdir.is_dir(), f"Error: 'masks' directory not found: {maskdir}"

        if self.fold_ix is None:
            paths = list(imdir.glob("*"))
        else:
            paths = os.listdir(imdir)

        self.imdir = imdir
        self.maskdir = maskdir
        self.paths = [p for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        stem = self.paths[ix]
        impath = os.path.join(self.imdir, stem)
        maskpath = os.path.join(self.maskdir, stem)
        
        im = cv2.imread(str(impath))
        mask = cv2.imread(str(maskpath))

        if self.nucleus_type_labels is False:
            # only look at "background" mask in last channel
            mask = mask[:, :, 0]
            # invert so that ones are nuclei pixels
            mask = 1 - mask

        if self.transforms is not None:
            transformed = self.transforms(image=im, mask=mask)
            im = transformed["image"]
            mask = transformed["mask"]

        # swap channel dim to pytorch standard (C, H, W)
        im = im.transpose((2, 0, 1))
        # compute hv map
        if self.hovernet_preprocess:
            if self.nucleus_type_labels:
                # sum across mask channels to squash mask channel dim to size 1
                # don't sum the last channel, which is background!
                mask_1c = pannuke_multiclass_mask_to_nucleus_mask(mask)
            else:
                mask_1c = mask

        if self.hovernet_preprocess:
            out = (
                torch.from_numpy(im),
                torch.from_numpy(mask),
            )
        else:
            out = torch.from_numpy(im), torch.from_numpy(mask)
        return out


class PanNukeDataModule(BaseDataModule):
    """
    DataModule for the PanNuke Dataset. Contains 256px image patches from 19 tissue types with annotations for 5
    nucleus types. For more information, see: https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke

    Args:
        data_dir (str): Path to directory where PanNuke data is
        download (bool, optional): Whether to download the data. If ``True``, checks whether data files exist in
            ``data_dir`` and downloads them to ``data_dir`` if not.
            If ``False``, checks to make sure that data files exist in ``data_dir``. Default ``False``.
        shuffle (bool, optional): Whether to shuffle images. Defaults to ``True``.
        transforms (optional): Data augmentation transforms to apply to images. Transform must accept two arguments:
            (mask and image) and return a dict with "image" and "mask" keys. See an example here:
            https://albumentations.ai/docs/getting_started/mask_augmentation/
        nucleus_type_labels (bool, optional): Whether to provide nucleus type labels, or binary nucleus labels.
            If ``True``, then masks will be returned with six channels, corresponding to

                0. Neoplastic cells
                1. Inflammatory
                2. Connective/Soft tissue cells
                3. Dead Cells
                4. Epithelial
                5. Background

            If ``False``, then the returned mask will have a single channel, with zeros for background pixels and ones
            for nucleus pixels (i.e. the inverse of the Background mask). Defaults to ``False``.
        split (int, optional): How to divide the three folds into train, test, and validation splits. Must be one of
            {1, 2, 3, None} corresponding to the following splits:

                1. Training: Fold 1; Validation: Fold 2; Testing: Fold 3
                2. Training: Fold 2; Validation: Fold 1; Testing: Fold 3
                3. Training: Fold 3; Validation: Fold 2; Testing: Fold 1

            If ``None``, then the entire PanNuke dataset will be used. Defaults to ``None``.
        batch_size (int, optional): batch size for dataloaders. Defaults to 8.
        hovernet_preprocess (bool): Whether to perform preprocessing specific to HoVer-Net architecture. If ``True``,
            the center of mass of each nucleus will be computed, and an additional mask will be returned with the
            distance of each nuclear pixel to its center of mass in the horizontal and vertical dimensions.
            This corresponds to Gamma(I) from the HoVer-Net paper. Defaults to ``False``.

    References
        Gamper, J., Koohbanani, N.A., Benet, K., Khuram, A. and Rajpoot, N., 2019, April. PanNuke: an open pan-cancer
        histology dataset for nuclei instance segmentation and classification. In European Congress on Digital
        Pathology (pp. 11-19). Springer, Cham.

        Gamper, J., Koohbanani, N.A., Graham, S., Jahanifar, M., Khurram, S.A., Azam, A., Hewitt, K. and Rajpoot, N.,
        2020. PanNuke Dataset Extension, Insights and Baselines. arXiv preprint arXiv:2003.10778.
    """

    def __init__(
        self,
        data_dir,
        download=False,
        shuffle=True,
        transforms=None,
        nucleus_type_labels=False,
        split=None,
        batch_size=8,
        hovernet_preprocess=False,
    ):
        self.data_dir = Path(data_dir)
        self.download = download
        if download:
            self._download_pannuke(self.data_dir)
        else:
            # make sure that subdirectories exist
            imdir = self.data_dir / "images"
            maskdir = self.data_dir / "masks"
            assert (
                imdir.is_dir()
            ), f"`download is False` but 'images' subdirectory not found at {imdir}"
            assert (
                maskdir.is_dir()
            ), f"`download is False` but 'masks' subdirectory not found at {maskdir}"

        self.shuffle = shuffle
        self.transforms = transforms
        self.nucleus_type_labels = nucleus_type_labels
        assert split in [
            1,
            2,
            3,
            None,
        ], f"Error: input split {split} not valid. Must be one of [1, 2, 3] or None."
        self.split = split
        self.batch_size = batch_size
        self.hovernet_preprocess = hovernet_preprocess

    def _get_dataset(self, fold_ix, augment=True):
        if augment:
            transforms = self.transforms
        else:
            transforms = None
        return PanNukeDataset(
            data_dir=self.data_dir,
            fold_ix=fold_ix,
            transforms=transforms,
            nucleus_type_labels=self.nucleus_type_labels,
            hovernet_preprocess=self.hovernet_preprocess,
        )

    @property
    def train_dataloader(self):
        """
        Dataloader for training set.
        Yields (image, mask, tissue_type), or (image, mask, hv, tissue_type) for HoVer-Net
        """
        return data.DataLoader(
            dataset=self._get_dataset(fold_ix=self.split, augment=True),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
        )
    @property
    def valid_dataloader(self):
        """
        Dataloader for validation set.
        Yields (image, mask, tissue_type), or (image, mask, hv, tissue_type) for HoVer-Net
        """
        if self.split in [1, 3]:
            fold_ix = 2
        else:
            fold_ix = 1
        return data.DataLoader(
            self._get_dataset(fold_ix=fold_ix, augment=False),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    @property
    def test_dataloader(self):
        """
        Dataloader for test set.
        Yields (image, mask, tissue_type), or (image, mask, hv, tissue_type) for HoVer-Net
        """
        if self.split in [1, 2]:
            fold_ix = 3
        else:
            fold_ix = 1
        return data.DataLoader(
            self._get_dataset(fold_ix=fold_ix, augment=False),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
        )
