import os
import json
import copy
import random

from collections import Counter, defaultdict

import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import Compose, RandomApply, RandomHorizontalFlip, RandomVerticalFlip, ToTensor

from data.data_loader_checks import check_kaggle2015_config
from data.transforms import Rotate, RandomHorizontalFlipRotate, RandomApplyChoice, MaskedStandardization, KaggleWinner, GreenChannelOnly
from data.transforms_pytorch import RandomRotation, RandomResizedCrop


# Mean/Std data sets pixel values per channel (RGB)
# Default is mean 1 and std 1e-6 (to avoid nan values). Used for fake dataset.
DATASETS_MEAN_STD = defaultdict(lambda: {'mean': (1.0, 1.0, 1.0), 'std': (1e-6, 1e-6, 1e-6)}, {
    # Kaggle statistics have been computed with high ram
    'kaggle2015_512': {
        'mean': (126.004, 87.6528, 63.1658),
        'std': (58.794, 44.1894, 38.9899)},
    'kaggle2015_1024': {
        'mean': (127.5878, 88.6528, 63.7922),
        'std': (57.5784, 43.5043, 38.6764)}})


class RetinoDataset(Dataset):
    """Base class for retino datasets."""

    def __init__(self, data_folder, cfg, split='train', transform=ToTensor()):
        """
        Parameters
        ----------
        data_folder : str
            Data folder.
        cfg : dict
            resolution : int
                Input size in [512, 1024].
            sample_size : int
                Number of elements to use as sample size, for debugging purposes only.
                If -1, use all samples.
            use_other_eye: bool
                Whether or not to return the secondary eye to make the prediction
            target_name: str
                Can be 'screening_level' or 'DR_level'. Changes the type of the targets.
            train: dict
                filter_target : str
                    Filter out samples with DR_level equal 1 and 2.
                n_views : int
                    Number of views.
                augmentation : dict
                    different_transform_per_view: bool
                        Whether or not to apply a different transformation per view.
            eval: dict
                n_views : int
                    Number of views.
                apply_train_augmentations: bool
                    Whether or not to apply train transform.
        split : str, optional
            Should be in ['train', 'valid', 'test']. Default = 'train'
        transform : callable, optional
            Transform to be applied on a sample. Default = ToTensor().
        """

        self._transform = transform
        if split == 'train':
            self._n_views = cfg['train']['n_views']
        else:
            self._n_views = cfg['eval']['n_views']
        self._different_transform_per_view = cfg['train']['augmentation']['different_transform_per_view']
        self._use_other_eye = cfg['use_both_eyes']
        self._target_type = cfg['target_name']

        self._input_size = cfg['resolution']
        self._sample_size = cfg['sample_size']

        # Define num_class
        if self._target_type == "screening_level":
            self.num_class = 1
            self.raw_labels = list(range(self.num_class + 1))
        elif self._target_type == "DR_level":
            self.num_class = 5
            self.raw_labels = list(range(self.num_class))

        self._filter_target = (split == 'train' and self._target_type == "screening_level" and cfg['train']['filter_target'])

        # Get the split meta-data. They are a dict of dict where first level
        # of keys are ids and the second level of keys are to acces the different
        # meta-data associated with the ids.
        labels_file_name = os.path.join(data_folder, f"labels_{split}.json")
        if not os.path.exists(labels_file_name):
            raise ValueError(f"data_folder does not contain {labels_file_name}")

        with open(labels_file_name) as f:
            self._labels = json.load(f)

        # Get the list of ids that are part of the split.
        patient_ids = list(self._labels.keys())
        patient_ids.sort()

        if self._sample_size != -1:
            # divided by 2 because patients have 2 eye so 2 sample each
            patient_ids = patient_ids[:self._sample_size // 2]

        if self._filter_target:
            print("Warning: dataset.train.filter_target is True. Actual sample size may vary from the config one.")

        # Duplicate patients to have one example per eye
        get_left_patient_ids_prio = True
        get_right_patient_ids_prio = True
        filtered_dr_level = [1, 2]
        self.patient_target = []
        self.patient_target_5_classes = []
        patient_ids_prio = []
        for patient_id in patient_ids:
            if self._filter_target:
                # Update get_left/right_ids_prio depending on left/right DR level target
                get_left_patient_ids_prio = self._labels[patient_id]['left_DR_level'] not in filtered_dr_level
                get_right_patient_ids_prio = self._labels[patient_id]['right_DR_level'] not in filtered_dr_level
            if get_left_patient_ids_prio:
                patient_ids_prio += [self._get_prio(patient_id, 'left', 'right')]
                self.patient_target += [self._labels[patient_id][f'left_{self._target_type}']]
                self.patient_target_5_classes += [self._labels[patient_id]['left_DR_level']]
            if get_right_patient_ids_prio:
                patient_ids_prio += [self._get_prio(patient_id, 'right', 'left')]
                self.patient_target += [self._labels[patient_id][f'right_{self._target_type}']]
                self.patient_target_5_classes += [self._labels[patient_id]['right_DR_level']]

        self._patient_ids = patient_ids_prio
        self.img_dir = None

    def __len__(self):
        return len(self._patient_ids)

    def __getitem__(self, index):
        """
        Returns
        -------
        sample : dict
            Sample (meta-)data dictionary. Contain the input images,
            the masks for the view and the appropriate targets.
        """

        eye_prio = self._patient_ids[index]['eye_prio']
        patient = self._labels[self._patient_ids[index]['id']]

        # Set targets taken from the primary eye
        sample = {}
        sample['target'] = patient[f"{eye_prio['primary']}_{self._target_type}"]
        if self._target_type == "screening_level":
            sample['target'] = float(sample['target'])

        # Set primary eye views
        views, mask = self._process_image(patient, eye_prio['primary'])
        sample['inputs'] = [views]
        sample['masks'] = [mask]

        # Set secondary eye if needed
        if self._use_other_eye:
            views, mask = self._process_image(patient, eye_prio['secondary'])
            sample['inputs'] += [views]
            sample['masks'] += [mask]

        return sample

    def _process_image(self):
        raise NotImplementedError

    def _get_prio(self, patient_id, primary, secondary):
        return {'id': patient_id, 'eye_prio': {'primary': primary, 'secondary': secondary}}


class Kaggle2015(RetinoDataset):
    """
    Kaggle Dataset for diabetic retinopathy detection (kaggle.com/c/diabetic-retinopathy-detection/data)
    provide a large set of high-resolution retina images taken under a variety
    of imaging conditions. A left and right field is provided for every subject.

    A clinician has rated the presence of diabetic retinopathy in each image
    on a scale of 0 to 4, according to the following scale:
    - 0: No DR
    - 1: Mild
    - 2: Moderate
    - 3: Severe
    - 4: Proliferative DR

    Note: Eyepacs use ETDRS grading (https://www.eyepacs.org/consultant/Clinical/grading/EyePACS-DIGITAL-RETINAL-IMAGE-GRADING.pdf).

    The task is to create an automated analysis system capable of assigning a
    score based on this scale.

    The images in the dataset come from different models and types of cameras,
    which can affect the visual appearance of left vs. right. Some images are
    shown as one would see the retina anatomically (macula on the left, optic
    nerve on the right for the right eye). Others are shown as one would see
    through a microscope condensing lens (i.e. inverted, as one sees in a
    typical live eye exam). There are generally two ways to tell if an image
    is inverted:
    - It is inverted if the macula (the small dark central area) is slightly
      higher than the midline through the optic nerve. If the macula is lower
      than the midline of the optic nerve, it's not inverted.
    - If there is a notch on the side of the image (square, triangle, or circle)
    then it's not inverted. If there is no notch, it's inverted.

    Like any real-world data set, there is noise in both the images and labels.
    Images may contain artifacts, be out of focus, underexposed, or overexposed.
    A major aim of this competition is to develop robust algorithms that can
    function in the presence of noise and variation.

    We preprocessed the images (crop, extand and resize into 512x512 and 1024x1024)
    given three datasets. We created a .json for the meta-data and consider the
    train (35116 graded images: left and right field of 17558 patients) as
    training set, the public test (10906 graded images: left and right field of
    5453 patients) as validation set and the private test (42670 graded images:
    left and right field of 21335 patients) as holdout test set. Note that we
    removed 5 patients from the train set due to the impossibility to perform
    automatic preprocessing (the images of those patients being too dark).

    As a target we use the target of the competition (the 5 classes presented
    above: no DR, mild, moderate, severe and proliferative DR) as well as
    another target for the screening. For the screening we have created two
    classes: referable (i.e., we merged the no DR and mild grades) and
    non-referable (i.e., we merged the moderate, severe and proliferative DR
    grades) for an ophthalmologic examination.

    The distribution of data across the 5 and 2 classes is shown in the tables
    below:

    | classes        | 0               | 1             | 2              | 3           | 4           |
    |----------------|-----------------|---------------|----------------|-------------|-------------|
    | training set   | 25,805 (73.5%)  | 2,438 (6.9%)  | 5,292 (15.1%)  | 873 (2.5%)  | 708 (2.0%)  |
    | validation set | 8,130 (74.5%)   | 720 (6.6%)    | 1579 (14.5%)   | 237 (2.2%)  | 240 (2.2%)  |
    | test set       | 31,403 (73.6%)  | 3,042 (7.1%)  | 6,282 (14.7%)  | 977 (2.3%)  | 966 (2.3%)  |

    | classes        | 0               | 1             |
    |----------------|-----------------|---------------|
    | training set   | 28,243 (80.4%)  | 6,873 (19.6%) |
    | validation set | 8,850 (81.1%)   | 2,056 (18.9%) |
    | test set       | 34,445 (80.7%)  | 8,225 (19,3%) |

    Note: the best results of the kaggle challenge on the private test is 0.84957 QuadraticWeightedKappa.

    """

    def __init__(self, data_folder, cfg, split='train', transform=ToTensor()):
        check_kaggle2015_config(data_folder, cfg, split, transform)
        super(Kaggle2015, self).__init__(data_folder, cfg, split, transform)

        # Path to image folder. Resolution can be 512 or 1024.
        folder_split = ('train' if split == 'train' else 'test')
        self.img_dir = os.path.join(data_folder, f"{folder_split}_{self._input_size}")
        if not os.path.isdir(self.img_dir):
            raise ValueError(f"Expected image folder does not exist: {self.img_dir}.")

    def _process_image(self, patient, side, dtype=torch.float32):
        image_path = os.path.join(self.img_dir, patient[f"{side}_image_path"] + ".jpeg")
        with Image.open(image_path) as im_view:
            view = np.array(im_view, dtype=np.float32)

        if self._different_transform_per_view:
            views = []
            for i in range(self._n_views):
                views.append(self._transform(copy.deepcopy(view)).to(dtype=dtype))
        else:
            views = [self._transform(view).to(dtype=dtype)] * self._n_views

        mask = torch.ones(self._n_views, dtype=dtype)
        return views, mask


class Fake(Dataset):
    """
    Fake data set of sample_size*3*input_size*input_size tensors of 1 with
    multiple-view option with output class equal to 1.
    """

    def __init__(self, data_folder, cfg, split='train', transform=ToTensor()):
        """
        Parameters
        ----------
        data_folder : str
            Data folder.
        cfg : dict
            resolution : int, optional
                Input size in [512, 1024].
            sample_size : int, optional
                Number of elements to use as sample size, for debugging purposes only.
                If -1, use all samples.
            use_other_eye: bool
                Whether or not to return the secondary eye to make the prediction.
            target_name: str
                Can be 'screening_level' or 'DR_level'. Changes the type of the targets.
            train: dict
                n_views : int
                    Number of views.
                augmentation : dict
                    different_transform_per_view: bool
                        Whether or not to apply a different transformation per view.
            eval: dict
                n_views : int
                    Number of views.
                apply_train_augmentations: bool
                    Whether or not to apply train transform.
        split : str, optional
            Should be in ['train', 'valid', 'test']. Default = 'train'
        transform : callable, optional
            Transform to be applied on a sample. Default = None.
        """

        self._transform = transform
        if split == 'train':
            self._n_views = cfg['train']['n_views']
        else:
            self._n_views = cfg['eval']['n_views']
        self._different_transform_per_view = cfg['train']['augmentation']['different_transform_per_view']
        self._use_other_eye = cfg['use_both_eyes']
        self._target_type = cfg['target_name']

        self._input_size = cfg['resolution']
        self._sample_size = (1000 if cfg['sample_size'] == -1 else cfg['sample_size'])

        # Define num_class
        if self._target_type == "screening_level":
            self.num_class = 1
            self.raw_labels = list(range(self.num_class + 1))
        elif self._target_type == "DR_level":
            self.num_class = 5
            self.raw_labels = list(range(self.num_class))

        self.patient_target = None
        self.patient_target = [1] * self._sample_size
        self.patient_target_5_classes = self.patient_target

        self._patient_ids = None
        self.img_dir = None
        self._fake_image = np.array(Image.new("RGB", size=(self._input_size, self._input_size), color=(1, 1, 1)))

        print(f"# of examples: {self._sample_size}")

    def __len__(self):
        return self._sample_size

    def __getitem__(self, index):
        """
        Returns
        -------
        sample : dict
            Sample (meta-)data dictionary. Contain the input images,
            the masks for the view and the appropriate targets.
        """

        sample = {}
        # Set target
        sample['target'] = 1
        if self._target_type == 'screening_level':
            sample['target'] = float(sample['target'])

        # Set primary eye views
        views, mask = self._generate_inputs()
        sample['inputs'] = [views]
        sample['masks'] = [mask]

        # Set secondary eye if needed
        if self._use_other_eye:
            sample['inputs'] += sample['inputs']
            sample['masks'] += sample['masks']

        return sample

    def _generate_inputs(self, dtype=torch.float32):
        if self._different_transform_per_view:
            views = []
            for i in range(self._n_views):
                views.append(self._transform(copy.deepcopy(self._fake_image)).to(dtype=dtype))
        else:
            views = [self._transform(self._fake_image).to(dtype=dtype)] * self._n_views
        mask = torch.ones(self._n_views, dtype=torch.float32)
        return views, mask


def _batch_size_check(dataset, batch_size):
    """
    Make sure that the `batch_size` size is not bigger than the dataset.
    If it is, it reduces `batch_size` to the lenght of the dataset.
    This is necessary because if PyTorch data_loader receive a batch size bigger than
    the dataset_size it returns an empty generator.

    Parameters
    ----------
    dataset : Dataset
        PyTorch Dataset object.
    batch_size : int
        Size of mini-batches for on the given dataset.

    Returns
    -------
    batch_size : int
        Size of mini-batches for on the given dataset.
        Update to dataset size if needed.
    """

    dataset_size = len(dataset)
    if dataset_size < batch_size:
        print(f"Warning: batch_size({batch_size}) > than dataset_size({dataset_size})."
              "\nSetting batch_size to dataset_size.")
        return dataset_size
    return batch_size


def _get_loader(data_folder, cfg_data, split, train_mode, transforms):
    """
    Parameters
    ----------
    data_folder : str
        Data folder.
    cfg_data : dict
        resolution : int
            Input size.
        name : str
            Dataset name.
        sample_size : int
            Number of elements to use as sample size, for debugging purposes only.
            If -1, use all samples.
        use_other_eye : bool
            Whether or not to return the secondary eye to make the prediction.
        target_name : str, optional
            Can be 'screening_level' or 'DR_level'. Changes the type of the targets.
        train : dict
            filter_target : str
                Filter out samples with DR_level equal 1 and 2.
            n_views : int
                Number of views.
            augmentation: dict
                apply_prob : float
                    Probability of applying the train transformations.
                rotation_type : str
                    Type of rotation to apply. Could be RightAngle or Any.
                resized_crop : bool
                    Whether or not to apply random resized crop.
                different_transform_per_view: bool
                    Whether or not to apply a different transformation per view.
            batch_size : int
                Train batch size
            accumulated_batch_size : int
                Train accumulated batch size.
            num_workers : int
                Train number of workers.
        eval : dict
            n_views : int
                Number of views.
            apply_train_augmentations: bool
                Whether or not to apply train transform.
            batch_size : int
                Valid batch size
            num_workers : int
                Valid number of workers.
    split : str
        Should be in ['train', 'valid', 'test'].
    train_mode : bool
        A boolean that tells if the loader should be in train or eval mode.
    transforms : callable
        Transform to be applied on a sample.

    Returns
    -------
        dataloader : DataLoader
            Data loader with the appropriate settings of the proper dataset and dataset split.
    """

    dataset_class = eval(cfg_data['name'].title())

    mode = cfg_data[split if train_mode else 'eval']

    dataset = dataset_class(data_folder=data_folder,
                            cfg=cfg_data,
                            split=split,
                            transform=Compose(transforms))

    batch_size = _batch_size_check(dataset, mode['batch_size'])

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=train_mode,
                      num_workers=mode['num_workers'],
                      pin_memory=True,
                      drop_last=train_mode)


def get_dataloader(data_folder, cfg_data, train=True):
    """
    Parameters
    ----------
    data_folder : str
        Data folder.
    cfg_data : dict
        resolution : int
            Input size.
        name : str
            Dataset name.
        sample_size : int
            Number of elements to use as sample size,
            for debugging purposes only. If -1, use all samples.
        use_other_eye : bool
            Whether or not to return the secondary eye to make the prediction.
        target_name : str
            Can be 'screening_level' or 'DR_level'. Changes the type of the targets.
        feature_scaling : str
            Type of feature scaling / run time pre-processing.
        train : dict
            filter_target : str
                Filter out samples with DR_level equal 1 and 2.
            n_views : int
                Number of views.
            augmentation: dict
                apply_prob : float
                    Probability of applying the train transformations.
                rotation_type : str
                    Type of rotation to apply. Could be RightAngle or Any.
                resized_crop : bool
                    Whether or not to apply random resized crop.
                different_transform_per_view: bool
                    Whether or not to apply a different transformation per view. Default = False.
            under_sampling_ratio : list of float
                Under sampling ratio for each class. Must sum to 1. Can be None.
            batch_size : int
                Train batch size
            accumulated_batch_size : int
                Train accumulated batch size.
            num_workers : int
                Train number of workers.
        eval:
            n_views : int
                Number of views.
            apply_train_augmentations: bool
                Whether or not to apply train transform.
    train : boolean, optional
        A boolean that tells if the loader should be in train or eval mode.

    Returns
    -------
    if train:
        loaders : dict
            train : DataLoader
                Train data loader.
            valid : DataLoader
                Validation data loader.
            train_exact : DataLoader
                Dataloader used to compute exact stats on train set.
    else:
        loaders : dict
            test : DataLoder
                Test data loader.
    """

    data_stats = DATASETS_MEAN_STD[f"{cfg_data['name']}_{cfg_data['resolution']}"]

    if cfg_data['feature_scaling'] == 'MaskedStandardization':
        valid_transform = [ToTensor(),
                           MaskedStandardization(data_stats['mean'], data_stats['std'])]
    elif cfg_data['feature_scaling'] == 'GreenChannelOnly':
        valid_transform = [ToTensor(),
                           GreenChannelOnly()]
    elif cfg_data['feature_scaling'] == 'GreenChannelOnlyMaskedStandardization':
        valid_transform = [ToTensor(),
                           MaskedStandardization(data_stats['mean'], data_stats['std']),
                           GreenChannelOnly()]
    elif cfg_data['feature_scaling'] == 'KaggleWinner':
        valid_transform = [KaggleWinner(image_size=cfg_data['resolution']),
                           ToTensor()]
    elif cfg_data['feature_scaling'] == 'None':
        valid_transform = [ToTensor()]
    else:
        raise ValueError("Unknown feature_scaling. Must be one in [None, MaskedStandardization, GreenChannelOnly, GreenChannelOnlyMaskedStandardization, KaggleWinner]")

    # Define train transformations.
    if cfg_data['train']['augmentation']['rotation_type'] == 'RightAngle':
        # Choose a transformation among 7. Possible transformation are rotations
        # with angles 90°, 180°, 270°; horizontal or vertical flip; or horizontal
        # flip plus a rotation with an angle of either 90° or 270°.
        rotations = [Rotate(90),
                     Rotate(180),
                     Rotate(270),
                     RandomVerticalFlip(p=1.0),
                     RandomHorizontalFlip(p=1.0),
                     RandomHorizontalFlipRotate(90),
                     RandomHorizontalFlipRotate(270)]
        transforms = [RandomApplyChoice(rotations)]
    elif cfg_data['train']['augmentation']['rotation_type'] == 'Any':
        transforms = [RandomRotation((-180, +180)), RandomHorizontalFlip(p=0.5)]
    else:
        raise ValueError("Train augmentation `rotation_type` should be RightAngle or Any.")

    if 'resized_crop' in cfg_data['train']['augmentation'] and cfg_data['train']['augmentation']['resized_crop']:
        transforms += [RandomResizedCrop(size=cfg_data['resolution'], scale=(0.9, 1.), ratio=(1., 1.), interpolation=Image.NEAREST)]

    train_transform = valid_transform + [RandomApply(transforms, p=cfg_data['train']['augmentation']['apply_prob'])]

    if cfg_data['eval']['apply_train_augmentations']:
        # Apply train transformation during evaluation
        valid_transform = copy.deepcopy(train_transform)

    loaders = {}
    if train:
        # Train dataset loader
        loaders['train'] = _get_loader(data_folder, cfg_data, 'train', True, train_transform)
        loaders['valid'] = _get_loader(data_folder, cfg_data, 'valid', False, valid_transform)
        loaders['train_exact'] = _get_loader(data_folder, cfg_data, 'train', False, valid_transform)

    else:
        # Test dataset loader
        loaders['test'] = _get_loader(data_folder, cfg_data, 'test', False, valid_transform)

    return loaders


class UndersampleBalancedBatchSampler(Sampler):

    def __init__(self, dataset, ratio, batch_size):
        """
        Sampler to get batch with given class ratio.

        Parameters
        ----------
        dataset : Dataset
            dataset
        ratio : list of float
            Under sampling ratio for each class. Must sum to 1.
        batch_size : int
            Batch size.
        """

        assert sum(ratio) == 1, f"{ratio} should sum to 1."
        print(f"Trying to undersample dataset with ratio: {ratio}")

        self.raw_labels = dataset.raw_labels
        class_count = Counter(dataset.patient_target)

        potential_num_batch = []
        self.index_per_class = {}
        self.batch_class_size = {}
        for cl in self.raw_labels:
            self.index_per_class[cl] = list(np.argwhere(np.array(dataset.patient_target) == cl).flatten())
            self.batch_class_size[cl] = int(round(batch_size * ratio[cl]))
            potential_num_batch.append(int(class_count[cl] / self.batch_class_size[cl]))

        if sum(self.batch_class_size.values()) < batch_size:
            diff = batch_size - sum(self.batch_class_size.values())
            for i in range(diff):
                self.batch_class_size[random.choice(self.raw_labels)] += 1
            potential_num_batch = []
            for cl in self.raw_labels:
                potential_num_batch.append(int(class_count[cl] / self.batch_class_size[cl]))

        elif sum(self.batch_class_size.values()) > batch_size:
            diff = sum(self.batch_class_size.values()) - batch_size
            for i in range(diff):
                v = self.batch_class_size[random.choice(self.raw_labels)]
                if v != 1:
                    self.batch_class_size[random.choice(self.raw_labels)] -= 1

        assert sum(self.batch_class_size.values()) == batch_size
        batch_ratio = [self.batch_class_size[k] / batch_size for k in self.raw_labels]
        print(f"Undersampling ratio used: {batch_ratio}")

        self.batch_size = batch_size
        self.num_batch = min(potential_num_batch)
        self.num_samples = self.num_batch * self.batch_size
        assert self.num_samples <= len(dataset.patient_target)
        print(f"# of samples used per epoch: {self.num_samples}")

    def __iter__(self):
        sample = []
        index_per_class = copy.deepcopy(self.index_per_class)
        for i in range(self.num_batch):
            for cl in self.raw_labels:
                idx = random.sample(index_per_class[cl], self.batch_class_size[cl])
                for elem in idx:
                    index_per_class[cl].remove(elem)
                sample += idx
        return iter(sample)

    def __len__(self):
        return self.num_samples


def update_data_loader_sampler(data_loader, ratio):
    """
    Update data loader sampler.

    Parameters
    ----------
    data_loader : DataLoader
        DataLoader
    ratio : list of float
        Under sampling ratio for each class. Must sum to 1.

    Returns
    -------
    updated_data_loader : DataLoader
        Updated dataloader
    """

    dataset = data_loader.dataset
    sampler = UndersampleBalancedBatchSampler(dataset, ratio, data_loader.batch_size)

    updated_data_loader = DataLoader(dataset,
                                     batch_size=data_loader.batch_size,
                                     shuffle=False,
                                     num_workers=data_loader.num_workers,
                                     pin_memory=True,
                                     drop_last=data_loader.drop_last,
                                     sampler=sampler)
    return updated_data_loader
