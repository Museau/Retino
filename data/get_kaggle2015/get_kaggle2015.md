# Kaggle 2015 Dataset

[Kaggle Dataset for diabetic retinopathy detection](kaggle.com/c/diabetic-retinopathy-detection/data) provide a large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject.

A clinician has rated the presence of diabetic retinopathy in each image
on a scale of 0 to 4, according to the following scale:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

Note: [Eyepacs use ETDRS grading](https://www.eyepacs.org/consultant/Clinical/grading/EyePACS-DIGITAL-RETINAL-IMAGE-GRADING.pdf).

The task is to create an automated analysis system capable of assigning a score based on this scale.

The images in the dataset come from different models and types of cameras, which can affect the visual appearance of left vs. right. Some images are shown as one would see the retina anatomically (macula on the left, optic nerve on the right for the right eye). Others are shown as one would see through a microscope condensing lens (i.e. inverted, as one sees in a typical live eye exam). There are generally two ways to tell if an image is inverted:
- It is inverted if the macula (the small dark central area) is slightly higher than the midline through the optic nerve. If the macula is lower than the midline of the optic nerve, it's not inverted.
- If there is a notch on the side of the image (square, triangle, or circle) then it's not inverted. If there is no notch, it's inverted.

Like any real-world data set, there is noise in both the images and labels. Images may contain artifacts, be out of focus, underexposed, or overexposed. A major aim of this competition is to develop robust algorithms that can function in the presence of noise and variation.

We preprocessed the images (crop, extand and resize into 512x512 and 1024x1024) given three datasets. We created a .json for the meta-data and consider the train (35116 graded images: left and right field of 17558 patients) as training set, the public test (10906 graded images: left and right field of 5453 patients) as validation set and the private test (42670 graded images: left and right field of 21335 patients) as holdout test set. Note that we removed 5 patients from the train set due to the impossibility to perform automatic preprocessing (the images of those patients being too dark).

As a target we use the target of the competition (the 5 classes presented above: no DR, mild, moderate, severe and proliferative DR) as well as another target for the screening. For the screening we have created two classes: referable (i.e., we merged the no DR and mild grades) and non-referable (i.e., we merged the moderate, severe and proliferative DR grades) for an ophthalmologic examination.

The distribution of data across the 5 and 2 classes is shown in the tables below:

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

## Download the data set
To download the data run `download_kaggle2015_data.sh`.

## Preproces the data set
To pre-process the dataset run in order:
1. `preprocess_kaggle2015_train_image_data.sh` -> This will crop, extend and resize to 512, 1024 and 2048 the train images and unzip the train labels;
2. `preprocess_kaggle2015_test_image_data.sh` -> This will crop, extend and resize to 512, 1024 and 2048 the test images;
3. `launch_generate_json_dataset_splits.sh` -> This will create the train, valid and test `.json` meta-data files.
4. `launch_mean_std_dataset.sh` -> This will calculate the mean and standard deviation per channel for the different training set.
