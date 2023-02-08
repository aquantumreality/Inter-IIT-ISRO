# TMC-Super-resolution

Pretrained Weights for Super-Resolution model (SRUN): (https://drive.google.com/file/d/1jDtWT_fbT9O2xmU-Mb5ycEl6ARF0Q5FZ/view?usp=share_link).

Pretrained Weights for SORTN model (required to run the code): (https://drive.google.com/file/d/15ImRmGoORsCLSIy4tMJxtbH9NXLXGvVV/view?usp=share_link). 

## Train
**Command**:
```bash
gdown --no-cookies 1oa8qAHleja_HelWXiq24Ombf0UkFSMcB

pip3 install -r requirements.txt

./run_script.sh train
```
Downloads the dataset required to perform the training. The data is curated from [browse_derived.zip](https://pradan.issdc.gov.in/ch2/protected/downloadFile/tmc2/browse_derived.zip) from Pradan, ISRO. The method uses both DTM and ortho-derived TMC images for training.

Installs the required libraries from requirements.txt and runs the train script.

Before running the code, respective directories for the dataset and the directories where the results are to be stored must be given in the config.yaml file. New folders will be created for result directories already if they are not present.

## Validation
**Command**:
```bash
gdown --no-cookies 1vWCmwifFBENYm5LrrjJen9aJM4-fcJO4

pip3 install -r requirements.txt

./run_script.sh validate
```

Downloads the dataset required to perform validation. The data is curated from [browse_calibrated_2022.zip](https://pradan.issdc.gov.in/ch2/protected/downloadFile/tmc2/browse_calibrated_2022.zip) from Pradan, ISRO.

Before running the code, the directory to the input validation dataset must be given. Please make sure that the TMC images given are in jpeg format.

## Test/Inference

**Command**:

```
pip3 install -r requirements.txt

./run_script.sh inference
```

Before running the code, the directory to input TMC images must be given (Inference Images by default). Any number of TMC images can be put in the folder and the model would be able to generate super-resolved SAR images for each of them. Note that the TMC images must be in jpeg format.

## Super_Resolution_Demo jupyter notebook
This notebook contains the same commands required for validation and inference.

## Solution Development
* The model is trained on around 18000 TMC-ortho pairs of patches. Patch size is 400x400.
* Model: SORTN (Present in next branch) is the generator using to obtain the ortho-derived image from TMC.
* SRUN: Responsible for super-resolving the TMC image, the architecture is similar to U-Net along with spatial attention modules. 
* Loss Functions:
- (Content Loss + 0.1 x Evaluation Loss) is used for SRUN.
- cGAN loss + 100 * L1 loss between optical ground truth and optical generated is used for SORTN.

## Results
(a.) Spacenet 6 dataset
![Screen Shot 1944-02-23 at 3 05 57 PM](https://user-images.githubusercontent.com/82506345/168256472-910eadd5-8345-4a6c-8bb4-84dfb5758c45.png)



(b.) Sentinel -1 

![imgonline-com-ua-twotoone-RgxpMML7YNeRf](https://user-images.githubusercontent.com/82506345/168259244-e30333f6-6dff-4788-891d-23eff516af76.jpeg)


