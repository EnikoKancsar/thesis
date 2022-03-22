Utilised:
- [bmartacho/UniPose](https://github.com/bmartacho/UniPose)
- [RobertS312312/UniPose](https://github.com/RobertS312312/UniPose) (a fork of bmartacho/UniPose)

# Configuration

The `conf_sample.ini` file contains config data, like paths that are relative to local.

Either rename it to `conf.ini` or copy it.
```
cp conf_sample.ini conf.ini
```
Then fill it in with your local data.

## Prepare Data

### MPII

The Max Planck Insitut Informatik Dataset can be downloaded from [here](http://human-pose.mpi-inf.mpg.de/#download)

The annotations have a `.mat` extension which has to be converted to `.json` for UniPose.

After research this json conversion couldn't be found and also proved to have an annotation structure that differs from the original and isn't documented, therefore I introduced a new conversion based on: [Qi Zhang@StackOverflow](https://stackoverflow.com/a/61074404/13497164).

In your `conf.ini` you have to specify

* where to load from
    * the `.mat` annotation file (`MPII.ANNOTATIONS_MAT`)
    * the folder that contains the images (`MPII.DIR_IMAGES`)
* where to write the train data
    * a directory for the train images (`MPII._DIR_IMAGES_TRAIN`)
    * a file path for the train annotations file (`MPII.ANNOTATIONS_TRAIN`)
* where to write the val data
    * a directory for the val images (`MPII.DIR_IMAGES_VAL`)
    * a file path for the val annotations file (`MPII.ANNOTATIONS_VAL`)

Basically after the conversion you will have the mat annotations file, the json annotations file, the images folder, and a folder and a json annotations file containing the train and val images&annotations only.

After setting up your `conf.ini` run this script.
```
python prepare_mpii.py
```

##### Annotation structure

Original:

* annolist(imgidx) - annotations for image imgidx
    * image
        * name - image filename
    * annorect(ridx) - body annotations for a person ridx
        * .x1, .y1, .x2, .y2 - coordinates of the head rectangle
        * .scale - person scale w.r.t. 200 px height
        * .objpos - rough human position in the image
        * .annopoints.point - person-centric body joint annotations
            * .x, .y - coordinates of a joint
            * id - joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
            * is_visible - joint visibility
    * .vidx - video index in video_list
    * .frame_sec - image position in video, in seconds
* img_train(imgidx) - training/testing image assignment
* single_person(imgidx) - contains rectangle id ridx of sufficiently separated individuals
* act(imgidx) - activity/category label for image imgidx
    * act_name - activity name
    * cat_name - category name
    * act_id - activity id
* vid

After conversion:
```
{
    'act': {
        'act_id': -1,
        'act_name': [],
        'cat_name': []
    },
    'annolist': {
        'annorect': [
            {
                'objpos': {'x': 424, 'y': 382},
                'scale': 2.563847109326139
            },  
            {
                'objpos': {'x': 615, 'y': 426},
                'scale': 1.8672246785001532
            }
        ],
        'frame_sec': [],
        'image': {'name': '001386214.jpg'},
        'vididx': []
    },
    'single_person': [1, 2]}
```

# Command line usage

## Install dependencies

Using Python 3.9, PyTorch 1.10.2.
PyTorch has to be installed separately from the other requirements.
```
$ pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
$ pip install -r requirements.txt
```

`-r, --requirement <file>`:
Install from the given requirements file.

## Arguments

- `--pretrained`
    * If the model is pretrained, specify the path to the pretrained weights from where they can be loaded with torch.load()
    * '/PATH/TO/WEIGHTS'
    * default=None
- `--dataset`
    * choices=['MPII'], default='MPII'
- `--model_name` (default=None)
    * Only used as a filename to save the torch model.
Added:
- `--test`
    * bool, default=False
    * If True, the model will do testing instead of training.
    * Refactored from the original code commenting out way.
Removed from the original:
- `--model_arch` (default='unipose')
    * Passed down without actually being used ever.

## Command

```
$ python main.py --model_name "unipose"
```

# FAQ

## VSC Terminal conda activate not working

Open Windows Power Shell as administrator or Anaconda Prompt.
Go to the folder.
Activate the desired conda environment there.
Type `$ code` which will open VSC in the current folder.
Open the Terminal in VSC.
If the conda environment is not set already, the activate command should work correctly now after being started from the command line.

## `cudnn=True`

> It enables benchmark mode in cudnn.
>
> Benchmark mode is good whenever your input sizes for your network do not vary. This way, `cudnn` will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
>
> But if your input sizes changes at each iteration, then `cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.

[fmassa's answer](https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2)

## `freeze_bn=False`

[Why should you freeze BN layers while fine-tuning](https://stackoverflow.com/questions/63016740/why-its-necessary-to-frozen-all-inner-state-of-a-batch-normalization-layer-when)

[How to freeze BN layers while training](https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/11)

# Notes

## Removing `dataset`

This parameter is actually not used within `Unipose()`.
Except one place in the `Decoder()` where it checks if it's NTID which I don't plan to use.

## Removing `if __name__ == “main”:`

`if __name__ == “main”:` is used to execute some code only if the file was run directly, and not imported. ([Source](https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/))