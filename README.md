Utilised:
- [bmartacho/UniPose](https://github.com/bmartacho/UniPose)
- [RobertS312312/UniPose](https://github.com/RobertS312312/UniPose) (a fork of bmartacho/UniPose)

# Command line usage

## Install dependencies

```
$ pip install -r requirements.txt
```

`-r, --requirement <file>`:
Install from the given requirements file. This option can be used multiple times.

## Arguments

- pretrained
    * If the model is pretrained, specify the path to the pretrained weights from where they can be loaded with torch.load()
    * '/PATH/TO/WEIGHTS'
    * default=None
- dataset
    * choices=['LSP', 'MPII'], default='LSP'
- train_dir
    * default='/PATH/TO/TRAIN'
- val_dir
    * default='/PATH/TO/LSP/VAL'
- model_name (default=None)
    * Only used as a filename to save the torch model.
Added:
- test
    * bool, default=False
    * If True, the model will do testing instead of training.
    * Refactored from the original code commenting out way.
Removed from the original:
- model_arch (default='unipose')
    * Never actually used, only passed on without utilizing.

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