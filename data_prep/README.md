# MPII Dataset

The Max Planck Insitut Informatik Dataset can be downloaded from [here](http://human-pose.mpi-inf.mpg.de/#download)

The annotations have a `.mat` extension which has to be converted to `.json` for UniPose.

[Deep High Resolution Net Data Preparation for MPII Data](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

The json files can be loaded as Python dictionaries with `json.load()`.

### Annotation structure
```
{
    'center': [594.0, 257.0],
    'image': '015601864.jpg',
    'joints': [
        [620.0, 394.0],
        [616.0, 269.0],
        [573.0, 185.0],
        [647.0, 188.0],
        [661.0, 221.0],
        [656.0, 231.0],
        [610.0, 187.0],
        [647.0, 176.0],
        [637.0201, 189.8183],
        [695.9799, 108.1817],
        [606.0, 217.0],
        [553.0, 161.0],
        [601.0, 167.0],
        [692.0, 185.0],
        [693.0, 240.0],
        [688.0, 313.0]
    ],
    'joints_vis': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'scale': 3.021046
}
```

### Joint ID

* 0 - r ankle, 1 - r knee, 2 - r hip,
* 3 - l hip, 4 - l knee, 5 - l ankle,
* 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top,
* 10 - r wrist, 11 - r elbow, 12 - r shoulder,
* 13 - l shoulder, 14 - l elbow, 15 - l wrist
