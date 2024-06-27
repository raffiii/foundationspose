# FoundationPose with realsense
This repo is a fork of [FoundationPose](https://github.com/NVlabs/FoundationPose) with the aim to use the Pose estimation method described there for a live depth camera feed from a realsense camera.
## Setup & run
The preferred method to run this is docker. 
Similar to the original FoundationPose repo, there is a shell script to start docker correctly:

```sh
cd realsense
./run_container.sh
```

Then you need to build some missing dependencies, to do so run inside the docker container.
This is only necessary when running it for the first time.
```sh
bash build_all.sh
```


To run the live estimation, remove the object from the camera and run inside the container:

```sh
python run_realsense.py
```
When prompted, place the object in the camera frame and hit enter. This allows to estimate a rough mask required for the first frame of the estimator.
## Implementation notes
- The image is based on the FoundationPose docker, because it was easiet to get this running compared to installing it directly.
- The code is combined from FoundationPose examples and pyrealsense2 examples to get the desired behavior, sometimes there were type conversions necessary.
- In order to align the depth from the tested camera with what the model expects, it is divided to be in the range up to $4.0$ (only empirically tested)
- The camera intrinsics are obtained from realsense and (hopefully correctly) converted to the expected matrix form.
- Before the first estimation, there are $2$ images captured, where the user is prompted to place the object in the second image in order to calculate a mask by the difference. 
- By default, the object is a banana in the `assets/banana/ref_views_16` directory (Reference images from YCB dataset). This can be changed with the `--ref_view_dir` and the `ob_id` flag.