GPU based optical flow extraction in OpenCV
====================
### Features:
* OpenCV wrapper for Real-Time optical flow extraction on GPU
* Automatic directory handling using Qt
* Allows saving of optical flow to disk, 
  * either with clipping large displacements 
  * or by adaptively scaling the displacements to the radiometric resolution of the output image

### Dependencies
* [OpenCV 2.4](http://opencv.org/downloads.html)
* [Qt 5.4](https://www.qt.io/qt5-4/)
* [cmake](https://cmake.org/)

### Installation
1. `mkdir -p build`
2. `cd build`
3. `cmake ..`
4. `make`
5. `sudo make install`

### Usage:
```
./compute_flow [OPTION]...
```

Available options:
* `--input-dir`: directory containing video files
* `--output-dir`: directory to dump frames and flow to
* `--start-video`: start with video number in `vid_path` directory structure
* `--gpu-id`: use this GPU ID
* `--method`: use this flow method Brox = 0, TVL1 = 1
* `--step`: specify the number of frames between sampled frames used to compute
  optical flow
* `--min-size=256`: defines the smallest side of the frame for optical flow computation
* `--output-size=256`: defines the smallest side of the frame for saving as .jpeg 
* `--clip-flow/--clip-flow=false`: defines whether to clip the optical flow larger than
  [-20 20] pixels and maps the interval [-20 20] to  [0 255] in grayscale image
  space. If no clipping is performed the mapping to the image space is achieved
  by finding the frame-wise minimum and maximum displacement and mapping to [0
  255] via an adaptive scaling, where the scale factors are saved as a binary
  file to `out_path`.

#### Docker

```
$ docker run --runtime=nvidia --rm -t \
    --volume /path/to/video-dir:/video \
    --volume  /path/to/output:/output \
    willprice/gpu_flow:latest
```
