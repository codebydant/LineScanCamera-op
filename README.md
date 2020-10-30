# Line Scan Camera
This is a simple implementation of a Line Scan Camera from a video of an Area Scan Camera using OpenCV 4.1.2. A line scan camera uses a single line of sensor pixels (effectively one-dimensional) to build up a two-dimensional image. The second dimension results from the motion of the object being imaged. Two-dimensional images are acquired line by line by successive single-line scans while the object moves (perpendicularly) past the line of pixels in the image sensor. For a given field of view, one line scan camera typically provides more resolution than multiple area scan cameras, at a lower cost, without image smear or the redundant processing of frame overlaps.<br>

Difference between a line scan camera and an area scan camera can be found here: 

-	Line Scan and Area Scan Cameras for the Inspection of Pharmaceutical Products:
https://www.youtube.com/watch?v=EzL_3BbEI20

-	Area Scan Camera vs Line Scan Camera:
https://www.youtube.com/watch?v=DkIQl06jloM


## How to use
1. Install dependencies
```
pip install -r requirements.txt
```

2. Run script
```
python3 main.py <scan_mode> <video_file>

example:

python3 main.py column C:/Users/xXx/Downloads/videos/video.mp4
```

3. Output image

You will find the output file in: ```RESULT``` folder (this folder is created automatically) with the same name as the input file.

4. Configuration file (optional)

In this folder you will find a ```config.ini``` file. You can modify the parameters to test differents width ROis in the scanning process.

## Explanation
This program uses two modes: 1. ```COLUMN MODE```: this mode generate a scanned image taking one column pixel per image sequence. This is commonly the process done by a line scan camera. 2. ```WIDTH MODE```: this mode generate a scanned image taking a region of interest (centroid) in the image sequence. This process is more time consuming, but will provide a better ouput resolution.

## State-of-the-art
There is also a C++ implementation of a line scan camera in OpenCV here: https://github.com/ppalasek/linescan. <br><br>

An example video of this is action can be found here:
https://www.youtube.com/watch?v=1X8DVp0Amh8



