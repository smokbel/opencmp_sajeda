## ParaView Tips and Tricks

*By Elizabeth Monte*



ParaView is a data visualisation and post-processing software. This document lists out some tips/tricks I've learned. Other people in the group (Tommy especially) have more ParaView experience than me, so ask around if you get stuck. All instructions below assume you are working in the GUI. 



****



**Renaming Variables**

It may be necessary to rename variables if you are working with multiple datasets that use the same variable names.

1. Apply a calculator to the dataset with the variable to be renamed.
2. Set "Result Array Name" to the new variable name.
3. In the calculator input just add the variable.



**Combining Multiple Datasets**

Datasets must be combined in order to be used as inputs to the same filters.

1. Select all datasets to combine.
2. Apply an append attributes filter.



**Extracting Surface Vectors**

Surface vectors are needed for various filters like surface LICs.

1. Apply a calculator to the dataset of interest.
2. Construct a vector input as "X \* iHat + Y \* jHat + Z \* kHat". X, Y, and Z can just be the scalar components of some vector field in the dataset or can be any scalar values.



**Surface LICs**

Surface LICs are the preferred way to visualise flow patterns.

1. Ensure the surface LICs plugin is loaded by checking "Tools" $\rightarrow$ "Manage Plugins" $\rightarrow$ "SurfaceLIC".
2. Ensure you have the surface vectors for your vector field of interest.
3. Turn on visualisation for your vector field of interest. Under "Representation" select "Surface LIC". Note that if you just loaded the surface LICs plugin you may need to reload your vector field before "Surface LIC" will show up as an option.



**Animations**

Videos of time data are a nice way to show simulation results.

1. Load your time-varying dataset.

2. Use "File" $\rightarrow$ "Save Animation" to save a set of images of your time-varying dataset.

3. On Linux you can use ffmpeg to convert the set of images into a video.

   `ffmpeg -framerate 20 -i velocity.%04d.png -vcodec libx264 -y -an velocity.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"`

ParaView can also export directly to .avi, although my understanding is that exporting to .png and then converting into a video gives a better end result. If you do prefer to export to video directly James wrote a script to downsize video files to a user-specified size ("vid_file_size.sh" in Files/). Run it as `./vid_file_size.sh VIDEO_NAME FILE_SIZE` passing in the name of the .avi file and the desired final size in MB (must be an integer).







