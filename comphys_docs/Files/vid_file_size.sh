#!/bin/bash

# run this script to reduce a video file to a desired file size.
#   example:   ./script original_video.mp4 size
#   size = an integer in MB

bitrate="$(awk "BEGIN {print int($2 * 1024 * 1024 * 8 / $(ffprobe \
    -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$1" \
) / 1000)}")k"
ffmpeg \
    -y \
    -i "$1" \
    -c:v libx264 \
    -preset medium \
    -b:v $bitrate \
    -pass 1 \
    -an \
    -f mp4 \
    /dev/null \
&& \
ffmpeg \
    -i "$1" \
    -c:v libx264 \
    -preset medium \
    -b:v $bitrate \
    -pass 2 \
    -an \
    "${1%.*}-$2mB.mp4"
