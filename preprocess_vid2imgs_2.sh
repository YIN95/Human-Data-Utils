#!/bin/bash
echo "Convert Videos to Images ..."
datafolder=/mnt/md0/yinw/project/data/totalcapture
for subject in 'S4' 'S5'
do
    echo $subject

    for action in 'acting3' 'freestyle1' 'freestyle3' 'rom3' 'walking2'
    do
        videofolder=$datafolder/$subject/video/$action
        savefolder=$datafolder/$subject/images/$action/

        filelist=`ls $videofolder|grep -i '.*mp4'`
        echo $video
        for file in $filelist
        do 
            # echo $file \n
            video_path=$videofolder/${file}
            echo $video_path \n
            python preprocess_vid2imgs.py --videoPath $video_path --savePath $savefolder
        done
    done
done
