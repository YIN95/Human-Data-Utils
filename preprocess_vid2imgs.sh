#!/bin/bash
echo "Convert Videos to Images ..."
datafolder=/media/ywj/Data/totalcapture/totalcapture
for subject in 'S1' 'S2' 'S3'
do
    echo $subject

    for action in 'acting1' 'acting2' 'acting3' 'freestyle1' 'freestyle2' 'freestyle3' 'rom1' 'rom2' 'rom3' 'walking1' 'walking2' 'walking3'
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