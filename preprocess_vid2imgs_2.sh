#!/bin/bash
echo "Convert Videos to Images ..."
datafolder=/media/ywj/Data/totalcapture/totalcapture

# ------------------------------------------
tempfifo=$$.fifo
trap "exec 1000>&-;exec 1000<&-;exit 0" 2
mkfifo $tempfifo
exec 1000<>$tempfifo
rm -rf $tempfifo

for ((i=1; i<=4; i++))
do
    echo >&1000
done

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
            read -u1000
            python preprocess_vid2imgs.py --videoPath $video_path --savePath $savefolder
            echo >&1000
        done
    done
done
