#!/bin/bash
echo "Convert Videos to Images ..."

arg1=$1
arg2=$2
datafolder=${arg1:-1} 
multi_num=${arg2:-4}

datafolder=/media/ywj/Data/totalcapture/totalcapture

# ----------------------------------------------------
tempfifo=$$.fifo
trap "exec 1000>&-;exec 1000<&-;exit 0" 2
mkfifo $tempfifo
exec 1000<>$tempfifo
rm -rf $tempfifo

i=1

while [ ${i} -le ${multi_num} ]
do
    echo >&1000
    echo $i
    i=$(( i+1 ))
done
# ----------------------------------------------------

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
            read -u1000
            video_path=$videofolder/${file}
            echo $video_path \n
            python preprocess_vid2imgs.py --videoPath $video_path --savePath $savefolder
            echo >&1000
        done &
    done
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
            read -u1000
            video_path=$videofolder/${file}
            echo $video_path \n            
            python preprocess_vid2imgs.py --videoPath $video_path --savePath $savefolder 
            echo >&1000
        done &
    done
done
