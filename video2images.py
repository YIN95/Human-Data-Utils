import hmdata as hd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--videoPath',
        type=str,
        default='/media/ywj/File/C-Code/OpenPose-PyTorch/examples/video.avi',
        help='The path of the video')
    parser.add_argument(
        '--savePath',
        type=str,
        default='/media/ywj/File/C-Code/OpenPose-PyTorch/examples/',
        help='The path of the save video images')
    args = parser.parse_args()

    videoPath = args.videoPath
    savePath = args.savePath

    hd.video2images(videoPath, savePath)
