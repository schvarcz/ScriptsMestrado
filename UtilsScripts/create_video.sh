currentDir=`pwd`

cd $1

avconv -framerate 10  -i fig_%06d.png -b:v 1000k ../out.mp4

cd $currentDir
