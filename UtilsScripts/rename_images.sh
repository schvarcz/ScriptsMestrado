
current_folder=`pwd`
cd $1
i=0
for f in *.png; do
    mv $f `printf "I1_%06d.png" $i`
    i=$(($i+1))
done

cd $current_folder

