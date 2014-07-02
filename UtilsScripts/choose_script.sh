
folder=~/Dissertacao/datasets
src_folder=$folder/motox/VID_20140617_162058756_GRAY/
dst_folder=$folder/motox/VID_20140617_162058756_GRAY_ESCOLHA/

mkdir $dst_folder

for i in `seq 0 5 623`; do
    cp $src_folder`printf "I1_%06d.png" $i` $dst_folder`printf "I1_%06d.png" $i`
done

sh rename_images.sh $dst_folder
