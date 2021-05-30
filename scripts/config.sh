ROOT=/data2/lab-tang.jiapeng
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

INPUT_PATH=$ROOT/D-FAUST/scripts/registration_meshes

BUILD_PATH=$ROOT/Humans.build
OUTPUT_PATH=$ROOT/Humans/D-FAUST

NPROC=6
TIMEOUT=180
#N_VIEWS=4

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}
