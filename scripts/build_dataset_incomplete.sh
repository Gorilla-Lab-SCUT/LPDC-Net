source config.sh

# Make output directories
mkdir -p $BUILD_PATH


echo " Building Human Dataset for Occupancy Flow Project."
echo " Input Path: $INPUT_PATH"
echo " Build Path: $BUILD_PATH"


echo "Sample points sdf and pointcloud ..."
python compute_incomplete.py $INPUT_PATH \
   --n_proc 8 --resize \
   --partial_pointcloud_size 30000 --start 0.0 --end 1.0 --float16 --radius 0.1
echo "done!"
