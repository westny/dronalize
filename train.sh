# apptainer run --nv build/dronalize.sif python train.py --add-name Test --dry-run $1 --use-cuda 1 --num-workers 4
python train.py --add-name Test --dry-run $1 --use-cuda 1 --num-workers 4
