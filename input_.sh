torchrun --nproc-per-node=$CUDA_DEVICE_COUNT  tools/train.py --cfg test_clip.yaml ./configs/imagenet/optim/adamw.yaml

torchrun --nproc-per-node=4  tools/train.py --cfg ./configs/imagenet/dinov2/test.yaml ./configs/imagenet/optim/adamw.yaml


export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_COUNT=4
export OMP_NUM_THREADS=4
torchrun --nproc-per-node=$CUDA_DEVICE_COUNT --master_port=12345  tools/train.py --cfg ./configs/imagenet/clip/amd_compare.yaml ./configs/imagenet/optim/adamw.yaml

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_COUNT=4
export OMP_NUM_THREADS=4
torchrun --nproc-per-node=$CUDA_DEVICE_COUNT --master_port=6789  tools/train.py --cfg ./configs/imagenet/clip/AMD.yaml ./configs/imagenet/optim/adamw.yaml

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_COUNT=8
export OMP_NUM_THREADS=4
torchrun --nproc-per-node=$CUDA_DEVICE_COUNT --master_port=6789  tools/train.py --cfg ./configs/imagenet/dinov2/amd.yaml ./configs/imagenet/optim/adamw.yaml


export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_COUNT=4
export OMP_NUM_THREADS=4
torchrun --nproc-per-node=$CUDA_DEVICE_COUNT --master_port=6789  tools/train.py --cfg ./configs/imagenet/deit3/amd-masking.yaml ./configs/imagenet/optim/adamw.yaml


export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_COUNT=4
export OMP_NUM_THREADS=4
torchrun --nproc-per-node=$CUDA_DEVICE_COUNT --master_port=12345  tools/train.py --cfg ./configs/imagenet/deit3/amd-wo-masking.yaml ./configs/imagenet/optim/adamw.yaml
