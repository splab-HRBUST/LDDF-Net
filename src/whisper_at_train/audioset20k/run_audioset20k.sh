
#!/bin/bash
#SBATCH -p a5
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --qos regular
#SBATCH --mem=48000
#SBATCH --job-name="as20k"
#SBATCH --output=./log/%j_as20k.txt

set -x

#########################################################
# 环境
#########################################################

source /data/sls/scratch/yuangong/whisper-a/venv-a5/bin/activate

export TORCH_HOME=../../pretrained_models

#########################################################
# 训练超参数
#########################################################

lr=2e-5

batch_size=24

epoch=30

freqm=0
timem=10
mixup=0.5

#########################################################
# 模型
#########################################################

model=whisper-high-ghost
model_size=large-v1

cuda=3
tdim=25

#########################################################
# AudioSet
#########################################################

dataset=as-bal
bal=bal
n_class=527

#########################################################
# scheduler
#########################################################

CosineAnnealingLR=False
CosTmax=15
Cosmin=2e-5

lrscheduler_start=5
lrscheduler_decay=0.9
lrscheduler_step=1

#########################################################
# 其它
#########################################################

wa=False
wa_start=16
wa_end=30

lr_adapt=False
warmup=True
label_smooth=0

#########################################################
# 数据路径（修改这里）
#########################################################

tr_data=/hy-tmp/sml/EfficentWhisper1/src/whisper_at_train/audioset20k/audioset_train.json
te_data=/hy-tmp/sml/EfficentWhisper1/src/whisper_at_train/audioset20k/audioset_test.json
label_csv=/hy-tmp/sml/EfficentWhisper1/src/whisper_at_train/audioset20k/class_labels_indices.csv

#########################################################
# whisper 特征根目录（修改这里）
#########################################################

train_tar_path=/hy-tmp/sml/EfficentWhisper1/src/whisper_at_train/audioset20k/data/whisper_

eval_tar_path=/hy-tmp/sml/EfficentWhisper1/src/whisper_at_train/audioset20k/data/whisper_

#########################################################
# 实验目录
#########################################################
base_exp_dir=./exp/as20k-${model}-${model_size}-lr${lr}-bs${batch_size}
#########################################################
# 自动检查是否已经训练完成
#########################################################

if [ -f "${base_exp_dir}/stats_${epoch}.pickle" ]; then
    echo "Training already completed."
    exit
fi
mkdir -p ${base_exp_dir}

#########################################################
# 开始训练
#########################################################

CUDA_VISIBLE_DEVICES=${cuda} python -W ignore ../runs.py \
    --model ${model} \
    --dataset ${dataset} \
    --data-train ${tr_data} \
    --data-val ${te_data} \
    --exp-dir ${base_exp_dir} \
    --label-csv ${label_csv} \
    --n_class ${n_class} \
    --lr ${lr} \
    --n-epochs ${epoch} \
    --batch-size ${batch_size} \
    --save_model False \
    --freqm ${freqm} \
    --timem ${timem} \
    --mixup ${mixup} \
    --bal ${bal} \
    --model_size ${model_size} \
    --label_smooth ${label_smooth} \
    --lrscheduler_start ${lrscheduler_start} \
    --lrscheduler_decay ${lrscheduler_decay} \
    --lrscheduler_step ${lrscheduler_step} \
    --loss BCE \
    --metrics mAP \
    --warmup ${warmup} \
    --wa ${wa} \
    --wa_start ${wa_start} \
    --wa_end ${wa_end} \
    --lr_adapt ${lr_adapt} \
    --num-workers 8 \
    --CosineAnnealingLR ${CosineAnnealingLR} \
    --CosTmax ${CosTmax} \
    --Cosmin ${Cosmin} \
    --tdim ${tdim} \
    --train_tar_path ${train_tar_path} \
    --eval_tar_path ${eval_tar_path}

echo "================ Training Finished ================"