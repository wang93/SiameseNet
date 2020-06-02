
gpus=$1
feats=$2
model_name=$3
exp_name=$4

python main_reid.py train  --dataset=market1501 --test_batch=256 --workers=32  \
       --model_name=$model_name --exp_name=$exp_name --train_phase_num=2 --eval_phase_num=2 --gpus=$gpus --feats=$feats --check_element_discriminant=train

python main_reid.py train  --dataset=market1501 --test_batch=256 --workers=32  \
       --model_name=$model_name --exp_name=$exp_name --train_phase_num=2 --eval_phase_num=2 --gpus=$gpus --feats=$feats --check_element_discriminant=test

#python main_reid.py train  --dataset=market1501 --test_batch=256 --workers=32  \
#       --model_name=$model_name --exp_name=$exp_name --train_phase_num=2 --eval_phase_num=2 --gpus=$gpus --feats=$feats --check_discriminant=train

#python main_reid.py train  --dataset=market1501 --test_batch=256 --workers=32  \
#       --model_name=$model_name --exp_name=$exp_name --train_phase_num=2 --eval_phase_num=2 --gpus=$gpus --feats=$feats --check_discriminant=test

python main_reid.py train  --dataset=market1501 --test_batch=256 --workers=32  \
       --model_name=$model_name --exp_name=$exp_name --train_phase_num=2 --eval_phase_num=2 --gpus=$gpus --feats=$feats --check_pair_effect=train

python main_reid.py train  --dataset=market1501 --test_batch=256 --workers=32  \
       --model_name=$model_name --exp_name=$exp_name --train_phase_num=2 --eval_phase_num=2 --gpus=$gpus --feats=$feats --check_pair_effect=test