GPU=0
batchsize=16
lr=1e-4
for wirm in 1.0
do
	CUDA_VISIBLE_DEVICES=$GPU python trainval_irm.py --model-name resnet101 --exp-setting visda --source-datasets train --target-datasets validation --batch-size $batchsize --save-dir output/resnet101_visda_batch${batchsize}_lr${lr}_factor_${factor}_irm${wirm} --print-console --warmup-learning-rate $lr --learning-rate $lr --weight-irm $wirm --sm-loss #--in-features 2048
done
