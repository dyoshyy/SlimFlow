sampling.cifar10:
	python ./image_sampling.py \
		--config ./configs/rectified_flow/cifar10_rf_gaussian.py \
		--sampling_dir "./ckpts/CIFAR_2_rectified_flow_100000edm_flip_warmup_300000_28m.pth" \
		--config.eval.batch_size 64 \
		--config.model.nf 128 --config.model.num_res_blocks 2 --config.data.image_size 32 --config.model.ch_mult '(1, 2, 2)'

sampling.imagenet64:
	python ./image_sampling.py \
		--config ./configs/rectified_flow/cifar10_rf_gaussian.py \
		--sampling_dir "./ckpts/ImageNet_2_rectified_flow_400000edm_flip_warmup_300000_44m.pth" \
		--config.eval.batch_size 64 \
		--config.model.name DhariwalUNet --config.model.nf 128 --config.model.num_res_blocks 2 --config.model.ch_mult '(1, 2, 2, 2)' --config.data.num_classes 1000 --config.data.image_size 64 --config.model.attn_resolutions '32, 16'
	
sampling.ffhq:
	python ./image_sampling.py \
		--config ./configs/rectified_flow/cifar10_rf_gaussian.py \
		--sampling_dir "./ckpts/FFHQ_2_flow_200000edm_flip_warmup_300000_30m_checkpoint_15.pth" \
		--config.eval.batch_size 64 \
		--config.model.nf 128 --config.model.num_res_blocks 2 --config.data.image_size 64 --config.model.ch_mult '(1, 2, 2)'