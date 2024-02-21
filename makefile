.PHONY: run_text_mask, run_inpaint

max_frame = 400

#video_name = 文字倾斜
#video_name = 亮背景\ 白色字\ 无描边
video_name = 亮背景\ 白色字\ 黑描边
#video_name = 描边

# input_video = /home/wegatron/win-data/data/subtile_poj/亮背景\ 白色字\ 无描边.mp4
# mask_dir = /home/wegatron/win-data/data/subtile_poj/results/亮背景\ 白色字\ 无描边/mask/
# inpaint_result_dir = /home/wegatron/win-data/data/subtile_poj/results/亮背景\ 白色字\ 无描边/

input_video = /home/wegatron/win-data/data/subtile_poj/$(video_name).mp4
mask_dir = /home/wegatron/win-data/data/subtile_poj/results/$(video_name)/mask/
inpaint_result_dir = /home/wegatron/win-data/data/subtile_poj/results/$(video_name)/

run_text_mask:
	mkdir -p $(mask_dir)
	python seg_text.py $(input_video) $(mask_dir) $(max_frame)

run_inpaint:
	mkdir -p $(inpaint_result_dir)
	python test.py --model e2fgvi_hq --video $(input_video) --mask $(mask_dir)  --ckpt release_model/E2FGVI-HQ-CVPR22.pth --max_frame $(max_frame)

all: run_text_mask run_inpaint