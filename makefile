.PHONY: run_text_mask, run_inpaint



#export PATH=/home/wegatron/opt/anaconda3/envs/envs/zsw/lib/python3.10/site-packages/torch/lib:$PATH
#export LD_LIBRARY_PATH=/home/wegatron/opt/anaconda3/envs/envs/zsw/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

max_frame = 10000
inpaint_board = 0

#video_name = 文字倾斜
#video_name = 亮背景\ 白色字\ 无描边
#video_name = 亮背景\ 白色字\ 黑描边
#video_name = 描边
#video_name = 20240218-160032
video_name = 20240222-094450_m

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
	python test.py --model e2fgvi_hq --video $(input_video) --ckpt release_model/E2FGVI-HQ-CVPR22.pth --max_frame $(max_frame) --inpaint_board $(inpaint_board)

comp_result:
	ffmpeg  -i $(input_video)  -i results/$(video_name)_results.mp4  -filter_complex "[0:v]pad=iw*2:ih*1[a];[a][1:v]overlay=w" results/$(video_name)_res_cmp.mp4

all: run_inpaint comp_result