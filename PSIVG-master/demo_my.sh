# caa
# '''
# 提取的帧：/data/gaoya/AAA_test_video/arxiv_Code/PSIVG/INPUT_DATA/Frames/0000/00000.jpg
# 分割掩码：/data/gaoya/AAA_test_video/arxiv_Code/PSIVG/OUT_Perception/0000/00000/mask/mask.jpg
# 3D 网格：/data/gaoya/AAA_test_video/arxiv_Code/PSIVG/OUT_Perception/0000/00000/meshes/ball.obj
# 背景修复结果：/data/gaoya/AAA_test_video/arxiv_Code/PSIVG/OUT_Perception/0000/00000/inpaint/inpainted_all.jpg
# '''
CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/PSIVG-master/main_part1.py --video /home/gaoya/Code_Video/PSIVG-master/assets/0000.mp4

# cas
# {DATA_ROOT}/OUT_ViPE_Export/{sample_id}/success.txt（成功标记文件）
# {DATA_ROOT}/OUT_ViPE_Export/{sample_id}/video_info/video_info.json（视频全局信息，如相机参数、平面模型、重力方向等）
# {DATA_ROOT}/OUT_ViPE_Export/{sample_id}/frames_info/{frame_idx:05d}/{frame_idx:05d}.json（每帧的物体姿态、相机位姿等信息）
# {DATA_ROOT}/OUT_ViPE_Export/{sample_id}/video_info/size_estimate_{idx}/{object_name}.obj（优化尺度后的物体网格，如 ball.obj）
# {DATA_ROOT}/OUT_ViPE_Export/{sample_id}/video_info/size_estimate_{idx}/box_ious.png（尺度优化过程的 IOU 曲线可视化）
# {DATA_ROOT}/OUT_ViPE_Export/{sample_id}/video_info/size_estimate_{idx}/RGB_side_by_side.png（原始视频帧与优化后网格渲染的对比图）
# {DATA_ROOT}/OUT_ViPE_Export/{sample_id}/estimate_success.txt（尺度优化成功标记文件）
# {DATA_ROOT}/OUT_Simulation/{sample_id}/{simulation_id}/point_cloud/{frame:05d}.ply（每帧的粒子点云，用于可视化模拟过程）
# {DATA_ROOT}/OUT_Simulation/{sample_id}/{simulation_id}/particles/{frame:05d}.npz（每帧的粒子数据，包含位置、速度等物理量）
# {DATA_ROOT}/OUT_Simulation/{sample_id}/{simulation_id}/metadata.json（模拟元数据，如粒子数量、物理参数等）
# {DATA_ROOT}/OUT_Simulation/{sample_id}/{simulation_id}/success.txt（模拟成功标记文件）
# 渲染后的 RGB 视频文件（具体路径需参考 render_RGB_video 函数实现，通常位于 OUT_RENDERING_DIR 或 OUT_Simulation 下）
# 光流文件和点对应关系文件（具体路径需参考 calculate_flow 函数实现，通常位于 OUT_Simulation 下）
CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/PSIVG-master/main_part2.py --video "0000"