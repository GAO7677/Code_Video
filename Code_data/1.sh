PYTHONNOUSERSITE=1
python /home/gaoya/Code_Video/Code_data/try1_physxnet_articulation_mpm.py \
    --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/ \
    --object_id 19925 \
    --output_root /data/gaoya/AAA_test_video/Dataset_physV/physxnet_genesis_mpm_case_0613 \
    --run_genesis \
    --num_random_cases 4 \
    --prefer_existing_runtime_meshes \
    --dt 0.003 \
    --substeps 40 \
    --ball_posx 0.03 \
    --disable_rigid_visual_double_sided_shell
