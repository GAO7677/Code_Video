rm -rf /data/gaoya/AAA_test_video/Dataset_physV/liquid_cache
rm -rf /data/gaoya/AAA_test_video/Dataset_physV/liquid_dataset

/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/try2_luquid_genesis0412.py \
    --num-scenes 1 \
    --solver sph \
    --container-mode procedural \
    --container-counts 1,2,3 \
    --min-containers 1 \
    --max-containers 3 \
    --particle-size 0.008 \
    --tracked-particles 1024 \
    --max-particles 65536 \
    --liquid-preset triple_arc \
    --liquid-volume-scale 3.0 \
    --liquid-volume-range 1.4,6.0 \
    --camera-distance-scale 1.35 \
    --camera-fov 58 \
    --output-dir /data/gaoya/AAA_test_video/Dataset_physV/liquid_dataset \
    --cache-dir /data/gaoya/AAA_test_video/Dataset_physV/liquid_cache \
    --render-platform auto --force-cpu \
    --fluid-vis-mode particle \
    --render-rgb-modes particle \
    --container-double-sided false \
    --render-scene-bounds false \
    --source-mode spout \
    --source-delay-frames 6 \
    --overwrite


# sh /home/gaoya/Code_Video/Code_data/try2_luquid.sh
