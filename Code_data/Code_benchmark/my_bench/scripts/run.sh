# 停掉 8766
lsof -ti tcp:8766 | xargs -r kill

# 在 /data/gaoya/AAA_test_video 下启动 8766
cd /data/gaoya/AAA_test_video
python -m http.server 8766

# http://127.0.0.1:8766/Benchmark/VBench/dashboard/