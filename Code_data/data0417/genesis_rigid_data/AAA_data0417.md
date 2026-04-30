# data0417 可视化运行记录

## by_raw_window 分类可视化

当前用于可视化 `/home/gaoya/Code_Video/Code_data/data0417/data_summary/by_raw_window` 下所有分类样本的脚本是：

`/home/gaoya/Code_Video/Code_data/data0417/data_check/build_by_raw_window_portal.py`

这个脚本会：

- 读取 `by_raw_window` 下面各个分类 `.json` 中记录的样本路径
- 为每个分类默认抽取前 `10` 个样本
- 生成左侧目录树导航页面，按照 `raw/window -> train/test/benchmark -> category` 浏览
- 对可视化媒体统一转成浏览器更稳定的 `mp4`
- `window` 样本展示 `Context / Future / Full`
- `raw` 样本展示 `Raw Video`

## 运行命令

建议使用 `wan` 环境运行：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/build_by_raw_window_portal.py
```

如果想先清空旧的门户资产再重建，可运行：

```bash
rm -rf /home/gaoya/portal_hub_sim/by_raw_window_portal/assets
python /home/gaoya/Code_Video/Code_data/data0417/data_check/build_by_raw_window_portal.py
```

语法检查命令：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python -m py_compile \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/build_by_raw_window_portal.py
```



## 本地访问地址

当前本地静态服务根目录是 `/`，端口是 `8048`，因此浏览器访问：

`http://127.0.0.1:8048/home/gaoya/portal_hub_sim/by_raw_window_portal/index.html`

## 页面使用说明

- 左侧目录树用于切换分类
- 顶部搜索框可按类别名筛选
- 右侧显示当前分类对应的样本卡片
- 每个样本卡片包含：
  - `Caption`
  - `Detail Caption`
  - `Context Video / Future Video / Full Video`，或者 `Raw Video`

如果浏览器仍显示旧版本页面，直接强制刷新即可。
