# 代码 Bug 修复清单

本清单采用追加方式维护。以后本项目每次修复代码 bug，必须追加日期、原因、修改文件、验证结果和限制，不覆盖历史记录。项目规则见 `AGENTS.md`。

## 2026-09-22 — 缺失支撑准入检查与平台射线投影实验

### 已确认的问题

- `pybullet_initial_contact.py` 在穿插未超过 1 mm 时直接放行，未检查调用者声明的支撑接触是否实际存在。旧 support_edge_g03_v0620 的 C 因估计平台横向偏移没有初始支撑，却被当作有效轨迹推进。
- 平台拟合用球体估计高度修正顶面 z 后，继续沿用原深度点的 x/y 范围，导致顶面几何与观测像素射线不一致。
- 剩余尺度偏差未解决；不能把初始化拒绝当作成功恢复几何。

### 修改清单

| 文件 | 修改 | 启用状态 |
|---|---|---|
| `code/pybullet_initial_contact.py` | 无明显穿插时仍校验已声明支撑、朝上法向及球心是否位于有限 AABB 内；缺失时拒绝且不推进时间 | 默认启用；支撑必须由调用者声明 |
| `code/evaluate_context_rgb_pybullet.py` | 固定 support_edge pilot 自动声明 RGB0–7 连续左平台支撑前提，防止遗漏传参 | 默认启用，仅该 pilot 协议 |
| `code/fit_context_collision_primitives.py` | 增加 `support_ray_projection=True` 实验：将已选顶面像素射线与接触平面求交，再拟合有限边界；记录原边界及位移 | 显式实验，默认关闭，无自动 fallback |
| `tests/test_initial_contact_gate.py` | 增加缺失支撑拒绝测试；保留穿插、对齐、速度、多 client 测试 | 7/7 通过 |
| `code/validate_support_projection.py` | 对 12 个平台 case 冻结估计后再读 GT，评测严格 C、接触对齐 C、严格 D；保存事件、几何误差与 overlay | EXECUTED |
| `AGENTS.md` | 后续 bug 修复必须追加本清单 | 项目规则 |

### 验证命令

在项目根目录执行（版本输出目录必须不存在）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B tests/test_initial_contact_gate.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/validate_support_projection.py
```

输出：`/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1/support_projection_fix_20260922_v1`。

### 实际结果与限制

- 初始化回归测试 7/7 通过；12 个平台 case × 3 模式全部完成检查。CPU、两线程，无新 GPU 推理、无训练、无未来 RGB 输入。
- 新几何下严格 C 12/12 拒绝；显式对齐 C 6/12 可推进、6/12 拒绝；D 12/12 初始支撑成立并可推进。BLOCKED 无预测轨迹，误差不填零。
- 目标 case 平台 y 上边界从 -0.00830 m 改为 +0.18544 m；GT 为 +0.520 m。GT 球心 y=0.21783 m 仍越界，接触为斜向边缘穿插，因此严格 C 与接触对齐 C 均拒绝。没有横向移动 GT 球或扩大平台。
- 目标缺口宽度由 0.525 m 改为 0.540 m，GT 为 0.620 m；仍不准确。
- 目标 D ADE 0.340197 → 0.339761 m，变化很小；12-case D 平均 ADE 0.296555 → 0.296594 m，未改善，因此新几何策略不默认上线。
- 目标 D 在 0.92083 s 后离开左平台（GT 0.875 s），说明未通过无限扩大平台阻止下落。但后续接触事件仍不匹配（预测接触右平台，GT 落地），不能宣称物理预测已修好。
- 本轮完成准入 bug 修复及可复现几何实验；完整米制状态/几何联合估计仍为 PARTIAL。后续需诊断球轮廓/透视模型与尺度来源，再验证平台区域和多帧支撑约束，不得用 GT 输入强行对齐。

## 2026-09-22 — 全 36 case / 144 显示分支回归复核

- 新增 `code/recheck_all_viewer_cases.py`。使用修复后准入代码重新执行页面中全部 A/B/C/D、GT omega 分支；冻结 VGGT 输入和 primitives_v3，实验射线几何关闭。C 保留 v2 原来的显式接触对齐策略，非自动 fallback。
- 命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/recheck_all_viewer_cases.py`。
- 预先固定轨迹一致性容差 1e-6 m。114 组轨迹一致，最大差异 1.07624e-7 m；9 组仍被拦截；21 组新增缺失支撑拒绝。A 36/36 不变，B 24 不变/12 新拒绝，C 18 不变/9 旧拒绝/9 新拒绝，D 36/36 不变。
- 所有拒绝分支实测 stepSimulation 调用为 0；可推进分支每组 328 次。额外验证了 v2 接触对齐 C 的严格模式先拒绝。
- 以前标 PASS 的 21 组实际缺少初始左平台接触；本轮改变的是准入判定，不能当作模型性能改善。未发现被保留轨迹数值退化；未宣称所有旧 PASS 预测正确。
- 原 support_edge_g03_v0620 C 现在缺失支撑即拒绝，错误自由落体不再作为新预测展示；估计几何本身仍未修复。18 组旧 C 接触对齐结果未变；其余 9 组旧拒绝仍未解决。
- 新结果：`all36_recheck_20260922_v1`；新页面：`overlay_viewer_v3`。旧 v2 数据及页面保留用于对照。所有结果位于原输出根目录下。

## 2026-09-22 — 显式自动提示与球体状态恢复独立分支（精度 FAIL，未替换旧实现）

- 用户确认沿用RGB自动目标选择，无需人工提示。代码审计确认RGB7 Hough框+SAM2双向跟踪，不是GT框；未调用另一路proxy-box fallback。
- 旧球心默认水平约束、平台默认厚度/接触高度为已确认隐式先验；本次新P1完全不调用这些步骤，不把它们直接定性为PyBullet bug。
- 新增 `code/sphere_state_recovery.py`：完整mask射线球拟合、三种多帧p/v；`code/run_sphere_state_gate.py`：估计/冻结/独立评测、逐例flags、family/group门限；`code/diagnose_sphere_state_gate.py`：评测专用GT投影、旧结果对照、36例RGB0–7轮廓overlay。
- 新增 `tests/test_sphere_state_recovery.py`，精确切线射线（含离轴）和终点速度测试2/2通过。真实轮廓精度没有通过，不能以单测通过代替实验合格。
- 实测36例/12组，CPU两线程。主方法预先固定robust_linear，p7 median 0.4234 m/p90 0.7892 m，v7 vector median 0.2897 m/s，direction median 7.119°；三family FAIL。GT仅在估计hash冻结后读取。
- 失败定位：aperture_g01_v0460 SAM2下沿超出球投影，RGB7切线残差median +13.41 mm；p7旧0.02245→新0.43166 m。support_edge_g03_v0620旧0.26156→新0.30286 m。精确投影模型不足以消除轮廓偏差，未宣称深度误差已修复。
- 输出 `sphere_state_gate_20260922_v1`（原大输出根目录内）；报告含命令。执行入口：`run_sphere_state_gate.py estimate/evaluate --output <新目录>`，evaluate另传phase1_v5的`--gt-root`；统一使用physrvg-full-sa Python、CUDA_VISIBLE_DEVICES为空及OMP/MKL/OPENBLAS=2。
- 新页面由原8899服务提供，浏览器36例加载/零异常通过。旧结果、权重和求解器未变。P0全面旧几何逐例审计仍PARTIAL；P2–P6按用户P1失败停止条件NOT_RUN。新分支仅实验诊断，非默认精度修复上线。

## 2026-09-22 — 六例匿名 RGB-only 通用输入实验（PARTIAL，未替换 v3）

- 目标：分离旧pilot的已知半径、相机标定、family几何、默认支撑与GT状态依赖。新增独立分支，不改旧结果或动力学求解器。
- 文件：`prepare_generic_six.py` 匿名context筛选/导出；`run_generic_six.py` 预声明六例、VGGT/SAM2推理和冻结；`generic_context_geometry.py` 运动候选、未知半径球面拟合、有限可见mesh和重力UNKNOWN；`generic_bullet_input.py` 无family/blueprint的输入适配；`evaluate_generic_six.py` 冻结后GT评测；`tests/test_generic_context_geometry.py` 合成恢复、运动歧义拒绝和Bullet时序测试。尺度配置已实际经协议快照读取，值5.819486884015457。
- 六例固定ordinal 000/015/030/040/045/050，先从匿名RGB0筛单球/非关节布局，每布局取首例，推理前冻结，不按结果换样本。GPU6串行视觉推理，CPU两线程。
- 实际结果：6/6视觉推理/可见mesh；1/6球状态内部拟合通过但p7误差0.2346 m、v7误差0.3337 m/s、半径误差0.02797 m；1个拟合候选FAIL，其余4个mask/球形支持检查失败；完整physics输入0/6，gravity UNKNOWN 6/6，rollout NOT_RUN。不能把文件生成当作精度或端到端成功。
- 重要限制：重力仅提取unsigned法向轴，未实现可靠的axis/sign判别，不声称已完成通用重力恢复。mesh未知厚度不补齐，遮挡孔洞不填；静态几何碰撞等价性未验证。
- fixed-scale跨case尚未成立：两个case的局部GT诊断尺度约6.50/6.35，固定5.819低估10.47%/8.34%；没有GT对齐补偿。5/6缺对应Cycles静态深度，明确BLOCKED而非使用另一相机depth。
- 评测数据bug确认：v1 Cycles GT masks纵轴相反，六例原centroid投影差12–311 px，显式vertical flip后0.50–0.97 px。只转换新评测副本，未改原始文件；v2桌面mask已正确。修复空GT锚点导致NaN序列化失败，空集合记BLOCKED，不调整估计。
- 初始碰撞只测获准状态1例：0/1重叠、contact=0、step=0、未移动位置；其余5例未检查，不能报0/6或支撑正确。
- 验证：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B tests/test_generic_context_geometry.py`，3/3通过。时序测试328调用、首1/30秒、末41/30秒。完整执行命令与失败阶段见产物report.md。
- 产物：`/data/gaoya/agent-data/outputs/context_generic_six_20260922_v1`；最终评测`evaluation_v3/report.json`，前两评测目录为失败尝试，不混报。估计freeze SHA256 `cd820a41a6353f2a046720f93c9832b2f38ca74ef5b72b30d3dc064ef07565f7`。
- 启用状态：独立实验入口，非默认v3替换。本轮停止六例机制验证，不增加模型/场景/物体类别、不使用GT omega救活主结果。

## 2026-09-22 — 重力方向prior启用前验证（FAIL，未启用）

- 用户授权先验证水平面法向，可靠才加入prior。新增`code/check_generic_gravity.py`，六例RGB0/7静态点RANSAC候选，冻结后独立GT重力评测；没有模型重跑或动力学改动。
- 最大共识平面方案仅3/6在两帧都接近重力轴：clip_01/04/05约0.6–1.2°；clip_02/03稳定误选墙面约90°；clip_00主平面切换，首尾差89.8°。正确平面存在于候选不等于可以不用GT可靠选中。
- 预设角度门限5°、首尾漂移3°。稳定性无法排除稳定墙面；向下符号没有独立观测证据。结论为验证FAIL，不启用prior，保持gravity UNKNOWN和原结果。
- 产物`/data/gaoya/agent-data/outputs/context_generic_six_20260922_v1/gravity_check_v1/`；report.md含实际estimate/evaluate命令、逐例角度和限制。CPU两线程，GT仅冻结后评测，无GT择优、默认Z-up或位置对齐。

## 2026-09-22 — 用户授权的下方平面重力prior（六例验证PASS，独立入口启用）

- 新增`code/lower_plane_gravity.py`：图像底部35%静态点平面候选，显式正立相机/相机位于平面上方prior；最低35%共识、与图像下方向<=60°、首尾漂移<=3°。不增加GT/family/支撑对齐，solver不变。
- 候选hash冻结后独立GT验证，有符号方向误差六例0.224–1.267°，均通过预设5°；首尾漂移0.007–0.126°。旧全图最大面导致墙面误选，此显式prior在当前六例消除了该问题，但不证明任意视频泛化。
- 新独立实验分支实际应用：clip_00零omega续推41帧/328steps，ADE0.515493 m、FDE1.020542 m；其余五例原状态FAIL维持，0steps。不以重力通过冒称轨迹正确。
- 产物`context_generic_six_20260922_v1/lower_plane_gravity_v1/`，report.md含命令、逐例误差、局限。旧v3和源冻结文件未改。CPU两线程，无GPU或模型重跑。

## 2026-09-22 — 旧36例完整新链路重跑（全体状态准入FAIL，旧v3保留）

- 新增`prepare_generic_pilot36.py`匿名导出36例RGB0–7/时间戳、核验原始VGGT缓存；`run_generic_six.py`增加显式匿名原始缓存复用；`lower_plane_gravity.py`缺跟踪时明确UNKNOWN；`finish_generic_pilot36.py`观测侧rollout冻结后GT评测与36例overlay。
- GPU6重跑运动候选+SAM2；不读取旧状态、已知半径/标定、family几何或GT omega。固定scale、未知半径拟合、观测mesh、下方平面prior和zero omega均进入新分支。
- 结果36/36状态FAIL：14 mask碎裂、17球形检查失败、2半径CV失败、3表面残差及半径CV失败；全部0 step，新轨迹/ADE/FDE/contact NOT_RUN。保持失败，不使用旧逻辑fallback。
- 重力36/36观测侧接受，GT误差median0.473°/max5.403°，四个deflector v28000超过5°验证门限；未用GT择优或修正。
- 可视化新mask与旧D/GT轨迹，旧D明确标GT omega；无新轨迹时不伪造。浏览器36例加载与详情检查PASS，完整命令/失败表见产物report.md。
- 产物`/data/gaoya/agent-data/outputs/context_generic_pilot36_20260922_v1`；8899路径`generic_pilot36_v1/`。独立实验，不替换旧v3；模型权重/solver未修改。

## 2026-09-23 — test70无形状先验mask诊断与评测覆盖修正

- 新增`code/test70_mask_diagnostic.py`、`code/evaluate_test70_masks.py`、`web/test70_mask_diagnostic.html`。完整70例匿名RGB0–7，沿用运动候选+SAM2，移除本诊断入口的球形/碎片准入；不修改原pipeline、不修整mask、不运行3D/物理。
- 实测GPU6/CPU两线程：59例SAM2执行、10例多候选UNKNOWN、1例无候选FAIL。循环115.396秒，不含模型构建/评测；EXECUTED不等于正确。
- 确认评测问题：原Cycles GT mask纵轴反转，评测副本按GT投影中心审计后翻转；GT漏标domino触发球和seesaw板，不能将这些目标匹配到无关actor后按IoU=0汇总。补充动态actor覆盖审计，缺标且身份不确定的6例BLOCKED_TARGET_COVERAGE；原重叠保留为诊断字段。第一版evaluation保留，最终采用evaluation_v2。
- 确认展示问题：仅填外轮廓会隐藏原mask的孔洞。纯mask面板改为直接读取原始二值PNG；未改变模型输出。
- 最终53例可评分mask：mean IoU0.515805，median0.574810；>=0.8共21例，[0.5,0.8)共9例，<0.5共23例。GT仅冻结后读取，未回流提示选择。不能将全部失败归因于画质；没有画质/提示配对消融。
- 产物`/data/gaoya/agent-data/outputs/test70_motion_sam2_20260923_v1`，完整命令、GT覆盖限制和旧tracker的JPEG暂存路径见report.md。冻结hash6188c6edd0021267ca1b73373cdfd34f567b6a7e3c9ce621f1b73597163b545b。默认pipeline未替换；诊断页面复用8899/test70_masks_v1/。
- 验证：原六例运动检测决策/框一致性检查PASS；70例推理和第二版评测完成；`git diff --check`通过。浏览器逐例结果另见产物browser_check.json。

## 2026-09-23 — test70单短语Grounding DINO定位诊断（实验入口）

- 新增`code/diagnose_grounding_text.py`、`code/assemble_grounding_diagnostic.py`、`web/test70_grounding_text.html`和最终配置`configs/test70_grounding_vocab_single_final_20260923.json`。每组只使用一个共享目标短语；不使用多类别prompt、top1截断、NMS、GT框或future。
- 通过r1–r7记录所有调词尝试。最终组合在70例×8帧共560帧均返回恰好一个原始框；r6全量560帧实测59.911s，斜坡长度组单独r7复跑40帧解决clip_012两帧多框。全部候选叠框和原始JSON保留。
- 目标身份为context视觉人工检查PASS；不是独立人工标注或GT IoU验证。SAM2未重跑，不能称为mask改善。跷跷板短语指定载荷木块，多米诺短语指定触发球，均不声称场景只有一个运动刚体。
- `code/diagnose_grounding_text.py`修正输出目录按group/clip分层，避免不同组相同clip名覆盖；模型权重missing_keys为空，unused keys记录在load_audit。GPU6/CPU两线程，GPU4未使用。
- 产物`/data/gaoya/agent-data/outputs/test70_grounding_text_20260923`，页面8899路径`test70_grounding_text/`。实验性诊断入口，不替换默认运动候选或v3 pipeline；词表是在本70例上调优，不能当作盲测泛化。
- 验证：最终页面逐70例、逐8帧加载、Grounding框图和旧SAM2对照图PASS；共560帧，未进行GT指标或新SAM2推理。

## 2026-09-23 — pilot36/test70文本唯一框＋SAM2重跑（独立诊断）

- 新增`code/run_grounded_mask_diagnostic.py`、`code/compare_grounded_masks.py`、`web/grounded_mask_comparison.html`。以Grounding DINO RGB7唯一原始框替换运动差分框，SAM2权重/预处理/传播保持一致，无mask修整、GT或旧框fallback。
- pilot36先统一ball时4例挡板产生额外框；保留失败输出，整个deflector组统一brown ball，最终288/288帧唯一；aperture/support_edge使用ball。36例实际SAM完成，旧二维检查14碎裂/17圆形度失败/5通过→新36/36通过；GT mask IoU不可得，3D未跑，不能称全链路PASS。
- test70最终词表560/560唯一，70/70 SAM执行。共同可评分53例IoU均值0.515805→0.953046；49例提升>0.01、4例变化≤0.01，无下降；旧21例IoU≥0.8仍全部≥0.8。新65例可评分均值0.955307，全部≥0.8；5多米诺触发球缺标不计分。
- test70首轮SAM被用户中断，旧v1不完整结果保留；最终在v2新目录全部重跑，不将部分输出标完成。pilot36完整SAM循环83.164s，test70为133.088s，GPU6/CPU两线程，无VGGT、3D、Bullet或训练。
- 产物`pilot36_grounding_sam2_20260923_v1`、`test70_grounding_sam2_20260923_v2`（均在/data/gaoya/agent-data/outputs）；报告包含输入边界、执行命令、条件均值分母、GT覆盖与调词泛化限制。新页面8899/pilot36_grounded_masks/、8899/test70_grounded_masks/，原页面和原结果保留。
- 验证：浏览器106例×8帧新旧overlay/二值mask全部加载PASS，prediction_freeze验证通过，`git diff --check`通过。当前仍为实验诊断入口，不替换默认pipeline。

## 2026-09-23 — Grounding DINO＋SAM2接入通用36例完整链路（实验集成，准确性FAIL）

- 新增`code/run_grounded_generic_pilot36.py`、`code/publish_grounded_generic.py`、`web/grounded_generic_pipeline.html`。入口验证既有VGGT input freeze、SAM2 prediction freeze及288帧RGB身份；复用冻结模型输出，CPU两线程重跑未知半径球拟合、有限mesh、平面重力prior、zero-omega Bullet。未占GPU、未改solver/阈值、未启用旧family几何/支撑对齐。
- 这次解决的是新mask尚未接入3D/物理的实验链路缺口，不宣称修复了深度或平台。二维36/36通过；状态20/36通过数值准入（aperture12、deflector8），16例残差/半径稳定性失败（deflector4、support_edge12）。生成36份mesh与重力结果；20例实际328step调用，16例0step。
- 冻结后评测：全部36例p7误差中位数0.5011m、v7向量误差0.4099m/s；实际执行20例ADE3.3631m、FDE9.2462m，同输入CV ADE0.7514m。初始穿插0/20但初始接触0/20、30Hz未来接触帧总数0；不能以无穿插宣称支撑正确。mask并集排除会留下mesh未知区，尚未独立分离该因素与状态/深度偏差的因果贡献。
- 原`overlay_viewer_v3/index.html`加入新旧版本切换，新版默认展示；旧HTML完整备份在新产物`legacy_v3_index.html`，旧app/轨迹/模型文件未修改。新增每帧mask/深度、RGB0实际mesh覆盖、RGB7上的GT/旧D/新D/CV轨迹；失败分支不画新轨迹。所有GT度量发生在rollout freeze后。
- 产物：`/data/gaoya/agent-data/outputs/context_grounded_generic_pilot36_20260923_v1`。完整命令见report.md；模型前段为CACHED，本次后段实测39.977s（含评测，不是完整视觉耗时）。固定scale与调优短语仍是显式prior，不能宣称arbitrary-video泛化。
- 验证：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B tests/test_generic_context_geometry.py`，3/3通过（该环境未安装pytest，使用测试自带入口）。`node /tmp/check_grounded_generic_v3.cjs`验证36例×8帧四面板、旧版36例及版本切换PASS。页面展示PASS不等于预测正确；contact precision/recall与GT primitive边界误差仍NOT_EVALUATED。

## 2026-09-23 — 新结果恢复原v3播放布局（展示修正）

- 用户要求沿用原v3而非默认四面板诊断页。新增`web/grounded_v3_layout.js`和`code/publish_grounded_v3_layout.py`，复用旧v3 HTML/CSS布局：case列表、overlay播放、6步骤、右侧指标、常驻图层解释；原URL默认展示该布局，四面板仍在报告链接中。
- 新结果图层准确标为GT评测/CV/旧D/new zero-omega D，不把未运行的A/B/C消融伪装成已执行。16例FAIL不借用旧预测。原v3可切回，物理输出与模型未修改。
- 重新核验36条rollout：20条均有41 positions/velocities/contact frames、328step calls，16条失败均0step；能运行的案例已全部完成，不重复启动同样仿真。
- `node /tmp/check_grounded_v3_layout.cjs`验证36例×6阶段及新旧切换PASS；截图与检查结果保存于`context_grounded_generic_pilot36_20260923_v1/v3_layout_*`。仅展示修正，未改善轨迹准确性。

## 2026-09-23 — 按用户要求回退默认展示

- 将`overlay_viewer_v3/index.html`的新版iframe入口从`v3_layout.html`恢复到先前四面板诊断首页。HTTP页面读取成功；仅恢复展示入口，原v3切换和全部计算结果保留，未重跑或回退物理代码。

## 2026-09-23 — 按截图模板独立重跑36例并发布v2

- 用户以截图明确原v3模板。默认入口改为`grounded_generic_pilot36_20260923_v2/v3_layout.html`，保留列表、单overlay播放器、六步骤、右侧指标、常驻图层表及原v3切换。`publish_grounded_v3_layout.py`支持显式输出root；JS改为读取当前目录，防止新版本页面意外展示v1数据。
- `run_grounded_generic_pilot36.py`在新v2目录重新执行全部36例，模型前段复用核验缓存、后段CPU两线程独立计算，39.362s；20例完整41帧/328step、16例状态FAIL。36份rollout内容与v1逐文件一致，estimate/rollout freeze通过。未改变阈值、solver或输入恢复机制，结果准确性仍FAIL。
- `publish_grounded_generic.py`新增实际mesh边界的RGB7投影，用紫色overlay展示；从真实三角形边计数得到边界，不复用旧family门框盒。图层为GT/CV/旧D/新zero-omega D，未伪造A/B/C实验。
- 验证：`node /tmp/check_grounded_v3_layout.cjs /data/gaoya/agent-data/outputs/context_grounded_generic_pilot36_20260923_v2`，36例×6阶段/切换PASS；`git diff --check`通过。新报告、截图、浏览器检查及冻结文件均位于上述v2目录，旧产物保留。

## 2026-09-23 — 通用静态表面按帧可见性融合，修复跨帧mask误删（部分修复）

- CONFIRMED_BUG：RGB0深度建mesh却删除RGB0–7全部动态mask并集，导致其他帧可见静态区域被错误删除。新增`generic_context_geometry.finite_mesh_observed`：逐帧独立mask及同样9像素扩张、相机重投影/z-buffer、至少两视图3%深度一致性中位数、原有限三角形规则。不读取case/family/GT，不假定平台、不补所有帧均未观察的表面。
- `run_grounded_generic_pilot36.py`默认使用observed_multiframe；显式`--geometry-mode legacy_union`保留旧协议。状态/尺度/omega/solver全部保持不变并逐例比对。此处修复为通用观测逻辑，未保证多数case轨迹正确。
- `tests/test_observed_surface_fusion.py`：旧代码在“后续mask错误删除原可见平面”断言FAIL；新实现2个测试PASS，覆盖移动遮挡恢复、真实缺口保留、动态表面排除、相机平移及scale。原`tests/test_generic_context_geometry.py`3项PASS。命令均CPU两线程、CUDA_VISIBLE_DEVICES空，未使用GPU。
- 独立36例输出`/data/gaoya/agent-data/outputs/context_grounded_geometry_fix_20260923_v1`；`audit_geometry_fix.py`检查冻结输出并记录逐例配对。20例执行/16例状态失败不变；20例平均ADE3.3631→3.1717m，6例改善>1cm、3例退化>1cm、11例变化≤1cm；记录到接触的案例0→10。不能将改善平均值称全链路PASS。
- aperture_g00_v0280未解决：下方候选平面投影点8帧均在目标mask内；按当前“不补不可见区域”边界无法由真实观测融合恢复。局部共面补全需要用户确认，当前未实施；不能使用GT/默认平台规避。球p7/radius偏差未修复。
- 可视化继续原v3模板，新增修复前对照及逐例回归报告链接。发布脚本修正初始重叠检查分母为实际检查数，并按协议生成报告，防止新版本显示旧常量结论。页面/物理正确性分开验收，浏览器结果另存产物。

## 2026-09-23 — 用户授权的独立局部平面补全模式（实验，非默认fallback）

- 新增`code/local_plane_completion.py`及`tests/test_local_plane_completion.py`，接入`finite_mesh_observed(..., complete_local_planes=True)`和显式CLI`--geometry-mode local_plane_completion`；默认仍observed_multiframe。只处理参考帧目标遮挡范围内未知深度，检查周围环带90%有效观测、至少40点和8方向支持、平面相对RMS≤0.005/P95≤0.01、有限深度外推范围。拒绝证据不足、台阶/非平面、图像边界，不按family或case提供几何值。
- 新增面保存`face_inferred`、逐连通域`completion_audit`（法向、offset、残差、源点数、拒绝原因）。不改球p/v/radius、已观测mesh顶点、重力/solver，不补厚度。真正完全遮挡的缺口不可辨识，局部共面先验不能保证遮挡区实际无缺口；报告明确此限制。
- 全36例输出`/data/gaoya/agent-data/outputs/context_local_plane_completion_20260923_v1`。13例有推断像素，但只有3例产生有效碰撞三角形；其余细小推断像素不虚报为有效补全。20例实际rollout/16例原状态失败保持；相对纯观测多帧mesh，3例改善>1cm、0例退化>1cm、17例变化≤1cm；平均ADE3.1717→2.8034m。
- aperture_g00_v0280：ADE3.1004→0.7165m，接触帧0→38/41；同组v0460/v0720也改善。不宣称状态/尺度误差已修复或多数新场景已验证可用。`report_local_plane_completion.py`在冻结后逐例验证状态和原观测顶点完全未改，输出completion_comparison.json及completion_report.md。
- 验证：CPU两线程运行`tests/test_local_plane_completion.py`（2项，覆盖平面、两斜面、真实可见缺口、台阶、开放边缘、已观测像素不变）、`tests/test_observed_surface_fusion.py`（2项）、`tests/test_generic_context_geometry.py`（3项），全部PASS；无GPU占用或训练。完整运行命令见产物report.md。
- 原v3模板显示黄色推断面与紫色观测mesh边界，保留纯观测mesh对照入口，报告不将推断面标为真值。浏览器验收结果见产物v3_layout_browser_check.json。

## 2026-09-23 — 最新补全输入的A/B/C/D zero-omega六层对照

- 按用户确认，B包含估计radius+p7/v7；A/C使用GT radius+p7/v7；所有组zero omega、共享冻结估计重力与固定物性。GT只在验证视觉/几何freeze后读入独立评测分支。A非旧GT omega完整oracle重放，报告与UI明确标注。D继续使用估计半径，不为对照改成0.11m。
- 新增`run_grounded_abcd.py`、`mesh_contact_alignment.py`、`publish_grounded_abcd.py`、`web/grounded_abcd_viewer.js`。扩展`generic_bullet_input.run`可选GT世界构造、C诊断对齐及API步接触日志；原默认动力学路径不变。GT几何由原builder加载后做刚体坐标注册，不拟合尺度，不按mesh AABB上表面伪造支撑。GT半径字段必需，无默认fallback。
- 严格C和C_aligned独立：后者用A初始接触确认支撑，只允许近水平支撑穿插向上≤55mm，不吸附悬空球、不修改速度/几何。新增`test_mesh_contact_alignment.py`验证严格拒绝、小穿插恢复、超限拒绝、无支撑拒绝和无吸附；原3项几何/动力学测试PASS。
- `/data/gaoya/agent-data/outputs/context_grounded_abcd_zero_20260923_v1`完成36×5=180模式尝试。A36执行、B12执行（16状态FAIL+8穿插拒绝）、C_strict32执行（4穿插拒绝）、C_aligned36拒绝（32缺初始支撑+4侧向接触）、D20执行/16状态FAIL。各自有效均值ADE：A0.1281m、B0.3026m、C_strict3.0909m、D2.8034m；分母不同，不当独立因果效应。共同有效ABCD严格队列8例，另单列。
- 180份结果已审计：执行项41帧/328step/328条API调用接触快照；拒绝项0step；20条D轨迹与冻结部署D最大差<1e-8m。接触按法向划分support/obstacle角色，非family GT语义事件precision/recall。GT所有36个半径值核对来自blueprint显式字段。
- 原v3页面新增实际六层：A/B/C/D、Estimated geometry、GT geometry；C模式可切换严格/aligned，拒绝不绘轨迹。估计mesh紫色、推断面黄色、真实GT盒边绿色；未将mesh伪装为旧family盒。浏览器验证36例×6步骤、六层开关、C切换及新旧页入口，截图与check JSON在产物目录。
