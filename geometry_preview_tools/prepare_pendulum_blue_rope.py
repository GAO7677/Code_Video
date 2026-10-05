from pathlib import Path
import json,shutil
D=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2/test70_v2');R=D.parent.parent/'20261005_test70_v2_pendulum_blue_rope';R.mkdir(exist_ok=True)
cfg=json.loads((D/'config.json').read_text());cfg['scene_by_family']['V2V_PENDULUM']['actor_material_overrides']={'pendulum_rope':dict(kind='rough woven rope',color_srgb='#123B78',roughness=.72,bump_strength=.18,bump_distance=.0006)}
(R/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
for n in ['fit_room.py','egl_guard.py']:shutil.copy2(D/n,R/n)
s=(D/'render.py').read_text();marker='actors={};dynamic=m'
helper='''def rope_material(name,settings):
 mat=bpy.data.materials.new('Task_'+name+'_deep_blue');mat.use_nodes=True;nodes=mat.node_tree.nodes;links=mat.node_tree.links;bsdf=nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(*srgb(settings['color_srgb']),1);bsdf.inputs['Roughness'].default_value=settings['roughness'];bsdf.inputs['Specular'].default_value=.25
 coords=nodes.new('ShaderNodeTexCoord');noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=150;noise.inputs['Detail'].default_value=2;links.new(coords.outputs['Object'],noise.inputs['Vector']);bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=settings['bump_strength'];bump.inputs['Distance'].default_value=settings['bump_distance'];links.new(noise.outputs['Fac'],bump.inputs['Height']);links.new(bump.outputs['Normal'],bsdf.inputs['Normal']);material_audit[name]=dict(settings);return mat
'''
assert marker in s;s=s.replace(marker,helper+marker)
needle=" obj=orig.add_actor('Task_'+name,a,mat,edge_clarity=False);"
assert needle in s;s=s.replace(needle," if name in placement.get('actor_material_overrides',{}):mat=rope_material(name,placement['actor_material_overrides'][name])\n"+needle)
compile(s,'render.py','exec');(R/'render.py').write_text(s)
c='v2v_pendulum_l110';shutil.copytree(D/'source_cases'/c,R/'source_cases'/c,dirs_exist_ok=True);(R/'samples'/c).mkdir(parents=True,exist_ok=True);(R/'logs').mkdir(exist_ok=True)
T=D.parent.parent/'20261003_test70_v2_domino_closeup';s=(T/'render_preview.py').read_text().replace('v2v_domino_g090',c).replace('domino_preview.log','rope_preview.log');(R/'render_preview.py').write_text(s)
s=(T/'run_revision.py').read_text().replace("['V2V_DOMINO']","['V2V_PENDULUM']").replace('all 5 domino cases','all 5 pendulum rope cases')
s=s.replace("# Only publish after every revised case has passed export validation.","""# Verify appearance-only change before publishing.
 import numpy as np
 for c in cases:
  before=D/'samples'/c;after=R/'samples'/c
  a=np.load(before/'raw/trajectories.npz');b=np.load(after/'raw/trajectories.npz');assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
  assert json.loads((before/'camera.json').read_text())==json.loads((after/'camera.json').read_text())
  assert (before/'contacts.json').read_bytes()==(after/'contacts.json').read_bytes()
  report=json.loads((after/'render_report.json').read_text());assert report['materials']['pendulum_rope']['color_srgb']=='#123B78'
 # Only publish after every revised case has passed export validation.""")
s=s.replace(" print('PUBLISHED all 5 pendulum rope cases',flush=True)"," subprocess.run(['/home/gaoya/miniconda3/envs/sam/bin/python',str(R/'refresh_links.py')],check=True)\n print('PUBLISHED all 5 pendulum rope cases',flush=True)")
compile(s,'run_revision.py','exec');(R/'run_revision.py').write_text(s);print(R)
