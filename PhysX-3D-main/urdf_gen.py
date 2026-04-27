import os
import json
import xml.etree.ElementTree as ET
import ipdb
import numpy as np

def make_origin_element(xyz, rpy):
    origin = ET.Element('origin')
    origin.set('xyz', ' '.join(xyz))
    origin.set('rpy', ' '.join(rpy))
    return origin

def add_inertial(link_element,xyz="0 0 0"):
    inertial = ET.SubElement(link_element, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=xyz, rpy="0 0 0")
    ET.SubElement(inertial, 'mass', value="1.0")
    ET.SubElement(inertial, 'inertia', ixx="1.0", ixy="0.0", ixz="0.0",
                  iyy="1.0", iyz="0.0", izz="1.0")

def add_fixed_joint(robot, name, parent, child, xyz="0 0 0", rpy="0 0 0"):
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    return joint



basepath='/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/' #your dataset path

urdfpath=os.path.join(basepath,'urdf')
os.makedirs(urdfpath, exist_ok=True)
jsonpath=os.path.join(basepath,'finaljson')
geopath=os.path.join(basepath,'partseg')

namelist=os.listdir(geopath)   
# namelist=namelist[:3]
for index in namelist:
    
            jsonfile=os.path.join(jsonpath,index+'.json')
            urdf_save_path = os.path.join(urdfpath, index+'.urdf')
            build_urdf_from_json(jsonfile, index, geopath, urdf_save_path)


def build_urdf_from_json(jsonfile, index, geopath, urdf_save_path):
    """将单个物体的 group_info 解析为 URDF 并写入文件。

    Args:
        jsonfile (str): 物体对应的 JSON 元数据文件路径。
        index (str): 物体 ID（用于拼接零件路径和命名关节）。
        geopath (str): partseg 目录路径，包含各物体的 objs/ 子目录。
        urdf_save_path (str): 输出 URDF 文件的保存路径。

    Returns:
        bool: True 表示 URDF 已写入，False 表示无可动关节跳过。
    """
    with open(jsonfile, 'r') as fp:
        jsondata = json.load(fp)

    mov = jsondata['group_info']

    robot = ET.Element('robot', name='scene')
    link = ET.SubElement(robot, 'link', name='l_world')
    add_inertial(link)

    save = 1

    # ── 底座零件（组0）：创建 link，链式 fixed joint 连接 ──
    fixlist = mov['0']
    for fixindex in fixlist:
        link = ET.SubElement(robot, 'link', name='l_'+str(fixindex))
        add_inertial(link)
        if os.path.exists(os.path.join(geopath, index, 'objs', str(fixindex)+'.obj')):
            visual = ET.SubElement(link, 'visual')
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(geometry, "mesh",
                          filename=os.path.join('./../partseg', index, 'objs', str(fixindex)+'.obj'),
                          scale="1 1 1")
            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

    for i in range(len(fixlist)-1):
        add_fixed_joint(robot,
                        'joint_fixed_'+str(fixlist[i])+'_'+str(fixlist[i+1]),
                        'l_'+str(fixlist[i]), 'l_'+str(fixlist[i+1]))
    add_fixed_joint(robot, 'joint_fixed_world'+str(fixlist[0]),
                    'l_world', 'l_'+str(fixlist[0]))

    # ── 可动运动组（组1+） ──
    groupnum = len(mov)
    for groupindex in range(1, groupnum):
        fixlist = mov[str(groupindex)][0]

        for fixindex in fixlist:
            link = ET.SubElement(robot, 'link', name='l_'+str(fixindex))
            add_inertial(link)
            if os.path.exists(os.path.join(geopath, index, 'objs', str(fixindex)+'.obj')):
                visual = ET.SubElement(link, 'visual')
                geometry = ET.SubElement(visual, "geometry")
                ET.SubElement(geometry, "mesh",
                              filename=os.path.join('./../partseg', index, 'objs', str(fixindex)+'.obj'),
                              scale="1 1 1")
                ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

        for i in range(len(fixlist)-1):
            add_fixed_joint(robot,
                            'joint_fixed_'+str(fixlist[i])+'_'+str(fixlist[i+1]),
                            'l_'+str(fixlist[i]), 'l_'+str(fixlist[i+1]))

        parent_group_val = mov[mov[str(groupindex)][1]]
        if isinstance(parent_group_val[0], int):
            parentgroupindex = str(parent_group_val[0])
        else:
            parentgroupindex = str(parent_group_val[0][0])

        childgroupindex = fixlist[0]
        parentgroupname = 'l_'+str(parentgroupindex)
        childgroupname  = 'l_'+str(childgroupindex)
        abs_name = 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex)

        abs_link = ET.SubElement(robot, 'link', name=abs_name)
        add_inertial(abs_link)

        joint_type = mov[str(groupindex)][-1]
        params     = mov[str(groupindex)][-2]
                for fixindex in fixlist:
                    link = ET.SubElement(robot, 'link', name='l_'+str(fixindex))
                    add_inertial(link)
                    if os.path.exists(os.path.join(geopath,index,'objs',str(fixindex)+'.obj')):
                        visual = ET.SubElement(link, 'visual')
                        geometry = ET.SubElement(visual, "geometry")
                        ET.SubElement(geometry, "mesh", filename=os.path.join('./../partseg',index,'objs',str(fixindex)+'.obj'), scale="1 1 1")
                        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

                for i in range(len(fixlist)-1):
                    parentname='l_'+str(fixlist[i])
                    childname='l_'+str(fixlist[i+1])
                    add_fixed_joint(robot, 'joint_fixed_'+str(fixlist[i])+'_'+str(fixlist[i+1]), parentname, childname, xyz="0 0 0", rpy="0 0 0")
                add_fixed_joint(robot, 'joint_fixed_world'+str(fixlist[0]), 'l_world', 'l_'+str(fixlist[0]), xyz="0 0 0", rpy="0 0 0")


                groupnum=len(mov)
                for groupindex in range(1,groupnum):
                    fixlist=mov[str(groupindex)][0]
                    for fixindex in fixlist:
                        link = ET.SubElement(robot, 'link', name='l_'+str(fixindex))
                        add_inertial(link)
                        if os.path.exists(os.path.join(geopath,index,'objs',str(fixindex)+'.obj')):
                            visual = ET.SubElement(link, 'visual')
                            geometry = ET.SubElement(visual, "geometry")
                            ET.SubElement(geometry, "mesh", filename=os.path.join('./../partseg',index,'objs',str(fixindex)+'.obj'), scale="1 1 1")
                            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")

                    for i in range(len(fixlist)-1):
                        parentname='l_'+str(fixlist[i])
                        childname='l_'+str(fixlist[i+1])
                        add_fixed_joint(robot, 'joint_fixed_'+str(fixlist[i])+'_'+str(fixlist[i+1]), parentname, childname, xyz="0 0 0", rpy="0 0 0")

                    if isinstance(mov[mov[str(groupindex)][1]][0], int):
                        parentgroupindex=str(mov[mov[str(groupindex)][1]][0])
                    else:
                        parentgroupindex=str(mov[mov[str(groupindex)][1]][0][0])

                    childgroupindex=fixlist[0]
                    parentgroupname='l_'+str(parentgroupindex)
                    childgroupname='l_'+str(childgroupindex)

                    abs_link = ET.SubElement(robot, 'link', name='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))
                    add_inertial(abs_link)
                    


                    if mov[str(groupindex)][-1]=='A':
                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz="0 0 0", rpy="0 0 0")

                        joint = ET.SubElement(robot, "joint", name='joint_free_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="floating")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

                    elif mov[str(groupindex)][-1]=='B':    
                        save+=1
                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz="0 0 0", rpy="0 0 0")

                        xyz=str(mov[str(groupindex)][-2][0])+' '+str(mov[str(groupindex)][-2][1])+' '+str(mov[str(groupindex)][-2][2])

                        joint = ET.SubElement(robot, "joint", name='joint_prismatic_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="prismatic")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz=xyz)  
                        ET.SubElement(joint, "limit", lower=str(mov[str(groupindex)][-2][-2]), upper=str(mov[str(groupindex)][-2][-1]), effort="2000.0", velocity="2.0")

                    elif mov[str(groupindex)][-1]=='C': 
                        save+=1
                        point=str(mov[str(groupindex)][-2][3])+' '+str(mov[str(groupindex)][-2][4])+' '+str(mov[str(groupindex)][-2][5])  
                        pointrev=str(-mov[str(groupindex)][-2][3])+' '+str(-mov[str(groupindex)][-2][4])+' '+str(-mov[str(groupindex)][-2][5])    
                        xyz=str(mov[str(groupindex)][-2][0])+' '+str(mov[str(groupindex)][-2][1])+' '+str(mov[str(groupindex)][-2][2])   

                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz=pointrev, rpy="0 0 0")

                        

                        joint = ET.SubElement(robot, "joint", name='joint_revolute_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz=xyz)  
                        ET.SubElement(joint, "limit", lower=str(mov[str(groupindex)][-2][-2]*np.pi), upper=str(mov[str(groupindex)][-2][-1]*np.pi), effort="2000.0", velocity="2.0")

                    elif mov[str(groupindex)][-1]=='D': 
                        save+=1

                        point=str(mov[str(groupindex)][-2][3])+' '+str(mov[str(groupindex)][-2][4])+' '+str(mov[str(groupindex)][-2][5])  
                        pointrev=str(-mov[str(groupindex)][-2][3])+' '+str(-mov[str(groupindex)][-2][4])+' '+str(-mov[str(groupindex)][-2][5])    
                        xyz=str(mov[str(groupindex)][-2][0])+' '+str(mov[str(groupindex)][-2][1])+' '+str(mov[str(groupindex)][-2][2])   

                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz=pointrev, rpy="0 0 0")
                        
                        abs_linkx = ET.SubElement(robot, 'link', name='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        add_inertial(abs_linkx,pointrev)
                        abs_linkz = ET.SubElement(robot, 'link', name='abstract_z_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        add_inertial(abs_linkz,pointrev)
                        #ipdb.set_trace()

                        joint = ET.SubElement(robot, "joint", name='joint_hinge_y_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_z_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz="0 0 1")  
                        ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

                        joint = ET.SubElement(robot, "joint", name='joint_hinge_z_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link='abstract_z_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "child", link='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz="1 0 0")  
                        ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

                        joint = ET.SubElement(robot, "joint", name='joint_hinge_x_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz="0 1 0")  
                        ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

                        #ipdb.set_trace()
                    elif mov[str(groupindex)][-1]=='CB': 
                        save+=1

                        point=str(mov[str(groupindex)][-2][3])+' '+str(mov[str(groupindex)][-2][4])+' '+str(mov[str(groupindex)][-2][5])  
                        pointrev=str(-mov[str(groupindex)][-2][3])+' '+str(-mov[str(groupindex)][-2][4])+' '+str(-mov[str(groupindex)][-2][5])    
                        xyz=str(mov[str(groupindex)][-2][0])+' '+str(mov[str(groupindex)][-2][1])+' '+str(mov[str(groupindex)][-2][2])
                        xyz1=str(mov[str(groupindex)][-2][8])+' '+str(mov[str(groupindex)][-2][9])+' '+str(mov[str(groupindex)][-2][10])   

                        add_fixed_joint(robot, 'joint_fixed_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), 'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), childgroupname, xyz=pointrev, rpy="0 0 0")
                        
                        
                        abs_linkx = ET.SubElement(robot, 'link', name='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        add_inertial(abs_linkx)

                        joint = ET.SubElement(robot, "joint", name='joint_prim_y_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="prismatic")
                        ET.SubElement(joint, "parent", link=parentgroupname)
                        ET.SubElement(joint, "child", link='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz=xyz1)  
                        ET.SubElement(joint, "limit", lower=str(mov[str(groupindex)][-2][-2]), upper=str(mov[str(groupindex)][-2][-1]), effort="2000.0", velocity="2.0")


                        joint = ET.SubElement(robot, "joint", name='joint_revo_x_'+parentgroupname+'_'+'abstract_'+str(parentgroupindex)+'_'+str(childgroupindex), type="revolute")
                        ET.SubElement(joint, "parent", link='abstract_x_'+str(parentgroupindex)+'_'+str(childgroupindex))
                        ET.SubElement(joint, "child", link='abstract_'+str(parentgroupindex)+'_'+str(childgroupindex))

                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                        ET.SubElement(joint, "axis", xyz=xyz)  
                        ET.SubElement(joint, "limit", lower=str(mov[str(groupindex)][-2][6]*np.pi), upper=str(mov[str(groupindex)][-2][7]*np.pi), effort="2000.0", velocity="2.0")

                    else:
                        print('error type') 
                        print(index)
            if save>0:   
                
                tree = ET.ElementTree(robot)
                ET.indent(tree, space="  ", level=0)
                tree.write(os.path.join(urdfpath,index+'.urdf'), encoding="utf-8", xml_declaration=True)




'''
rm -rf /data/gaoya/AAA_test_video/Dataset_physV/physxnet_genesis_ready/277

python /home/gaoya/Code_Video/Code_data/physxnet_articulation_demo.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --object_id 277 \
  --output_root /data/gaoya/AAA_test_video/Dataset_physV/physxnet_genesis_ready_rigid \
  --run_genesis \
  --object_scale_mult 1.8 \
  --striker_radius 0.08 \
  --ball_posx 1 \
  --solver_family_override rigid
  
  '''

