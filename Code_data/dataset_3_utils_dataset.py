

import os
import json
import xml.etree.ElementTree as ET
from typing import Iterable, Optional
import numpy as np


def add_inertial(link_element, xyz: str = "0 0 0") -> None:
    inertial = ET.SubElement(link_element, "inertial")
    ET.SubElement(inertial, "origin", xyz=xyz, rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="1.0")
    ET.SubElement(
        inertial,
        "inertia",
        ixx="1.0",
        ixy="0.0",
        ixz="0.0",
        iyy="1.0",
        iyz="0.0",
        izz="1.0",
    )


def add_fixed_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz: str = "0 0 0",
    rpy: str = "0 0 0",
) -> ET.Element:
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    return joint


def add_mesh_visual(
    link: ET.Element,
    part_id: int,
    index: str,
    geopath: str,
    mesh_rel_root: str = "./../partseg",
) -> None:
    mesh_abs_path = os.path.join(geopath, index, "objs", f"{part_id}.obj")
    if not os.path.exists(mesh_abs_path):
        return

    visual = ET.SubElement(link, "visual")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(
        geometry,
        "mesh",
        filename=os.path.join(mesh_rel_root, index, "objs", f"{part_id}.obj"),
        scale="1 1 1",
    )
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")


def add_link_with_visual(
    robot: ET.Element,
    part_id: int,
    index: str,
    geopath: str,
    mesh_rel_root: str = "./../partseg",
) -> None:
    link = ET.SubElement(robot, "link", name=f"l_{part_id}")
    add_inertial(link)
    add_mesh_visual(link, part_id, index, geopath, mesh_rel_root=mesh_rel_root)


def chain_fixed_links(
    robot: ET.Element,
    part_ids,
) -> None:
    for i in range(len(part_ids) - 1):
        parent_name = f"l_{part_ids[i]}"
        child_name = f"l_{part_ids[i + 1]}"
        add_fixed_joint(
            robot,
            f"joint_fixed_{part_ids[i]}_{part_ids[i + 1]}",
            parent_name,
            child_name,
            xyz="0 0 0",
            rpy="0 0 0",
        )


def resolve_parent_group_index(mov: dict, groupindex: int) -> str:
    parent_group_key = mov[str(groupindex)][1]
    parent_group_first = mov[parent_group_key][0]

    if isinstance(parent_group_first, int):
        return str(parent_group_first)
    return str(parent_group_first[0])


def urdf_gen(
    jsondata: dict,
    index: str,
    geopath: str,
    robot_name: str = "scene",
    mesh_rel_root: str = "./../partseg",
) -> ET.Element:
    """
    根据单个 jsondata 生成 URDF 的 <robot> XML Element。
    不写文件，只返回 robot，便于外部调用。
    """
    mov = jsondata["group_info"]

    robot = ET.Element("robot", name=robot_name)
    world_link = ET.SubElement(robot, "link", name="l_world")
    add_inertial(world_link)

    # 保留你原代码中的 save 语义
    save = 1

    # group 0: 固定基座组
    fixlist = mov["0"]
    for fixindex in fixlist:
        add_link_with_visual(robot, fixindex, index, geopath, mesh_rel_root=mesh_rel_root)

    chain_fixed_links(robot, fixlist)
    add_fixed_joint(
        robot,
        f"joint_fixed_world{fixlist[0]}",
        "l_world",
        f"l_{fixlist[0]}",
        xyz="0 0 0",
        rpy="0 0 0",
    )

    # 如果只有一个 group，直接返回
    if len(mov) == 1:
        return robot

    # 其余可动组
    groupnum = len(mov)
    for groupindex in range(1, groupnum):
        group = mov[str(groupindex)]
        fixlist = group[0]

        for fixindex in fixlist:
            add_link_with_visual(robot, fixindex, index, geopath, mesh_rel_root=mesh_rel_root)

        chain_fixed_links(robot, fixlist)

        parentgroupindex = resolve_parent_group_index(mov, groupindex)
        childgroupindex = fixlist[0]
        parentgroupname = f"l_{parentgroupindex}"
        childgroupname = f"l_{childgroupindex}"

        abs_name = f"abstract_{parentgroupindex}_{childgroupindex}"
        abs_link = ET.SubElement(robot, "link", name=abs_name)
        add_inertial(abs_link)

        joint_type = group[-1]

        if joint_type == "A":
            add_fixed_joint(
                robot,
                f"joint_fixed_{abs_name}",
                abs_name,
                childgroupname,
                xyz="0 0 0",
                rpy="0 0 0",
            )

            joint = ET.SubElement(
                robot,
                "joint",
                name=f"joint_free_{parentgroupname}_{abs_name}",
                type="floating",
            )
            ET.SubElement(joint, "parent", link=parentgroupname)
            ET.SubElement(joint, "child", link=abs_name)
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

        elif joint_type == "B":
            save += 1
            add_fixed_joint(
                robot,
                f"joint_fixed_{abs_name}",
                abs_name,
                childgroupname,
                xyz="0 0 0",
                rpy="0 0 0",
            )

            xyz = f"{group[-2][0]} {group[-2][1]} {group[-2][2]}"

            joint = ET.SubElement(
                robot,
                "joint",
                name=f"joint_prismatic_{parentgroupname}_{abs_name}",
                type="prismatic",
            )
            ET.SubElement(joint, "parent", link=parentgroupname)
            ET.SubElement(joint, "child", link=abs_name)
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz=xyz)
            ET.SubElement(
                joint,
                "limit",
                lower=str(group[-2][-2]),
                upper=str(group[-2][-1]),
                effort="2000.0",
                velocity="2.0",
            )

        elif joint_type == "C":
            save += 1
            point = f"{group[-2][3]} {group[-2][4]} {group[-2][5]}"
            pointrev = f"{-group[-2][3]} {-group[-2][4]} {-group[-2][5]}"
            xyz = f"{group[-2][0]} {group[-2][1]} {group[-2][2]}"

            add_fixed_joint(
                robot,
                f"joint_fixed_{abs_name}",
                abs_name,
                childgroupname,
                xyz=pointrev,
                rpy="0 0 0",
            )

            joint = ET.SubElement(
                robot,
                "joint",
                name=f"joint_revolute_{parentgroupname}_{abs_name}",
                type="revolute",
            )
            ET.SubElement(joint, "parent", link=parentgroupname)
            ET.SubElement(joint, "child", link=abs_name)
            ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz=xyz)
            ET.SubElement(
                joint,
                "limit",
                lower=str(group[-2][-2] * np.pi),
                upper=str(group[-2][-1] * np.pi),
                effort="2000.0",
                velocity="2.0",
            )

        elif joint_type == "D":
            save += 1
            point = f"{group[-2][3]} {group[-2][4]} {group[-2][5]}"
            pointrev = f"{-group[-2][3]} {-group[-2][4]} {-group[-2][5]}"

            add_fixed_joint(
                robot,
                f"joint_fixed_{abs_name}",
                abs_name,
                childgroupname,
                xyz=pointrev,
                rpy="0 0 0",
            )

            abs_linkx = ET.SubElement(robot, "link", name=f"abstract_x_{parentgroupindex}_{childgroupindex}")
            add_inertial(abs_linkx, pointrev)

            abs_linkz = ET.SubElement(robot, "link", name=f"abstract_z_{parentgroupindex}_{childgroupindex}")
            add_inertial(abs_linkz, pointrev)

            joint = ET.SubElement(
                robot,
                "joint",
                name=f"joint_hinge_y_{parentgroupname}_{abs_name}",
                type="revolute",
            )
            ET.SubElement(joint, "parent", link=parentgroupname)
            ET.SubElement(joint, "child", link=f"abstract_z_{parentgroupindex}_{childgroupindex}")
            ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz="0 0 1")
            ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

            joint = ET.SubElement(
                robot,
                "joint",
                name=f"joint_hinge_z_{parentgroupname}_{abs_name}",
                type="revolute",
            )
            ET.SubElement(joint, "parent", link=f"abstract_z_{parentgroupindex}_{childgroupindex}")
            ET.SubElement(joint, "child", link=f"abstract_x_{parentgroupindex}_{childgroupindex}")
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz="1 0 0")
            ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

            joint = ET.SubElement(
                robot,
                "joint",
                name=f"joint_hinge_x_{parentgroupname}_{abs_name}",
                type="revolute",
            )
            ET.SubElement(joint, "parent", link=f"abstract_x_{parentgroupindex}_{childgroupindex}")
            ET.SubElement(joint, "child", link=abs_name)
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz="0 1 0")
            ET.SubElement(joint, "limit", lower=str(-np.pi), upper=str(np.pi), effort="2000.0", velocity="2.0")

        elif joint_type == "CB":
            save += 1
            point = f"{group[-2][3]} {group[-2][4]} {group[-2][5]}"
            pointrev = f"{-group[-2][3]} {-group[-2][4]} {-group[-2][5]}"
            xyz = f"{group[-2][0]} {group[-2][1]} {group[-2][2]}"
            xyz1 = f"{group[-2][8]} {group[-2][9]} {group[-2][10]}"

            add_fixed_joint(
                robot,
                f"joint_fixed_{abs_name}",
                abs_name,
                childgroupname,
                xyz=pointrev,
                rpy="0 0 0",
            )

            abs_linkx = ET.SubElement(robot, "link", name=f"abstract_x_{parentgroupindex}_{childgroupindex}")
            add_inertial(abs_linkx)

            joint = ET.SubElement(
                robot,
                "joint",
                name=f"joint_prim_y_{parentgroupname}_{abs_name}",
                type="prismatic",
            )
            ET.SubElement(joint, "parent", link=parentgroupname)
            ET.SubElement(joint, "child", link=f"abstract_x_{parentgroupindex}_{childgroupindex}")
            ET.SubElement(joint, "origin", xyz=point, rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz=xyz1)
            ET.SubElement(
                joint,
                "limit",
                lower=str(group[-2][-2]),
                upper=str(group[-2][-1]),
                effort="2000.0",
                velocity="2.0",
            )

            joint = ET.SubElement(
                robot,
                "joint",
                name=f"joint_revo_x_{parentgroupname}_{abs_name}",
                type="revolute",
            )
            ET.SubElement(joint, "parent", link=f"abstract_x_{parentgroupindex}_{childgroupindex}")
            ET.SubElement(joint, "child", link=abs_name)
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz=xyz)
            ET.SubElement(
                joint,
                "limit",
                lower=str(group[-2][6] * np.pi),
                upper=str(group[-2][7] * np.pi),
                effort="2000.0",
                velocity="2.0",
            )

        else:
            raise ValueError(f"Unknown joint type: {joint_type}, index={index}")

    if save <= 0:
        raise RuntimeError(f"Invalid URDF state for index={index}")

    return robot


def write_urdf(robot: ET.Element, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def build_urdf_from_json_file(
    jsonfile: str,
    index: str,
    geopath: str,
    output_path: Optional[str] = None,
    robot_name: str = "scene",
    mesh_rel_root: str = "./../partseg",
) -> ET.Element:
    with open(jsonfile, "r") as fp:
        jsondata = json.load(fp)

    robot = urdf_gen(
        jsondata=jsondata,
        index=index,
        geopath=geopath,
        robot_name=robot_name,
        mesh_rel_root=mesh_rel_root,
    )

    if output_path is not None:
        write_urdf(robot, output_path)

    return robot


def batch_generate_urdfs(
    basepath: str,
    ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    robot_name: str = "scene",
    mesh_rel_root: str = "./../partseg",
) -> None:
    urdfpath = os.path.join(basepath, "urdf2")
    jsonpath = os.path.join(basepath, "finaljson")
    geopath = os.path.join(basepath, "partseg")

    os.makedirs(urdfpath, exist_ok=True)

    if ids is None:
        ids = sorted(os.listdir(geopath))

    ids = list(ids)
    if limit is not None:
        ids = ids[:limit]

    for index in ids:
        jsonfile = os.path.join(jsonpath, f"{index}.json")
        output_path = os.path.join(urdfpath, f"{index}.urdf")

        if not os.path.exists(jsonfile):
            print(f"[Skip] json not found: {jsonfile}")
            continue


        build_urdf_from_json_file(
            jsonfile=jsonfile,
            index=index,
            geopath=geopath,
            output_path=output_path,
            robot_name=robot_name,
            mesh_rel_root=mesh_rel_root,
        )
        print(f"[OK] {output_path}")


