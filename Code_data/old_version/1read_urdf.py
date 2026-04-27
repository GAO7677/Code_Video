import xml.etree.ElementTree as ET
import math
from pathlib import Path


def parse_float_list(s):
    if s is None:
        return None
    return [float(x) for x in s.strip().split()]


def nearly_equal_list(a, b, eps=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= eps for x, y in zip(a, b))


def elem_attrib(elem, key, default=None):
    return elem.attrib.get(key, default) if elem is not None else default


def find_child(elem, tag):
    return elem.find(tag) if elem is not None else None


def normalize_link(link):
    name = link.attrib.get("name")

    inertial = link.find("inertial")
    inertial_data = None
    if inertial is not None:
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        inertial_data = {
            "origin_xyz": parse_float_list(elem_attrib(origin, "xyz")),
            "origin_rpy": parse_float_list(elem_attrib(origin, "rpy")),
            "mass": float(elem_attrib(mass, "value")) if mass is not None else None,
            "ixx": elem_attrib(inertia, "ixx"),
            "ixy": elem_attrib(inertia, "ixy"),
            "ixz": elem_attrib(inertia, "ixz"),
            "iyy": elem_attrib(inertia, "iyy"),
            "iyz": elem_attrib(inertia, "iyz"),
            "izz": elem_attrib(inertia, "izz"),
        }

    visuals = []
    for v in link.findall("visual"):
        origin = v.find("origin")
        geometry = v.find("geometry")
        material = v.find("material")

        geom_data = {}
        if geometry is not None:
            for tag in ["mesh", "box", "cylinder", "sphere"]:
                g = geometry.find(tag)
                if g is not None:
                    geom_data["type"] = tag
                    geom_data["attrib"] = dict(sorted(g.attrib.items()))
                    break

        visuals.append({
            "origin_xyz": parse_float_list(elem_attrib(origin, "xyz")),
            "origin_rpy": parse_float_list(elem_attrib(origin, "rpy")),
            "geometry": geom_data,
            "material_name": elem_attrib(material, "name"),
        })

    collisions = []
    for c in link.findall("collision"):
        origin = c.find("origin")
        geometry = c.find("geometry")

        geom_data = {}
        if geometry is not None:
            for tag in ["mesh", "box", "cylinder", "sphere"]:
                g = geometry.find(tag)
                if g is not None:
                    geom_data["type"] = tag
                    geom_data["attrib"] = dict(sorted(g.attrib.items()))
                    break

        collisions.append({
            "origin_xyz": parse_float_list(elem_attrib(origin, "xyz")),
            "origin_rpy": parse_float_list(elem_attrib(origin, "rpy")),
            "geometry": geom_data,
        })

    return {
        "name": name,
        "inertial": inertial_data,
        "visuals": visuals,
        "collisions": collisions,
    }


def normalize_joint(joint):
    name = joint.attrib.get("name")
    jtype = joint.attrib.get("type")

    parent = joint.find("parent")
    child = joint.find("child")
    origin = joint.find("origin")
    axis = joint.find("axis")
    limit = joint.find("limit")

    return {
        "name": name,
        "type": jtype,
        "parent": elem_attrib(parent, "link"),
        "child": elem_attrib(child, "link"),
        "origin_xyz": parse_float_list(elem_attrib(origin, "xyz")),
        "origin_rpy": parse_float_list(elem_attrib(origin, "rpy")),
        "axis_xyz": parse_float_list(elem_attrib(axis, "xyz")),
        "limit": dict(sorted(limit.attrib.items())) if limit is not None else None,
    }


def parse_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError(f"{path} 不是合法 URDF: 根节点不是 <robot>")

    robot_name = root.attrib.get("name")
    links = {}
    joints = {}

    for link in root.findall("link"):
        d = normalize_link(link)
        links[d["name"]] = d

    for joint in root.findall("joint"):
        d = normalize_joint(joint)
        joints[d["name"]] = d

    return {
        "robot_name": robot_name,
        "links": links,
        "joints": joints,
    }


def compare_dict(a, b, prefix=""):
    diffs = []

    a_keys = set(a.keys())
    b_keys = set(b.keys())

    for k in sorted(a_keys - b_keys):
        diffs.append(f"{prefix}{k}: 只在A中存在")
    for k in sorted(b_keys - a_keys):
        diffs.append(f"{prefix}{k}: 只在B中存在")

    for k in sorted(a_keys & b_keys):
        va, vb = a[k], b[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(compare_dict(va, vb, prefix=f"{prefix}{k}."))
        else:
            if va != vb:
                diffs.append(f"{prefix}{k}: A={va} | B={vb}")

    return diffs


def compare_urdf(path_a, path_b):
    a = parse_urdf(path_a)
    b = parse_urdf(path_b)

    diffs = []

    if a["robot_name"] != b["robot_name"]:
        diffs.append(f"robot_name不同: A={a['robot_name']} | B={b['robot_name']}")

    diffs.extend(compare_dict(a["links"], b["links"], prefix="links."))
    diffs.extend(compare_dict(a["joints"], b["joints"], prefix="joints."))
    pprint.pprint(a)
    pprint.pprint(b)
    return diffs


if __name__ == "__main__":
    import sys
    import pprint
    path_b = "/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/urdf/14620.urdf"
    b = parse_urdf(path_b)
    pprint.pprint(b)



'''
python /home/gaoya/Code_Video/Code_data/1.py /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/urdf2/14620.urdf /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/urdf/14620.urdf 
'''