import os
import glob
import re
import numpy as np
import xml.etree.ElementTree as ET

IN_DIR = r"data/dem_xml"
OUT_DIR = r"data/dem_npy"

os.makedirs(OUT_DIR, exist_ok=True)


def find_tuple_list_text(root):
    for elem in root.iter():
        tag = elem.tag.lower()
        if "tuplelist" in tag:
            return elem.text
    return None


def parse_dem_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tuple_text = find_tuple_list_text(root)
    if tuple_text is None:
        raise RuntimeError(f"tupleList not found: {xml_path}")

    z_values = []

    for raw_line in tuple_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 区切りは空白 or カンマの両対応
        parts = re.split(r"[,\s]+", line)

        # 国土地理院GMLでは末尾が標高のことが多い
        # 例: "x y z" や "x,y,z"
        try:
            z = float(parts[-1])
            z_values.append(z)
        except ValueError:
            continue

    if len(z_values) == 0:
        raise RuntimeError(f"No elevation values parsed: {xml_path}")

    arr = np.array(z_values, dtype=np.float32)
    return arr


def main():
    xml_files = sorted(glob.glob(os.path.join(IN_DIR, "*.xml")))
    if not xml_files:
        print("No XML files found.")
        return

    print(f"[info] found {len(xml_files)} XML files")

    for xml_path in xml_files:
        dem = parse_dem_xml(xml_path)

        out_name = os.path.basename(xml_path).replace(".xml", ".npy")
        out_path = os.path.join(OUT_DIR, out_name)

        np.save(out_path, dem)

        print(
            f"[saved] {out_path} "
            f"shape={dem.shape} min={dem.min():.3f} max={dem.max():.3f}"
        )

    print("[done]")


if __name__ == "__main__":
    main()
