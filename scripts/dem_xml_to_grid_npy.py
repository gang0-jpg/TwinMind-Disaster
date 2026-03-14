import os
import glob
import re
import numpy as np
import xml.etree.ElementTree as ET

IN_DIR = r"data/dem_xml"
OUT_DIR = r"data/dem_npy_grid"

os.makedirs(OUT_DIR, exist_ok=True)


def get_localname(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_text_by_localname(root, name: str):
    for elem in root.iter():
        if get_localname(elem.tag).lower() == name.lower():
            if elem.text:
                return elem.text.strip()
    return None


def parse_low_high(root):
    low_txt = find_text_by_localname(root, "low")
    high_txt = find_text_by_localname(root, "high")

    if low_txt is None or high_txt is None:
        raise RuntimeError("low/high not found")

    low_parts = re.split(r"[\s,]+", low_txt.strip())
    high_parts = re.split(r"[\s,]+", high_txt.strip())

    x0, y0 = int(low_parts[0]), int(low_parts[1])
    x1, y1 = int(high_parts[0]), int(high_parts[1])

    cols = x1 - x0 + 1
    rows = y1 - y0 + 1

    return x0, y0, cols, rows


def parse_start_point(root):
    txt = find_text_by_localname(root, "startPoint")
    if txt is None:
        raise RuntimeError("startPoint not found")

    parts = re.split(r"[\s,]+", txt.strip())
    sx = int(parts[0])
    sy = int(parts[1])
    return sx, sy


def parse_sequence_rule(root):
    txt = find_text_by_localname(root, "sequenceRule")
    if txt is None:
        return "Linear"
    return txt.strip()


def find_tuple_list_text(root):
    for elem in root.iter():
        if get_localname(elem.tag).lower() == "tuplelist":
            return elem.text
    return None


def parse_tuplelist(tuple_text):
    z_values = []

    for raw_line in tuple_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 例:
        # 海水面,-9999.
        # 地表面,12.34
        parts = [p.strip() for p in line.split(",")]

        if len(parts) < 2:
            continue

        z_txt = parts[-1]

        # 末尾の "12." を float に通す
        try:
            z = float(z_txt)
        except ValueError:
            # 念のため空白区切りにも対応
            parts2 = re.split(r"[\s,]+", line)
            try:
                z = float(parts2[-1])
            except ValueError:
                continue

        z_values.append(z)

    if not z_values:
        raise RuntimeError("no z values parsed from tupleList")

    return np.array(z_values, dtype=np.float32)


def fill_grid_linear(rows, cols, sx, sy, values):
    """
    startPoint=(sx, sy) から row-major に順番に埋める
    x: 0..cols-1
    y: 0..rows-1
    """
    grid = np.full((rows, cols), np.nan, dtype=np.float32)

    x = sx
    y = sy

    for v in values:
        if 0 <= y < rows and 0 <= x < cols:
            grid[y, x] = v
        else:
            raise RuntimeError(f"index out of range while filling: x={x}, y={y}")

        x += 1
        if x >= cols:
            x = 0
            y += 1

        if y >= rows and len(values) > 0 and v is not values[-1]:
            # 途中であふれる場合
            break

    return grid


def parse_dem_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    _, _, cols, rows = parse_low_high(root)
    sx, sy = parse_start_point(root)
    seq = parse_sequence_rule(root)

    if seq.lower() != "linear":
        raise RuntimeError(f"unsupported sequenceRule: {seq}")

    tuple_text = find_tuple_list_text(root)
    if tuple_text is None:
        raise RuntimeError("tupleList not found")

    values = parse_tuplelist(tuple_text)

    dem = fill_grid_linear(rows, cols, sx, sy, values)

    # NoData を NaN に
    dem[dem <= -9999] = np.nan

    return dem, rows, cols, sx, sy, len(values)


def main():
    xml_files = sorted(glob.glob(os.path.join(IN_DIR, "*.xml")))
    print(f"[info] found {len(xml_files)} XML files")

    for xml_path in xml_files:
        dem, rows, cols, sx, sy, nvals = parse_dem_xml(xml_path)

        out_name = os.path.basename(xml_path).replace(".xml", ".npy")
        out_path = os.path.join(OUT_DIR, out_name)

        np.save(out_path, dem)

        valid = dem[np.isfinite(dem)]
        valid_count = valid.size

        if valid_count > 0:
            vmin = np.nanmin(valid)
            vmax = np.nanmax(valid)
            print(
                f"[saved] {out_path} "
                f"shape=({rows},{cols}) "
                f"start=({sx},{sy}) "
                f"tuple_count={nvals} "
                f"valid={valid_count} "
                f"min={vmin:.3f} max={vmax:.3f}"
            )
        else:
            print(
                f"[saved] {out_path} "
                f"shape=({rows},{cols}) "
                f"start=({sx},{sy}) "
                f"tuple_count={nvals} "
                f"valid=0"
            )

    print("[done]")


if __name__ == "__main__":
    main()
