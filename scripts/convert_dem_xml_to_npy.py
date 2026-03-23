import argparse
import json
from pathlib import Path

import numpy as np
from lxml import etree


def lname(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def find_first_elem(root, local_name: str):
    for elem in root.iter():
        if lname(elem.tag) == local_name:
            return elem
    return None


def find_first_text(root, local_name: str):
    elem = find_first_elem(root, local_name)
    if elem is None or elem.text is None:
        return None
    return elem.text.strip()


def parse_envelope(root):
    lower = find_first_text(root, "lowerCorner")
    upper = find_first_text(root, "upperCorner")
    if lower is None or upper is None:
        return None, None
    return [float(x) for x in lower.split()], [float(x) for x in upper.split()]


def parse_grid_shape(root):
    low = find_first_text(root, "low")
    high = find_first_text(root, "high")
    if low is None or high is None:
        raise ValueError("low/high not found")

    low = [int(x) for x in low.split()]
    high = [int(x) for x in high.split()]

    ncols = high[0] - low[0] + 1
    nrows = high[1] - low[1] + 1
    return low, high, nrows, ncols


def parse_start_point(root):
    txt = find_first_text(root, "startPoint")
    if txt is None:
        return [0, 0]
    return [int(x) for x in txt.split()]


def parse_sequence_rule(root):
    elem = find_first_elem(root, "sequenceRule")
    rule_text = elem.text.strip() if elem is not None and elem.text else None
    order_attr = elem.attrib.get("order") if elem is not None else None
    return rule_text, order_attr


def parse_tuple_values(root):
    tuple_text = find_first_text(root, "tupleList")
    if tuple_text is None:
        raise ValueError("tupleList not found")

    # この XML は基本的に「1行1点」。
    # 例: 地表面,23.16 / 海水面,-9999. / データなし,-9999.
    values = []
    for line in tuple_text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        val = parts[-1].strip()

        if val in {"", "NaN", "nan", "NODATA", "nodata", "-9999", "-9999.", "-9999.0"}:
            values.append(np.nan)
        else:
            try:
                values.append(float(val))
            except Exception:
                values.append(np.nan)

    return np.array(values, dtype=np.float32)


def fill_grid_from_sequence(values, nrows, ncols, start_point, order_attr):
    # order="+x-y" を前提
    if order_attr not in {None, "+x-y"}:
        raise ValueError(f"Unsupported sequenceRule order: {order_attr}")

    x0, y0 = start_point
    arr = np.full((nrows, ncols), np.nan, dtype=np.float32)

    x = x0
    y = y0

    for v in values:
        if y >= nrows:
            break
        arr[y, x] = v
        x += 1
        if x >= ncols:
            x = 0
            y += 1

    return arr


def parse_xml(xml_path: Path):
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    low, high, nrows, ncols = parse_grid_shape(root)
    lower_corner, upper_corner = parse_envelope(root)
    mesh = find_first_text(root, "mesh")
    dem_type = find_first_text(root, "type")
    start_point = parse_start_point(root)
    rule_text, order_attr = parse_sequence_rule(root)
    values = parse_tuple_values(root)

    arr = fill_grid_from_sequence(
        values=values,
        nrows=nrows,
        ncols=ncols,
        start_point=start_point,
        order_attr=order_attr,
    )

    meta = {
        "xml_file": str(xml_path),
        "mesh": mesh,
        "type": dem_type,
        "low": low,
        "high": high,
        "shape": [nrows, ncols],
        "lower_corner": lower_corner,
        "upper_corner": upper_corner,
        "start_point": start_point,
        "sequence_rule": rule_text,
        "sequence_order": order_attr,
        "tuple_count": int(values.size),
        "nan_count": int(np.isnan(arr).sum()),
        "min": None if np.all(np.isnan(arr)) else float(np.nanmin(arr)),
        "max": None if np.all(np.isnan(arr)) else float(np.nanmax(arr)),
    }
    return arr, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {input_dir}")

    for xml_path in xml_files:
        try:
            arr, meta = parse_xml(xml_path)

            stem = xml_path.stem
            npy_path = output_dir / f"{stem}.npy"
            meta_path = output_dir / f"{stem}.json"

            np.save(npy_path, arr)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(
                f"[OK] {xml_path.name} -> {npy_path.name} "
                f"shape={arr.shape} start={meta['start_point']} "
                f"tuple_count={meta['tuple_count']} nan={meta['nan_count']} "
                f"min={meta['min']} max={meta['max']}"
            )
        except Exception as e:
            print(f"[ERR] {xml_path.name}: {e}")


if __name__ == "__main__":
    main()
