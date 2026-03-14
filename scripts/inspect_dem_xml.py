import xml.etree.ElementTree as ET
import re

xml_path = r"data/dem_xml/FG-GML-5135-00-06-DEM5A-20250620.xml"


def local(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


tree = ET.parse(xml_path)
root = tree.getroot()

print("=== basic tags ===")
for elem in root.iter():
    name = local(elem.tag)
    if name.lower() in {
        "low", "high", "startpoint", "sequencerule",
        "tuplelist", "gridenvelope", "gridfunction",
        "coveragefunction"
    }:
        txt = (elem.text or "").strip()
        txt1 = txt[:300].replace("\n", "\\n")
        print(f"{name}: {txt1}")

print("\n=== first 10 tupleList lines ===")
tuple_text = None
for elem in root.iter():
    if local(elem.tag).lower() == "tuplelist":
        tuple_text = (elem.text or "").strip()
        break

if tuple_text is None:
    print("tupleList not found")
else:
    lines = [x.strip() for x in tuple_text.splitlines() if x.strip()]
    print("tupleList line count =", len(lines))
    for i, line in enumerate(lines[:10], 1):
        print(f"{i}: {line}")

print("\n=== first 30 tags ===")
count = 0
for elem in root.iter():
    print(local(elem.tag))
    count += 1
    if count >= 30:
        break
