import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_erosion, distance_transform_edt

st.set_page_config(layout="wide")

st.title("TwinMind Disaster")
st.caption("Observation OS for Disaster Resilience")

# -------------------------------
# sidebar
# -------------------------------
cases_dir = Path("data/cases")
case_files = sorted(list(cases_dir.glob("case_*.npz")))
case_names = [f.name for f in case_files]

if not case_files:
    st.error("No case files found in data/cases")
    st.stop()

case_selected = st.sidebar.selectbox("Case", case_names)

top_k = st.sidebar.slider("Sensor count", 1, 20, 8)
min_dist = st.sidebar.slider("Sensor minimum distance", 5, 40, 20)

lidar_k = st.sidebar.slider("LiDAR scan zones", 1, 10, 3)
lidar_dist = st.sidebar.slider("LiDAR minimum distance", 10, 60, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("**TwinMind = Observation OS**")

# -------------------------------
# helper: EIG-like greedy selection
# -------------------------------
def select_eig_points(score_map, valid_mask, top_k=8, min_dist=20, suppression_radius=18):
    score = score_map.copy().astype(np.float32)
    score[valid_mask == 0] = -1.0

    h, w = score.shape
    selected = []

    yy, xx = np.indices((h, w))

    for _ in range(top_k):
        idx = np.argmax(score)
        best_score = score.flat[idx]

        if best_score <= 0:
            break

        y, x = np.unravel_index(idx, score.shape)
        selected.append((y, x))

        dist2 = (yy - y) ** 2 + (xx - x) ** 2

        # 周辺は情報取得済みとして減衰
        local_mask = dist2 <= suppression_radius ** 2
        score[local_mask] *= 0.15

        # 最低距離以内は採用しない
        hard_mask = dist2 <= min_dist ** 2
        score[hard_mask] = -1.0

    return selected


# -------------------------------
# helper: plotting 4 panels
# -------------------------------
def plot_map(data, title, cmap="viridis", sensors=None, lidar_zones=None, roads=None):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(data, cmap=cmap)

    if roads is not None:
        road_overlay = np.ma.masked_where(roads == 0, roads)
        ax.imshow(road_overlay, cmap="gray", alpha=0.25, zorder=2)

    if sensors is not None:
        ys, xs = np.where(sensors > 0)
        ax.scatter(xs, ys, c="red", s=35, label="Sensor", zorder=3)

    if lidar_zones is not None:
        for i, (y, x) in enumerate(lidar_zones, start=1):
            circ = plt.Circle(
                (x, y),
                radius=12,
                edgecolor="cyan",
                facecolor="none",
                linewidth=2.0,
                linestyle="--",
                zorder=4,
            )
            ax.add_patch(circ)
            ax.text(
                x,
                y - 15,
                f"LiDAR {i}",
                color="cyan",
                fontsize=8,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5),
                zorder=5,
            )

    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


# -------------------------------
# helper: integrated digital twin map
# -------------------------------
def plot_digital_twin_map(dem, pred, road_mask=None, sensors=None, lidar_zones=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    # DEM background
    ax.imshow(dem, cmap="terrain")

    # Flood overlay
    flood_overlay = np.ma.masked_where(pred <= 0.001, pred)
    ax.imshow(flood_overlay, cmap="Blues", alpha=0.45)

    # Roads overlay
    if road_mask is not None:
        road_overlay = np.ma.masked_where(road_mask == 0, road_mask)
        ax.imshow(road_overlay, cmap="gray", alpha=0.35)

    # Sensors
    if sensors is not None:
        ys, xs = np.where(sensors > 0)
        ax.scatter(xs, ys, c="red", s=45, label="Fixed Sensor", zorder=4)

        for i, (y, x) in enumerate(zip(ys, xs), start=1):
            ax.text(
                x + 2,
                y + 2,
                f"S{i}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.5, edgecolor="none", pad=1.2),
                zorder=5,
            )

    # LiDAR zones
    if lidar_zones is not None:
        for i, (y, x) in enumerate(lidar_zones, start=1):
            circ = plt.Circle(
                (x, y),
                radius=14,
                edgecolor="cyan",
                facecolor="none",
                linewidth=2.0,
                linestyle="--",
                zorder=6,
            )
            ax.add_patch(circ)

            ax.text(
                x,
                y - 16,
                f"L{i}",
                color="cyan",
                fontsize=8,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5),
                zorder=7,
            )

    ax.set_title("Digital Twin Map: Terrain + Flood + Roads + Sensors + LiDAR")
    ax.axis("off")
    return fig


# -------------------------------
# load case
# -------------------------------
data = np.load(cases_dir / case_selected)

dem = data["dem"].astype(np.float32)
rain = data["rain"].astype(np.float32)
flood_gt = data["flood_gt"].astype(np.float32)
mask = data["mask"].astype(np.uint8)

# -------------------------------
# demo prediction
# 後で本物モデル出力に差し替え可能
# -------------------------------
rng = np.random.default_rng(42)

pred = flood_gt + rng.normal(0, 0.002, flood_gt.shape)
pred = np.clip(pred, 0, None) * mask

# -------------------------------
# uncertainty (display purpose)
# -------------------------------
gy, gx = np.gradient(pred)
uncertainty = np.sqrt(gx**2 + gy**2)
uncertainty *= mask

# -------------------------------
# demo road network
# 本番では GIS / OSM / 国土地理院道路データに置換
# -------------------------------
road_mask = np.zeros_like(mask, dtype=np.uint8)

h, w = mask.shape

# 縦道路
road_mask[:, int(w * 0.15):int(w * 0.15) + 3] = 1

# 斜め道路
for i in range(h):
    x = int(w * 0.65 + 0.15 * (i - h / 2))
    if 2 <= x < w - 2:
        road_mask[max(0, i-1):min(h, i+2), max(0, x-1):min(w, x+2)] = 1

# 横道路
road_mask[int(h * 0.78):int(h * 0.78) + 3, int(w * 0.25):int(w * 0.75)] = 1

# -------------------------------
# road proximity weight
# 道路に近いほど重みを高くする
# -------------------------------
road_distance = distance_transform_edt(road_mask == 0)
road_weight = np.exp(-road_distance / 12.0)
road_weight[road_distance > 25] *= 0.1

# -------------------------------
# valid mask for sensor placement
# 外周/境界アーチファクト回避
# -------------------------------
inner_mask = binary_erosion(mask.astype(bool), iterations=8).astype(np.uint8)

# 洪水がある程度ある場所を優先
flood_area = (pred > pred.mean() * 0.3).astype(np.uint8)

# 道路から近い候補だけ使う
road_candidate_mask = (road_distance <= 20).astype(np.uint8)

sensor_valid_mask = inner_mask * flood_area * road_candidate_mask

# -------------------------------
# sensor placement (road-aware EIG-like)
# -------------------------------
sensor_score = uncertainty * flood_area * road_weight

sensor_points = select_eig_points(
    score_map=sensor_score,
    valid_mask=sensor_valid_mask,
    top_k=top_k,
    min_dist=min_dist,
    suppression_radius=max(10, min_dist - 2),
)

sensor_map = np.zeros_like(dem, dtype=np.uint8)
for y, x in sensor_points:
    sensor_map[y, x] = 1

# -------------------------------
# LiDAR scan zones
# 洪水境界寄り + 外周除外 + diversified placement
# -------------------------------
gy2, gx2 = np.gradient(pred)
boundary_score = np.sqrt(gx2**2 + gy2**2)

flood_presence = (pred > pred.mean() * 0.3).astype(np.float32)
lidar_score = boundary_score * flood_presence

margin = 20
lidar_valid_mask = np.ones_like(mask, dtype=np.uint8)
lidar_valid_mask[:margin, :] = 0
lidar_valid_mask[-margin:, :] = 0
lidar_valid_mask[:, :margin] = 0
lidar_valid_mask[:, -margin:] = 0
lidar_valid_mask = lidar_valid_mask * inner_mask * flood_presence.astype(np.uint8)

lidar_points = select_eig_points(
    score_map=lidar_score,
    valid_mask=lidar_valid_mask,
    top_k=lidar_k,
    min_dist=lidar_dist,
    suppression_radius=max(12, lidar_dist - 4),
)

# -------------------------------
# UI layout: 4 panels
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Terrain / DEM")
    st.pyplot(plot_map(dem, "DEM", "terrain", roads=road_mask), clear_figure=True)

with col2:
    st.subheader("Flood Prediction")
    st.pyplot(plot_map(pred, "Flood Prediction", "Blues", roads=road_mask), clear_figure=True)

with col3:
    st.subheader("Uncertainty")
    st.pyplot(
        plot_map(
            uncertainty,
            "Uncertainty Map",
            "magma",
            lidar_zones=lidar_points,
            roads=road_mask,
        ),
        clear_figure=True,
    )

with col4:
    st.subheader("Sensor Placement")
    st.pyplot(
        plot_map(
            uncertainty,
            "Road-aware Sensors + LiDAR",
            "magma",
            sensors=sensor_map,
            lidar_zones=lidar_points,
            roads=road_mask,
        ),
        clear_figure=True,
    )

# -------------------------------
# bottom info
# -------------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown("### Scenario Summary")
    st.write("Case:", case_selected)
    st.write("Rain timesteps:", rain.shape[0])
    st.write("Sensor count:", top_k)
    st.write("LiDAR scan zones:", lidar_k)
    if mask.sum() > 0:
        st.write("Mean uncertainty:", float(uncertainty[mask > 0].mean()))
    else:
        st.write("Mean uncertainty:", float(uncertainty.mean()))

with col6:
    st.markdown("### Why these locations?")
    st.write("- AI predicts flood extent on terrain.")
    st.write("- Fixed sensors are placed near roads for realistic deployment.")
    st.write("- Road constraints are integrated into sensor optimization.")
    st.write("- LiDAR scan zones complement fixed observations at flood boundaries.")
    st.write("- TwinMind designs an observation network that is both informative and deployable.")

# -------------------------------
# Digital Twin Map
# -------------------------------
st.markdown("---")
st.header("TwinMind Digital Twin Map")

map_col1, map_col2 = st.columns([3, 2])

with map_col1:
    st.pyplot(
        plot_digital_twin_map(
            dem=dem,
            pred=pred,
            road_mask=road_mask,
            sensors=sensor_map,
            lidar_zones=lidar_points,
        ),
        clear_figure=True,
    )

with map_col2:
    st.markdown("### Map Legend")
    st.write("🟫/🟩 DEM terrain")
    st.write("🔵 Flood prediction overlay")
    st.write("⚫ Gray roads")
    st.write("🔴 Fixed sensors (ICOT-LINK)")
    st.write("🔵 Dashed circles = Citizen LiDAR scan zones")

    st.markdown("### TwinMind Interpretation")
    st.write("- AI predicts flood extent on terrain.")
    st.write("- AI measures uncertainty along the flood boundary.")
    st.write("- Fixed sensors are placed near roads for realistic deployment.")
    st.write("- Road constraints are integrated into sensor optimization.")
    st.write("- LiDAR scan zones complement fixed observations.")

    st.markdown("### Key Message")
    st.info(
        "TwinMind does not only predict disasters. "
        "It designs how infrastructure should observe the world."
    )

st.markdown("---")

st.markdown(
    """
### Observation is the new prediction

TwinMind designs where infrastructure should observe the world.

It recommends:

- 🔴 Fixed sensors (IoT)
- 🔵 Citizen LiDAR observation zones

to reduce uncertainty in disaster prediction.
"""
)
st.markdown("---")
st.header("TwinMind Evolution")

evo_col1, evo_col2 = st.columns([3, 2])

with evo_col1:
    st.markdown(
        """
### Closed-loop Observation AI

1. **Flood Prediction**  
   TwinMind predicts flood extent from DEM and rainfall.

2. **Uncertainty Analysis**  
   It identifies where the model is uncertain.

3. **Observation Design**  
   It places fixed sensors and recommends LiDAR scan zones.

4. **New Observation**  
   Real-world data is collected from ICOT-LINK and citizen LiDAR.

5. **Twin Update**  
   The digital twin is updated with the new observation.

6. **Improved Prediction**  
   The next prediction becomes more accurate.
"""
    )

with evo_col2:
    st.markdown("### Evolution Loop")
    st.info(
        "Predict → Measure uncertainty → Observe → Update twin → Predict again"
    )

    st.markdown("### Key Idea")
    st.success(
        "TwinMind is not a one-shot prediction tool. "
        "It is a self-improving Observation OS."
    )
