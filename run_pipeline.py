import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import openslide
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
QUPATH_TEMPLATE_DIR = PROJECT_ROOT / "common_qupath"
QUPATH_PROJECTS_ROOT = PROJECT_ROOT / "qupath_projects"


class AnnotationLoader:
    """Load artery/lumen annotations and classify wall homogeneity."""

    def __init__(
        self,
        json_path: Path,
        overlap_threshold: float = 0.05,
        shape_similarity_thresh: float = 0.6,
    ):
        self.json_path = Path(json_path)
        self.overlap_threshold = overlap_threshold
        self.shape_similarity_thresh = shape_similarity_thresh

    def load(self):
        with self.json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_polygon(self, svg_str: str):
        match = re.search(r'points="([^"]+)"', svg_str)
        if not match:
            return None

        coords = np.array(
            [
                list(map(float, pair.split(",")))
                for pair in match.group(1).strip().split(" ")
                if "," in pair
            ],
            dtype=np.float64,
        )
        return coords if len(coords) >= 3 else None

    def extract_artery_lumen_mapping(self, data: dict):
        artery_polys, lumen_polys = [], []

        for ann in data.get("anno", {}).values():
            try:
                tag = ann["body"][0]["value"].lower()
                svg_str = ann["target"]["selector"]["value"]
                coords = self._parse_polygon(svg_str)
                if coords is None:
                    continue

                if "artery" in tag:
                    artery_polys.append(coords)
                elif "lumen" in tag:
                    lumen_polys.append(coords)
            except Exception as exc:
                print(f"Skipping invalid annotation: {exc}")

        mapping = {}
        for i, artery in enumerate(artery_polys):
            artery_poly = Polygon(artery).buffer(0)
            if artery_poly.is_empty or artery_poly.area <= 0:
                continue

            lumen_matches = []
            extended = artery_poly

            for lumen in lumen_polys:
                lumen_poly = Polygon(lumen).buffer(0)
                if lumen_poly.is_empty or lumen_poly.area <= 0:
                    continue

                inside = artery_poly.contains(lumen_poly.centroid)
                overlap = artery_poly.intersection(lumen_poly).area
                overlap_ratio = overlap / artery_poly.area if artery_poly.area > 0 else 0.0

                if inside or overlap_ratio >= self.overlap_threshold:
                    lumen_matches.append(lumen)
                    extended = extended.union(lumen_poly)

            mapping[i] = {
                "artery": np.array(extended.exterior.coords, dtype=np.float64),
                "lumens": lumen_matches,
            }

        return mapping

    def extract_artery_polygons(self, data: dict):
        artery_polys = []
        for ann in data.get("anno", {}).values():
            try:
                tag = ann["body"][0]["value"].lower()
                if "artery" not in tag:
                    continue
                svg_str = ann["target"]["selector"]["value"]
                coords = self._parse_polygon(svg_str)
                if coords is None:
                    continue
                artery_polys.append(coords)
            except Exception as exc:
                print(f"Skipping invalid annotation: {exc}")
        return artery_polys

    def _extract_intersection_points(self, geom):
        if geom.is_empty:
            return []
        geom_type = geom.geom_type
        if geom_type == "Point":
            return [geom]
        if geom_type == "MultiPoint":
            return list(geom.geoms)
        if geom_type == "LineString":
            return [Point(c) for c in geom.coords]
        if geom_type == "MultiLineString":
            points = []
            for line in geom.geoms:
                points.extend([Point(c) for c in line.coords])
            return points
        if geom_type == "GeometryCollection":
            points = []
            for g in geom.geoms:
                points.extend(self._extract_intersection_points(g))
            return points
        return []

    def _sample_exterior(self, poly: Polygon, n: int = 128):
        ring = poly.exterior
        if ring is None:
            return None
        perimeter = ring.length
        if perimeter <= 0:
            return None
        samples = np.linspace(0.0, perimeter, n, endpoint=False)
        return np.array(
            [(ring.interpolate(s).x, ring.interpolate(s).y) for s in samples],
            dtype=np.float64,
        )

    def _principal_angle(self, x_vals: np.ndarray):
        cov = np.cov(x_vals.T)
        vals, vecs = np.linalg.eigh(cov)
        vec = vecs[:, np.argmax(vals)]
        return np.arctan2(vec[1], vec[0])

    def _angle_diff(self, a: float, b: float):
        delta = abs(a - b)
        delta = np.mod(delta, np.pi)
        return min(delta, np.pi - delta)

    def _shape_similarity(
        self,
        artery_poly: Polygon,
        lumen_poly: Polygon,
        n_samples: int = 128,
        angle_tol_deg: float = 25.0,
    ):
        artery_pts = self._sample_exterior(artery_poly, n_samples)
        lumen_pts = self._sample_exterior(lumen_poly, n_samples)
        if artery_pts is None or lumen_pts is None:
            return 0.0

        artery_c = artery_pts - artery_pts.mean(axis=0)
        lumen_c = lumen_pts - lumen_pts.mean(axis=0)

        scale_a = np.sqrt((artery_c**2).sum(axis=1).mean()) + 1e-12
        scale_l = np.sqrt((lumen_c**2).sum(axis=1).mean()) + 1e-12
        artery_n = artery_c / scale_a
        lumen_n = lumen_c / scale_l

        ang_a = self._principal_angle(artery_n)
        ang_l = self._principal_angle(lumen_n)
        if self._angle_diff(ang_a, ang_l) > np.deg2rad(angle_tol_deg):
            return 0.0

        diff = artery_n[:, None, :] - lumen_n[None, :, :]
        distances = np.sqrt((diff**2).sum(axis=2))
        d_a_to_l = distances.min(axis=1).mean()
        d_l_to_a = distances.min(axis=0).mean()
        chamfer = 0.5 * (d_a_to_l + d_l_to_a)
        return float(1.0 / (1.0 + chamfer))

    def _ray_distance_profile(self, artery_poly: Polygon, lumen_union, n_rays: int = 72):
        centroid = artery_poly.centroid
        outer_distances = []
        inner_distances = []
        wall_thickness = []
        lumen_ratios = []
        ray_segments = []

        for ray_idx, angle in enumerate(np.linspace(0, 2 * np.pi, n_rays, endpoint=False)):
            dx, dy = np.cos(angle), np.sin(angle)
            far = Point(centroid.x + dx * 1e4, centroid.y + dy * 1e4)
            ray = LineString([centroid, far])

            art_inter = artery_poly.intersection(ray)
            art_pts = self._extract_intersection_points(art_inter)
            if not art_pts:
                continue
            out_candidates = [centroid.distance(p) for p in art_pts]
            out_candidates = [d for d in out_candidates if d > 1e-6]
            if not out_candidates:
                continue
            d_outer = max(out_candidates)

            d_inner = 0.0
            lum_inter = lumen_union.intersection(ray)
            lum_pts = self._extract_intersection_points(lum_inter)
            if lum_pts:
                inner_candidates = [
                    centroid.distance(p)
                    for p in lum_pts
                    if 1e-6 < centroid.distance(p) < (d_outer - 1e-6)
                ]
                if inner_candidates:
                    # Use the farthest lumen boundary hit on this ray.
                    d_inner = max(inner_candidates)

            outer_distances.append(float(d_outer))
            inner_distances.append(float(d_inner))
            thickness = float(max(0.0, d_outer - d_inner))
            wall_thickness.append(thickness)
            lumen_ratios.append(float(d_inner / d_outer) if d_outer > 0 else 0.0)
            ray_segments.append(
                {
                    "ray_index": int(ray_idx),
                    "x_inner": float(centroid.x + dx * d_inner),
                    "y_inner": float(centroid.y + dy * d_inner),
                    "x_outer": float(centroid.x + dx * d_outer),
                    "y_outer": float(centroid.y + dy * d_outer),
                    "thickness": thickness,
                }
            )

        return {
            "outer_distances": outer_distances,
            "inner_distances": inner_distances,
            "wall_thickness": wall_thickness,
            "lumen_ratios": lumen_ratios,
            "ray_segments": ray_segments,
        }

    def _compute_multilumen_metrics(self, lumen_polys: list[Polygon]):
        if not lumen_polys:
            return {"lumen_count": 0, "multi_lumen_complexity": 0.0}

        areas = [max(float(p.area), 0.0) for p in lumen_polys]
        total_area = float(sum(areas))
        if total_area <= 0:
            return {"lumen_count": 0, "multi_lumen_complexity": 0.0}

        n_lumen = len(lumen_polys)
        dominance = max(areas) / total_area
        complexity = 0.5 * min(1.0, max(0, n_lumen - 1) / 3.0) + 0.5 * (1.0 - dominance)
        return {
            "lumen_count": int(n_lumen),
            "multi_lumen_complexity": float(np.clip(complexity, 0.0, 1.0)),
        }

    def _compute_concentricity_index(self, artery_poly: Polygon, lumen_union):
        if lumen_union.is_empty or lumen_union.area <= 0 or artery_poly.area <= 0:
            return 0.0
        c_art = artery_poly.centroid
        c_lum = lumen_union.centroid
        d = c_art.distance(c_lum)
        r_art = float(np.sqrt(artery_poly.area / np.pi))
        r_lum = float(np.sqrt(lumen_union.area / np.pi))
        margin = max(r_art - r_lum, 1e-6)
        ci = 1.0 - min(1.0, d / margin)
        return float(np.clip(ci, 0.0, 1.0))

    def check_homogeneity(
        self,
        mapping: dict,
        n_rays: int = 72,
        tolerance: float = 0.1,
        min_ratio: float = 0.6,
        concentricity_thresh: float = 0.55,
        shape_samples: int = 128,
        shape_angle_tol_deg: float = 25.0,
    ):
        updated = mapping.copy()

        for idx, entry in mapping.items():
            artery_poly = Polygon(entry["artery"]).buffer(0)
            lumens = entry.get("lumens", [])

            if artery_poly.is_empty or artery_poly.area <= 0:
                updated[idx]["homogeneity"] = "uncertain"
                continue
            if not lumens:
                updated[idx]["homogeneity"] = "no_lumen"
                continue

            lumen_polys = [Polygon(l).buffer(0) for l in lumens]
            lumen_polys = [p for p in lumen_polys if not p.is_empty and p.area > 0]
            if not lumen_polys:
                updated[idx]["homogeneity"] = "no_lumen"
                continue

            clipped_lumens = [artery_poly.intersection(p).buffer(0) for p in lumen_polys]
            clipped_lumens = [p for p in clipped_lumens if not p.is_empty and p.area > 0]
            if not clipped_lumens:
                updated[idx]["homogeneity"] = "uncertain"
                continue

            lumen_union = unary_union(clipped_lumens).buffer(0)
            largest = max(clipped_lumens, key=lambda p: p.area)
            multi = self._compute_multilumen_metrics(clipped_lumens)
            concentricity_index = self._compute_concentricity_index(artery_poly, lumen_union)

            profile = self._ray_distance_profile(artery_poly, lumen_union, n_rays=n_rays)
            thickness = np.asarray(profile["wall_thickness"], dtype=np.float64)
            ratios = np.asarray(profile["lumen_ratios"], dtype=np.float64)
            inner = np.asarray(profile["inner_distances"], dtype=np.float64)

            if thickness.size < max(16, n_rays // 3):
                updated[idx]["homogeneity"] = "uncertain"
                continue

            t_mean = float(np.mean(thickness))
            t_min = float(np.min(thickness))
            t_max = float(np.max(thickness))
            t_std = float(np.std(thickness))
            t_cv = float(t_std / (t_mean + 1e-6))
            t_iqr = float(np.percentile(thickness, 75) - np.percentile(thickness, 25))
            if inner.size >= 8:
                p10 = float(np.percentile(inner, 10))
                p90 = float(np.percentile(inner, 90))
                narrowing_index = float(1.0 - (p10 / (p90 + 1e-6)))
            else:
                narrowing_index = 0.0

            ratio_center = float(np.median(ratios)) if ratios.size else 0.0
            good = (
                np.sum(np.abs(ratios - ratio_center) <= tolerance * max(ratio_center, 1e-6))
                if ratios.size
                else 0
            )
            ratio_consistency = float(good / max(1, ratios.size))

            similarity = self._shape_similarity(
                artery_poly,
                largest,
                n_samples=shape_samples,
                angle_tol_deg=shape_angle_tol_deg,
            )

            updated[idx]["lumen_count"] = multi["lumen_count"]
            updated[idx]["multi_lumen_complexity"] = multi["multi_lumen_complexity"]
            updated[idx]["concentricity_index"] = concentricity_index
            updated[idx]["wall_thickness_min"] = t_min
            updated[idx]["wall_thickness_mean"] = t_mean
            updated[idx]["wall_thickness_max"] = t_max
            updated[idx]["wall_thickness_cv"] = t_cv
            updated[idx]["wall_thickness_iqr"] = t_iqr
            updated[idx]["narrowing_index"] = narrowing_index
            updated[idx]["ray_valid_fraction"] = float(thickness.size / max(1, n_rays))
            updated[idx]["shape_similarity"] = similarity
            updated[idx]["ratio_consistency"] = ratio_consistency
            updated[idx]["ray_segments"] = profile.get("ray_segments", [])

            is_heterogeneous = (
                concentricity_index < concentricity_thresh
                or t_cv > 0.35
                or narrowing_index > 0.45
                or ratio_consistency < min_ratio
            )

            updated[idx]["shape_similarity"] = similarity
            updated[idx]["homogeneity"] = (
                "heterogeneous"
                if is_heterogeneous or similarity < self.shape_similarity_thresh
                else "homogeneous"
            )

        return updated


class SlideManager:
    def __init__(self, svs_path: Path, level: int = 2):
        self.slide = openslide.OpenSlide(str(svs_path))
        self.level = level
        self.scale = self.slide.level_downsamples[level]

    def read_image(self):
        img = self.slide.read_region(
            (0, 0), self.level, self.slide.level_dimensions[self.level]
        ).convert("RGB")
        return np.array(img)


def resolve_qupath_exe():
    env_qupath = os.environ.get("QUPATH_EXE")
    if env_qupath:
        env_path = Path(env_qupath)
        if env_path.exists():
            return str(env_path)
        raise FileNotFoundError(f"QUPATH_EXE is set but not found: {env_path}")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        root = Path(local_app_data)
        console_candidates = sorted(root.glob("QuPath-*/*(console).exe"), reverse=True)
        exe_candidates = sorted(root.glob("QuPath-*/*.exe"), reverse=True)
        for candidate in console_candidates + exe_candidates:
            if candidate.is_file() and candidate.name.lower().startswith("qupath-"):
                return str(candidate)

    raise RuntimeError(
        "Could not find QuPath executable.\n"
        "Set QUPATH_EXE to your QuPath executable path and re-run."
    )


def build_qupath_paths(svs_path: Path):
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", svs_path.stem)
    project_dir = QUPATH_PROJECTS_ROOT / safe_stem
    return {
        "project_dir": project_dir,
        "output_dir": project_dir / "output",
        "project_file": project_dir / "project.qpproj",
    }


def ensure_qupath_project_template(project_dir: Path):
    if project_dir.exists():
        return

    if QUPATH_TEMPLATE_DIR.exists():
        shutil.copytree(QUPATH_TEMPLATE_DIR, project_dir)
    else:
        project_dir.mkdir(parents=True, exist_ok=True)


def ensure_project_image(source_image_path: Path, project_dir: Path):
    project_image_path = project_dir / source_image_path.name
    project_dir.mkdir(parents=True, exist_ok=True)

    if source_image_path.resolve() == project_image_path.resolve():
        return project_image_path

    needs_copy = (
        not project_image_path.exists()
        or project_image_path.stat().st_size != source_image_path.stat().st_size
    )
    if needs_copy:
        print(f"Syncing image into QuPath project: {project_image_path.name}")
        shutil.copy2(source_image_path, project_image_path)

    return project_image_path


def ensure_qupath_project_files(
    project_image_path: Path,
    project_file: Path,
    project_dir: Path,
    output_dir: Path,
):
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "data" / "1").mkdir(parents=True, exist_ok=True)
    (project_dir / "classifiers").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    now_ms = int(time.time() * 1000)
    project_uri = project_file.resolve().as_uri()
    image_uri = project_image_path.resolve().as_uri()

    if project_file.exists():
        with project_file.open("r", encoding="utf-8") as f:
            project_data = json.load(f)
    else:
        project_data = {}

    images = project_data.get("images", [])
    existing = images[0] if images else {}
    entry_id = existing.get("entryID", 1)
    randomized_name = existing.get("randomizedName", str(uuid.uuid4()))

    image_entry = {
        "serverBuilder": {
            "builderType": "uri",
            "providerClassName": "qupath.lib.images.servers.bioformats.BioFormatsServerBuilder",
            "uri": image_uri,
            "args": ["--series", "0"],
        },
        "entryID": entry_id,
        "randomizedName": randomized_name,
        "imageName": project_image_path.name,
        "metadata": existing.get("metadata", {}),
        "tags": existing.get("tags", []),
    }

    project_data["version"] = "0.6.0"
    project_data["createTimestamp"] = project_data.get("createTimestamp", now_ms)
    project_data["modifyTimestamp"] = now_ms
    project_data["uri"] = project_uri
    project_data["lastID"] = max(project_data.get("lastID", 1), entry_id)
    project_data["images"] = [image_entry]

    with project_file.open("w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2)

    server_json_path = project_dir / "data" / "1" / "server.json"
    with server_json_path.open("w", encoding="utf-8") as f:
        json.dump(image_entry["serverBuilder"], f, indent=2)

    metadata_json_path = project_dir / "data" / "metadata.json"
    if not metadata_json_path.exists():
        with metadata_json_path.open("w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def run_qupath_headless(image_name: str, script_path: Path, project_file: Path):
    qupath_exe = resolve_qupath_exe()
    cmd = [
        qupath_exe,
        "script",
        "--project",
        str(project_file),
        "--image",
        image_name,
        str(script_path),
    ]
    print("Running QuPath headless...")
    subprocess.run(cmd, check=True)
    print("QuPath headless execution completed")


def write_qupath_csvs(artery_lumen_map: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries_csv = output_dir / "glomeruli_boundaries.csv"
    centroids_csv = output_dir / "lumen_centroids.csv"

    boundary_rows = []
    centroid_rows = []

    for artery_idx, entry in sorted(artery_lumen_map.items(), key=lambda x: int(x[0])):
        artery_coords = np.asarray(entry.get("artery", []), dtype=np.float64)
        if artery_coords.shape[0] < 3:
            continue

        category = entry.get("homogeneity", "uncertain")
        for order, (x, y) in enumerate(artery_coords):
            boundary_rows.append([artery_idx, category, order, float(x), float(y)])

        lumen_coords_list = entry.get("lumens", [])
        if lumen_coords_list:
            lumen_polys = []
            for lumen_coords in lumen_coords_list:
                poly = Polygon(lumen_coords).buffer(0)
                if not poly.is_empty and poly.area > 0:
                    lumen_polys.append(poly)
            if lumen_polys:
                lumen_union = unary_union(lumen_polys).buffer(0)
                c = lumen_union.centroid if not lumen_union.is_empty else max(lumen_polys, key=lambda p: p.area).centroid
                centroid_rows.append(
                    [
                        artery_idx,
                        category,
                        round(c.x, 3),
                        round(c.y, 3),
                        round(float(entry.get("shape_similarity", 0.0)), 4),
                    ]
                )

    with boundaries_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "category", "order", "x", "y"])
        writer.writerows(boundary_rows)

    with centroids_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["annotation_id", "category", "centroid_x", "centroid_y", "shape_similarity"]
        )
        writer.writerows(centroid_rows)

    print(f"QuPath CSVs written: {boundaries_csv}, {centroids_csv}")


def write_qupath_advanced_csvs(
    artery_lumen_map: dict,
    artery_flags: dict,
    artery_severity: dict,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "artery_metrics.csv"
    rays_csv = output_dir / "ray_thickness_segments.csv"

    metrics_rows = []
    ray_rows = []

    for artery_idx, entry in sorted(artery_lumen_map.items(), key=lambda x: int(x[0])):
        idx = _safe_int(artery_idx)
        if idx is None:
            continue

        flag = artery_flags.get(idx, {})
        sev = artery_severity.get(idx, {})
        metrics_rows.append(
            [
                idx,
                entry.get("homogeneity"),
                entry.get("shape_similarity"),
                entry.get("concentricity_index"),
                entry.get("lumen_count", 0),
                entry.get("multi_lumen_complexity"),
                entry.get("wall_thickness_mean"),
                entry.get("wall_thickness_cv"),
                entry.get("narrowing_index"),
                entry.get("ray_valid_fraction"),
                flag.get("flag_level"),
                flag.get("risk_score"),
                flag.get("aif_percent"),
                sev.get("severity_grade"),
            ]
        )

        thickness_mean = float(entry.get("wall_thickness_mean") or 0.0)
        for seg in entry.get("ray_segments", []):
            thickness = float(seg.get("thickness", 0.0))
            ray_rows.append(
                [
                    idx,
                    int(seg.get("ray_index", 0)),
                    float(seg.get("x_inner", 0.0)),
                    float(seg.get("y_inner", 0.0)),
                    float(seg.get("x_outer", 0.0)),
                    float(seg.get("y_outer", 0.0)),
                    thickness,
                    thickness / (thickness_mean + 1e-6) if thickness_mean > 0 else 0.0,
                ]
            )

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "annotation_id",
                "homogeneity",
                "shape_similarity",
                "concentricity_index",
                "lumen_count",
                "multi_lumen_complexity",
                "wall_thickness_mean",
                "wall_thickness_cv",
                "narrowing_index",
                "ray_valid_fraction",
                "flag_level",
                "risk_score",
                "aif_percent",
                "severity_grade",
            ]
        )
        writer.writerows(metrics_rows)

    with rays_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "annotation_id",
                "ray_index",
                "x_inner",
                "y_inner",
                "x_outer",
                "y_outer",
                "thickness",
                "thickness_norm",
            ]
        )
        writer.writerows(ray_rows)

    print(f"Advanced QuPath CSVs written: {metrics_csv}, {rays_csv}")


def write_aif_qupath_csvs(aif_records: list, output_dir: Path, aif_summary: dict | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    artery_csv = output_dir / "aif_artery_boundaries.csv"
    lumen_csv = output_dir / "aif_lumen_boundaries.csv"
    centroids_csv = output_dir / "aif_centroids.csv"
    circles_csv = output_dir / "aif_circles.csv"

    artery_rows = []
    lumen_rows = []
    centroid_rows = []
    circle_rows = []

    first_by_artery = {}
    for rec in aif_records:
        idx = rec["artery_idx"]
        if idx not in first_by_artery:
            first_by_artery[idx] = rec

    for idx, rec in sorted(first_by_artery.items(), key=lambda x: int(x[0])):
        summary = (aif_summary or {}).get(int(idx), {})
        category = summary.get("aif_class_multi", rec.get("aif_class", "insufficient"))
        artery_coords = np.asarray(rec["artery_coords"], dtype=np.float64)
        for order, (x, y) in enumerate(artery_coords):
            artery_rows.append([idx, category, order, float(x), float(y)])

        c_art = summary.get("centroid_artery", rec.get("centroid_artery", (0.0, 0.0)))
        c_lum = summary.get("centroid_lumen_union", rec.get("centroid_lumen", (0.0, 0.0)))
        aif_value = summary.get("aif_percent_multi", rec.get("aif_percent"))
        r_outer = summary.get("r_outer_equiv", rec.get("r_outer", 0.0))
        r_union = summary.get("r_lumen_union_equiv", rec.get("r_adj", 0.0))

        centroid_rows.append([idx, "artery", category, c_art[0], c_art[1], aif_value])
        centroid_rows.append([idx, "lumen", category, c_lum[0], c_lum[1], aif_value])
        circle_rows.append([idx, "outer", category, c_art[0], c_art[1], r_outer])
        circle_rows.append([idx, "union", category, c_lum[0], c_lum[1], r_union])

    for rec in aif_records:
        idx = rec["artery_idx"]
        summary = (aif_summary or {}).get(int(idx), {})
        category = summary.get("aif_class_multi", rec.get("aif_class", "insufficient"))
        lumen_coords = np.asarray(rec["lumen_coords"], dtype=np.float64)
        for order, (x, y) in enumerate(lumen_coords):
            lumen_rows.append([idx, category, order, float(x), float(y)])

    with artery_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "category", "order", "x", "y"])
        writer.writerows(artery_rows)

    with lumen_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "category", "order", "x", "y"])
        writer.writerows(lumen_rows)

    with centroids_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "type", "category", "centroid_x", "centroid_y", "aif_percent"])
        writer.writerows(centroid_rows)

    with circles_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "circle_type", "category", "center_x", "center_y", "radius"])
        writer.writerows(circle_rows)

    print(f"AIF QuPath CSVs written: {artery_csv}, {lumen_csv}, {centroids_csv}, {circles_csv}")


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_by_artery_index(payload: dict, idx: int):
    return payload.get(idx + 1) or payload.get(idx) or {}


def compute_artery_flags(artery_lumen_map: dict, aif_records: list, aif_summary: dict | None = None):
    """Assign review-priority flags per artery for medical triage."""
    largest_aif_by_artery = {}
    for rec in aif_records or []:
        idx = _safe_int(rec.get("artery_idx"))
        if idx is None or idx in largest_aif_by_artery:
            continue
        largest_aif_by_artery[idx] = rec

    flags = {}
    for raw_idx, entry in artery_lumen_map.items():
        idx = _safe_int(raw_idx)
        if idx is None:
            continue

        homogeneity = entry.get("homogeneity")
        shape_similarity = entry.get("shape_similarity")
        concentricity_index = entry.get("concentricity_index")
        wall_thickness_cv = entry.get("wall_thickness_cv")
        narrowing_index = entry.get("narrowing_index")
        multi_lumen_complexity = entry.get("multi_lumen_complexity")
        lumen_count = int(entry.get("lumen_count", 0))

        aif_info = _get_by_artery_index(aif_summary or {}, idx)
        aif_rec = largest_aif_by_artery.get(idx + 1) or largest_aif_by_artery.get(idx)
        aif_percent = (
            float(aif_info.get("aif_percent_multi"))
            if aif_info.get("aif_percent_multi") is not None
            else (float(aif_rec["aif_percent"]) if aif_rec and aif_rec.get("aif_percent") is not None else None)
        )
        aif_abs = abs(aif_percent) if aif_percent is not None else None
        contour_quality_mean = float(aif_info.get("contour_quality_mean", 0.0))

        reasons = []
        risk_score = 0

        if homogeneity == "heterogeneous":
            reasons.append("heterogeneous_wall_pattern")
            risk_score += 3
        elif homogeneity in {"uncertain", "no_lumen"}:
            reasons.append("insufficient_lumen_confidence")
            risk_score += 2

        if shape_similarity is not None:
            ss = float(shape_similarity)
            if ss < 0.50:
                reasons.append("low_shape_similarity")
                risk_score += 2
            elif ss < 0.60:
                reasons.append("borderline_shape_similarity")
                risk_score += 1

        if concentricity_index is not None:
            ci = float(concentricity_index)
            if ci < 0.40:
                reasons.append("low_concentricity_index")
                risk_score += 2
            elif ci < 0.55:
                reasons.append("borderline_concentricity_index")
                risk_score += 1

        if wall_thickness_cv is not None and float(wall_thickness_cv) > 0.35:
            reasons.append("high_wall_thickness_variation")
            risk_score += 1

        if narrowing_index is not None and float(narrowing_index) > 0.45:
            reasons.append("focal_lumen_narrowing_pattern")
            risk_score += 1

        if lumen_count > 1:
            reasons.append("multi_lumen_detected")
            risk_score += 1
        if multi_lumen_complexity is not None and float(multi_lumen_complexity) > 0.45:
            reasons.append("high_multi_lumen_complexity")
            risk_score += 1

        if aif_abs is not None:
            if aif_abs > 85.0:
                reasons.append("high_aif_deviation")
                risk_score += 2
            elif aif_abs > 70.0:
                reasons.append("mild_aif_deviation")
                risk_score += 1

        if contour_quality_mean < 0.25:
            reasons.append("low_contour_quality")
            risk_score += 1

        if risk_score >= 4:
            flag_level = "urgent_review"
        elif risk_score >= 2:
            flag_level = "review_recommended"
        else:
            flag_level = "routine"

        if not reasons:
            reasons.append("no_major_morphology_alert")

        flags[idx] = {
            "flag_level": flag_level,
            "risk_score": risk_score,
            "reasons": reasons,
            "aif_percent": aif_percent,
            "homogeneity": homogeneity,
            "shape_similarity": shape_similarity,
            "concentricity_index": concentricity_index,
            "wall_thickness_cv": wall_thickness_cv,
            "narrowing_index": narrowing_index,
            "lumen_count": lumen_count,
            "multi_lumen_complexity": multi_lumen_complexity,
            "contour_quality_mean": contour_quality_mean,
        }

    return flags


def write_artery_flags_csv(artery_flags: dict, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "artery_id",
                "flag_level",
                "risk_score",
                "homogeneity",
                "shape_similarity",
                "aif_percent",
                "concentricity_index",
                "wall_thickness_cv",
                "narrowing_index",
                "lumen_count",
                "multi_lumen_complexity",
                "contour_quality_mean",
                "reasons",
            ]
        )
        for artery_id, flag in sorted(artery_flags.items(), key=lambda x: int(x[0])):
            shape_similarity = flag.get("shape_similarity")
            aif_percent = flag.get("aif_percent")
            ci = flag.get("concentricity_index")
            wt_cv = flag.get("wall_thickness_cv")
            ni = flag.get("narrowing_index")
            mlc = flag.get("multi_lumen_complexity")
            cqs = flag.get("contour_quality_mean")
            writer.writerow(
                [
                    artery_id,
                    flag.get("flag_level"),
                    flag.get("risk_score"),
                    flag.get("homogeneity"),
                    "" if shape_similarity is None else round(float(shape_similarity), 4),
                    "" if aif_percent is None else round(float(aif_percent), 3),
                    "" if ci is None else round(float(ci), 4),
                    "" if wt_cv is None else round(float(wt_cv), 4),
                    "" if ni is None else round(float(ni), 4),
                    flag.get("lumen_count", 0),
                    "" if mlc is None else round(float(mlc), 4),
                    "" if cqs is None else round(float(cqs), 4),
                    "|".join(flag.get("reasons", [])),
                ]
            )

    print(f"Artery flags CSV written: {output_csv}")


def compute_artery_severity(artery_flags: dict):
    severity = {}
    for artery_id, flag in artery_flags.items():
        risk = int(flag.get("risk_score", 0))

        if risk >= 5:
            grade = 3
            label = "severe"
        elif risk >= 3:
            grade = 2
            label = "moderate"
        elif risk >= 1:
            grade = 1
            label = "mild"
        else:
            grade = 0
            label = "none"

        severity[artery_id] = {
            "severity_grade": grade,
            "severity_label": label,
            "severity_reasons": list(flag.get("reasons", [])),
        }

    return severity


def write_artery_severity_csv(artery_severity: dict, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artery_id", "severity_grade", "severity_label", "severity_reasons"])
        for artery_id, sev in sorted(artery_severity.items(), key=lambda x: int(x[0])):
            writer.writerow(
                [
                    artery_id,
                    sev.get("severity_grade"),
                    sev.get("severity_label"),
                    "|".join(sev.get("severity_reasons", [])),
                ]
            )

    print(f"Artery severity CSV written: {output_csv}")


def build_case_summary(stem: str, updated_map: dict, artery_flags: dict, artery_severity: dict):
    total = len(updated_map)
    homogeneity_counts = {"homogeneous": 0, "heterogeneous": 0, "no_lumen": 0, "uncertain": 0}
    for entry in updated_map.values():
        h = entry.get("homogeneity", "uncertain")
        if h not in homogeneity_counts:
            homogeneity_counts[h] = 0
        homogeneity_counts[h] += 1

    flag_counts = {"urgent_review": 0, "review_recommended": 0, "routine": 0}
    for f in artery_flags.values():
        lvl = f.get("flag_level", "routine")
        if lvl not in flag_counts:
            flag_counts[lvl] = 0
        flag_counts[lvl] += 1

    severity_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for sev in artery_severity.values():
        g = int(sev.get("severity_grade", 0))
        severity_counts[g] = severity_counts.get(g, 0) + 1

    urgent = flag_counts.get("urgent_review", 0)
    severe = severity_counts.get(3, 0)
    moderate_or_higher = severity_counts.get(2, 0) + severity_counts.get(3, 0)

    if total == 0:
        impression = "No analyzable arteries detected in this case."
    elif severe >= 1 or urgent >= 1:
        impression = "High vascular abnormality burden; prioritize expert review."
    elif moderate_or_higher / max(total, 1) >= 0.30:
        impression = "Moderate vascular abnormality burden; review recommended."
    else:
        impression = "Low vascular abnormality burden; routine review suggested."

    def _mean_from(values):
        vals = [float(v) for v in values if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _max_from(values):
        vals = [float(v) for v in values if v is not None]
        return float(np.max(vals)) if vals else 0.0

    def _min_from(values):
        vals = [float(v) for v in values if v is not None]
        return float(np.min(vals)) if vals else 0.0

    aif_vals = [f.get("aif_percent") for f in artery_flags.values()]
    ci_vals = [e.get("concentricity_index") for e in updated_map.values()]
    wt_cv_vals = [e.get("wall_thickness_cv") for e in updated_map.values()]
    ni_vals = [e.get("narrowing_index") for e in updated_map.values()]
    mlc_vals = [e.get("multi_lumen_complexity") for e in updated_map.values()]
    multi_lumen_count = sum(1 for e in updated_map.values() if int(e.get("lumen_count", 0)) > 1)
    mean_lumen_count = _mean_from([e.get("lumen_count", 0) for e in updated_map.values()])

    return {
        "case_id": stem,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_arteries": total,
        "homogeneity_counts": homogeneity_counts,
        "flag_counts": flag_counts,
        "severity_counts": {
            "grade_0_none": severity_counts.get(0, 0),
            "grade_1_mild": severity_counts.get(1, 0),
            "grade_2_moderate": severity_counts.get(2, 0),
            "grade_3_severe": severity_counts.get(3, 0),
        },
        "advanced_metrics": {
            "mean_aif_multi_percent": round(_mean_from(aif_vals), 3),
            "mean_concentricity_index": round(_mean_from(ci_vals), 4),
            "mean_wall_thickness_cv": round(_mean_from(wt_cv_vals), 4),
            "mean_narrowing_index": round(_mean_from(ni_vals), 4),
            "mean_multi_lumen_complexity": round(_mean_from(mlc_vals), 4),
            "multi_lumen_arteries": int(multi_lumen_count),
            "mean_lumen_count": round(mean_lumen_count, 3),
            "max_aif_multi_percent": round(_max_from(aif_vals), 3),
            "min_concentricity_index": round(_min_from(ci_vals), 4),
            "max_wall_thickness_cv": round(_max_from(wt_cv_vals), 4),
            "max_narrowing_index": round(_max_from(ni_vals), 4),
            "arteries_ci_below_0_55": int(sum(1 for x in ci_vals if x is not None and float(x) < 0.55)),
            "arteries_wtcv_above_0_35": int(sum(1 for x in wt_cv_vals if x is not None and float(x) > 0.35)),
            "arteries_ni_above_0_45": int(sum(1 for x in ni_vals if x is not None and float(x) > 0.45)),
            "arteries_aif_above_70": int(sum(1 for x in aif_vals if x is not None and float(x) >= 70.0)),
        },
        "impression": impression,
        "note": "Research support output; not a standalone clinical diagnosis.",
    }


def _pdf_escape(text: str):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_text(text: str, max_chars: int = 90):
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def write_styled_pdf(stream: bytes, output_pdf: Path):
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> /Contents 7 0 R >> endobj\n"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj\n")
    objects.append(b"6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >> endobj\n")
    objects.append(f"7 0 obj << /Length {len(stream)} >> stream\n".encode("ascii") + stream + b"\nendstream endobj\n")

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")

    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            "trailer << /Size {size} /Root 1 0 R >>\n"
            "startxref\n{xref}\n%%EOF\n"
        ).format(size=len(offsets), xref=xref_start).encode("ascii")
    )

    with output_pdf.open("wb") as f:
        f.write(pdf)

    print(f"Case summary PDF written: {output_pdf}")


def write_case_summary_pdf(
    case_summary: dict,
    artery_flags: dict,
    artery_severity: dict,
    output_pdf: Path,
    output_image: Path,
    output_aif_image: Path,
):
    ops = []

    def add_text(x: int, y: int, text: str, font: str = "F1", size: int = 10):
        safe = _pdf_escape(str(text)[:180])
        ops.append(f"BT /{font} {size} Tf {x} {y} Td ({safe}) Tj ET")

    def add_rule(y: int):
        ops.append(f"0.70 w 40 {y} m 555 {y} l S")

    def priority_short(level: str):
        if level == "urgent_review":
            return "URG"
        if level == "review_recommended":
            return "REV"
        return "RTN"

    # Header block
    add_text(40, 808, "Renal Glomeruli Diagnostic Summary", font="F2", size=18)
    add_text(40, 788, f"Case ID: {case_summary.get('case_id', '')}", font="F2", size=11)
    add_text(280, 788, f"Generated (UTC): {case_summary.get('generated_at_utc', '')}", size=10)
    add_rule(778)

    # Impression panel
    ops.append("0.95 0.97 1.00 rg 40 728 515 42 re f")
    ops.append("0 0 0 rg")
    add_text(48, 754, "Impression", font="F2", size=11)
    impression_lines = _wrap_text(case_summary.get("impression", ""), 92)
    y = 738
    for line in impression_lines[:2]:
        add_text(48, y, line, size=10)
        y -= 12

    # Metrics sections
    add_text(40, 700, "Case Metrics", font="F2", size=12)
    add_rule(694)
    add_text(48, 676, f"Total arteries: {case_summary.get('total_arteries', 0)}", font="F2", size=11)

    h = case_summary.get("homogeneity_counts", {})
    f = case_summary.get("flag_counts", {})
    s = case_summary.get("severity_counts", {})
    adv = case_summary.get("advanced_metrics", {})

    add_text(48, 654, "Homogeneity", font="F2", size=10)
    add_text(56, 640, f"Homogeneous: {h.get('homogeneous', 0)}")
    add_text(178, 640, f"Heterogeneous: {h.get('heterogeneous', 0)}")
    add_text(328, 640, f"No lumen: {h.get('no_lumen', 0)}")
    add_text(460, 640, f"Uncertain: {h.get('uncertain', 0)}")

    add_text(48, 622, "Review Priority", font="F2", size=10)
    add_text(56, 608, f"Urgent: {f.get('urgent_review', 0)}")
    add_text(178, 608, f"Review: {f.get('review_recommended', 0)}")
    add_text(328, 608, f"Routine: {f.get('routine', 0)}")

    add_text(48, 590, "Severity", font="F2", size=10)
    add_text(56, 576, f"G0: {s.get('grade_0_none', 0)}")
    add_text(122, 576, f"G1: {s.get('grade_1_mild', 0)}")
    add_text(188, 576, f"G2: {s.get('grade_2_moderate', 0)}")
    add_text(254, 576, f"G3: {s.get('grade_3_severe', 0)}")

    add_text(48, 556, "Advanced Summary", font="F2", size=10)
    add_text(56, 542, f"Mean AIFm: {adv.get('mean_aif_multi_percent', 0.0):.2f}%")
    add_text(192, 542, f"Max AIFm: {adv.get('max_aif_multi_percent', 0.0):.2f}%")
    add_text(328, 542, f"Mean CI: {adv.get('mean_concentricity_index', 0.0):.3f}")
    add_text(450, 542, f"Min CI: {adv.get('min_concentricity_index', 0.0):.3f}")
    add_text(56, 528, f"Mean WTcv: {adv.get('mean_wall_thickness_cv', 0.0):.3f}")
    add_text(192, 528, f"Max WTcv: {adv.get('max_wall_thickness_cv', 0.0):.3f}")
    add_text(328, 528, f"Mean NI: {adv.get('mean_narrowing_index', 0.0):.3f}")
    add_text(450, 528, f"Max NI: {adv.get('max_narrowing_index', 0.0):.3f}")
    add_text(56, 514, f"Multi-lumen arteries: {adv.get('multi_lumen_arteries', 0)}")
    add_text(222, 514, f"AIF>=70 count: {adv.get('arteries_aif_above_70', 0)}")
    add_text(366, 514, f"CI<0.55 count: {adv.get('arteries_ci_below_0_55', 0)}")
    add_text(486, 514, f"WTcv>0.35: {adv.get('arteries_wtcv_above_0_35', 0)}")

    # Top flagged arteries
    add_text(40, 488, "Top Flagged Arteries", font="F2", size=12)
    add_rule(482)
    add_text(48, 468, "A", font="F2", size=9)
    add_text(68, 468, "G", font="F2", size=9)
    add_text(88, 468, "P", font="F2", size=9)
    add_text(118, 468, "R", font="F2", size=9)
    add_text(144, 468, "CI", font="F2", size=9)
    add_text(178, 468, "ML", font="F2", size=9)
    add_text(210, 468, "WTcv", font="F2", size=9)
    add_text(254, 468, "NI", font="F2", size=9)
    add_text(292, 468, "AIFm", font="F2", size=9)
    add_text(338, 468, "Reasons", font="F2", size=9)

    ranked = []
    for artery_id, sev in artery_severity.items():
        flag = artery_flags.get(artery_id, {})
        ranked.append(
            (
                int(sev.get("severity_grade", 0)),
                int(flag.get("risk_score", 0)),
                int(artery_id),
                sev,
                flag,
            )
        )
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))

    row_y = 454
    for sev_grade, risk_score, artery_id, sev, flag in ranked[:7]:
        reasons = ", ".join((flag.get("reasons") or ["none"])[:2])
        if len(reasons) > 34:
            reasons = reasons[:34] + "..."
        ci_value = flag.get("concentricity_index")
        ci_text = "0.00" if ci_value is None else f"{float(ci_value):.2f}"
        add_text(48, row_y, str(artery_id), size=8)
        add_text(68, row_y, str(sev_grade), size=8)
        add_text(88, row_y, priority_short(flag.get("flag_level", "routine")), size=8)
        add_text(118, row_y, str(risk_score), size=8)
        add_text(144, row_y, ci_text, size=8)
        add_text(178, row_y, str(int(flag.get("lumen_count", 0))), size=8)
        add_text(210, row_y, f"{float(flag.get('wall_thickness_cv', 0.0)):.2f}", size=8)
        add_text(254, row_y, f"{float(flag.get('narrowing_index', 0.0)):.2f}", size=8)
        aif_val = flag.get("aif_percent")
        add_text(292, row_y, "" if aif_val is None else f"{float(aif_val):.1f}", size=8)
        add_text(338, row_y, reasons, size=8)
        row_y -= 12
        if row_y < 372:
            break

    # Threshold legend
    add_text(40, 352, "Threshold Guide", font="F2", size=11)
    add_rule(346)
    add_text(48, 332, "CI < 0.55 -> off-center risk", size=9)
    add_text(230, 332, "WTcv > 0.35 -> wall variability risk", size=9)
    add_text(48, 318, "NI > 0.45 -> focal narrowing risk", size=9)
    add_text(230, 318, "AIFm >= 70 moderate, >= 85 high narrowing", size=9)
    add_text(48, 304, "P tags: URG=urgent, REV=review, RTN=routine", size=9)
    add_text(230, 304, "Ray lines show inner-to-outer wall thickness", size=9)

    # Artifact references and note
    add_text(40, 284, "Generated Artifacts", font="F2", size=11)
    add_rule(278)
    add_text(48, 264, f"Artery overlay: {output_image.name}", size=8)
    add_text(48, 252, f"AIF overlay: {output_aif_image.name}", size=8)
    add_text(48, 240, f"Summary JSON: {case_summary.get('case_id')}_case_summary.json", size=8)
    add_text(48, 228, f"Severity CSV: {case_summary.get('case_id')}_severity.csv", size=8)
    add_text(300, 264, "QuPath: artery_metrics.csv", size=8)
    add_text(300, 252, "QuPath: ray_thickness_segments.csv", size=8)
    add_text(300, 240, "QuPath: aif_artery_boundaries.csv", size=8)
    add_text(300, 228, "QuPath: aif_lumen_boundaries.csv", size=8)

    note = case_summary.get("note", "")
    add_text(40, 204, "Interpretation Note", font="F2", size=10)
    for i, line in enumerate(_wrap_text(note, 95)[:2]):
        add_text(48, 190 - (i * 11), line, font="F3", size=8)

    stream = "\n".join(ops).encode("latin-1", errors="replace")
    write_styled_pdf(stream, output_pdf)


def draw_artery_on_slide(
    slide_img: np.ndarray,
    artery_lumen_map: dict,
    output_path: Path,
    scale: float,
    artery_flags: dict | None = None,
    artery_severity: dict | None = None,
):
    image = slide_img.copy()

    for artery_idx, entry in artery_lumen_map.items():
        artery_coords_scaled = (entry["artery"] / scale).astype(np.int32)
        cv2.polylines(image, [artery_coords_scaled], True, (0, 255, 0), 2)

        for lumen_coords in entry.get("lumens", []):
            lumen_coords_scaled = (lumen_coords / scale).astype(np.int32)
            cv2.polylines(image, [lumen_coords_scaled], True, (255, 0, 0), 2)

        status = entry.get("homogeneity")
        flag = (artery_flags or {}).get(_safe_int(artery_idx), {})
        flag_level = flag.get("flag_level")
        severity = (artery_severity or {}).get(_safe_int(artery_idx), {})
        sev_grade = severity.get("severity_grade")
        if status:
            moments = cv2.moments(artery_coords_scaled)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                if flag_level == "urgent_review":
                    color = (0, 0, 255)
                elif flag_level == "review_recommended":
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0) if status == "homogeneous" else (255, 255, 0)

                status_tag = "HOM" if status == "homogeneous" else ("HET" if status == "heterogeneous" else status)
                ci = flag.get("concentricity_index")
                ci_txt = f"{float(ci):.2f}" if ci is not None else "NA"
                ml_n = int(flag.get("lumen_count", 0))
                label = f"A{artery_idx} {status_tag} CI:{ci_txt} ML:{ml_n}"
                if sev_grade is not None:
                    label += f" G{sev_grade}"
                if flag_level:
                    label += f" {flag_level}"
                cv2.putText(
                    image,
                    label,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def compute_aif_percent(artery_area: float, lumen_union_area: float):
    if artery_area <= 0:
        return 0.0, "insufficient"
    aif_percent = 100.0 * max(0.0, (artery_area - max(0.0, lumen_union_area))) / artery_area
    if aif_percent >= 85.0:
        cls = "high_narrowing"
    elif aif_percent >= 70.0:
        cls = "moderate_narrowing"
    else:
        cls = "low_narrowing"
    return float(aif_percent), cls


def _compute_concentricity_index_geom(artery_poly: Polygon, lumen_union):
    if artery_poly.is_empty or artery_poly.area <= 0 or lumen_union.is_empty or lumen_union.area <= 0:
        return 0.0
    c_art = artery_poly.centroid
    c_lum = lumen_union.centroid
    d = c_art.distance(c_lum)
    r_art = float(np.sqrt(artery_poly.area / np.pi))
    r_lum = float(np.sqrt(lumen_union.area / np.pi))
    margin = max(r_art - r_lum, 1e-6)
    return float(np.clip(1.0 - min(1.0, d / margin), 0.0, 1.0))


def _compute_multi_lumen_complexity_from_areas(areas: list[float]):
    if not areas:
        return 0.0
    total = sum(max(0.0, float(a)) for a in areas)
    if total <= 0:
        return 0.0
    dominance = max(areas) / total
    n_lumen = len(areas)
    complexity = 0.5 * min(1.0, max(0, n_lumen - 1) / 3.0) + 0.5 * (1.0 - dominance)
    return float(np.clip(complexity, 0.0, 1.0))


def _contour_quality_score(contour: np.ndarray):
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return 0.0
    circularity = float(np.clip((4.0 * np.pi * area) / (perimeter**2 + 1e-6), 0.0, 1.0))
    area_term = float(np.clip(area / 400.0, 0.0, 1.0))
    return float(0.6 * circularity + 0.4 * area_term)


def detect_lumen_contours(gray: np.ndarray, artery_coords_scaled: np.ndarray):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [artery_coords_scaled], 255)
    x, y, w, h = cv2.boundingRect(artery_coords_scaled)
    if w < 6 or h < 6:
        return []

    roi_gray = gray[y : y + h, x : x + w]
    roi_mask = mask[y : y + h, x : x + w]
    if roi_gray.size == 0:
        return []

    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        -2,
    )
    lumen_region = cv2.bitwise_or(otsu, adaptive)
    lumen_region = cv2.bitwise_and(lumen_region, roi_mask)
    kernel = np.ones((3, 3), np.uint8)
    lumen_region = cv2.morphologyEx(lumen_region, cv2.MORPH_OPEN, kernel, iterations=1)
    lumen_region = cv2.morphologyEx(lumen_region, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(lumen_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shifted = []
    for c in contours:
        c = c.copy()
        c[:, :, 0] += x
        c[:, :, 1] += y
        shifted.append(c)
    return shifted


def summarize_aif_records(aif_records: list):
    grouped = {}
    for rec in aif_records:
        idx = _safe_int(rec.get("artery_idx"))
        if idx is None:
            continue
        grouped.setdefault(idx, []).append(rec)

    summaries = {}
    for idx, recs in grouped.items():
        artery_poly = Polygon(recs[0]["artery_coords"]).buffer(0)
        if artery_poly.is_empty or artery_poly.area <= 0:
            continue

        lumen_polys = []
        areas = []
        quality_scores = []
        for rec in recs:
            lp = Polygon(rec["lumen_coords"]).buffer(0)
            if lp.is_empty or lp.area <= 0:
                continue
            clipped = artery_poly.intersection(lp).buffer(0)
            if clipped.is_empty or clipped.area <= 0:
                continue
            lumen_polys.append(clipped)
            areas.append(float(clipped.area))
            quality_scores.append(float(rec.get("contour_quality", 0.0)))

        if not lumen_polys:
            summaries[idx] = {
                "artery_idx": idx,
                "aif_percent_multi": None,
                "aif_class_multi": "insufficient",
                "lumen_count": 0,
                "multi_lumen_complexity": 0.0,
                "concentricity_index": 0.0,
                "contour_quality_mean": 0.0,
            }
            continue

        lumen_union = unary_union(lumen_polys).buffer(0)
        aif_percent, aif_class = compute_aif_percent(float(artery_poly.area), float(lumen_union.area))
        ci = _compute_concentricity_index_geom(artery_poly, lumen_union)
        ml_complexity = _compute_multi_lumen_complexity_from_areas(areas)
        c_art = artery_poly.centroid
        c_lum = lumen_union.centroid
        r_outer = float(np.sqrt(artery_poly.area / np.pi))
        r_union = float(np.sqrt(max(lumen_union.area, 0.0) / np.pi))

        summaries[idx] = {
            "artery_idx": idx,
            "aif_percent_multi": aif_percent,
            "aif_class_multi": aif_class,
            "lumen_count": int(len(lumen_polys)),
            "multi_lumen_complexity": ml_complexity,
            "concentricity_index": ci,
            "centroid_artery": (float(c_art.x), float(c_art.y)),
            "centroid_lumen_union": (float(c_lum.x), float(c_lum.y)),
            "r_outer_equiv": r_outer,
            "r_lumen_union_equiv": r_union,
            "contour_quality_mean": float(np.mean(quality_scores)) if quality_scores else 0.0,
        }

    return summaries


def draw_aif_overlay(
    slide_img: np.ndarray,
    artery_polygons: list,
    gray: np.ndarray,
    output_path: Path,
    scale: float,
    min_lumen_area: float = 30.0,
):
    overlay_img = slide_img.copy()
    aif_records = []

    for artery_idx, coords in enumerate(artery_polygons, start=1):
        coords_scaled = (coords / scale).astype(np.int32)

        cv2.polylines(overlay_img, [coords_scaled], True, (0, 0, 255), 2)

        contours = detect_lumen_contours(gray, coords_scaled)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_lumen_area]

        if not valid_contours:
            continue

        artery_poly_scaled = Polygon(coords_scaled).buffer(0)
        if artery_poly_scaled.is_empty or artery_poly_scaled.area <= 0:
            continue

        artery_lumen_polys = []
        for contour in valid_contours:
            contour = contour.squeeze()
            if contour.ndim != 2:
                continue

            lumen_poly = Polygon(contour)
            clipped = artery_poly_scaled.intersection(lumen_poly).buffer(0)
            if clipped.is_empty or clipped.area <= 0:
                continue
            artery_lumen_polys.append(clipped)

            cv2.polylines(overlay_img, [contour.astype(np.int32)], True, (255, 0, 0), 2)
            c_m = cv2.moments(contour.astype(np.int32))
            if c_m["m00"] != 0:
                c_x = int(c_m["m10"] / c_m["m00"])
                c_y = int(c_m["m01"] / c_m["m00"])
                cv2.circle(overlay_img, (c_x, c_y), 3, (255, 0, 0), -1)

            aif_records.append(
                {
                    "artery_idx": artery_idx,
                    "artery_coords": coords,
                    "lumen_coords": contour.astype(np.float64) * scale,
                    "contour_quality": _contour_quality_score(contour.astype(np.int32)),
                }
            )

        if not artery_lumen_polys:
            continue
        lumen_union = unary_union(artery_lumen_polys).buffer(0)
        aif_multi, aif_class = compute_aif_percent(float(artery_poly_scaled.area), float(lumen_union.area))
        c_art = artery_poly_scaled.centroid
        c_lum = lumen_union.centroid
        r_outer = float(np.sqrt(artery_poly_scaled.area / np.pi))
        r_union = float(np.sqrt(max(lumen_union.area, 0.0) / np.pi))
        ci = _compute_concentricity_index_geom(artery_poly_scaled, lumen_union)
        ml = _compute_multi_lumen_complexity_from_areas([float(p.area) for p in artery_lumen_polys])

        cv2.circle(overlay_img, (int(c_art.x), int(c_art.y)), int(max(1.0, r_outer)), (0, 255, 0), 2)
        cv2.circle(overlay_img, (int(c_lum.x), int(c_lum.y)), int(max(1.0, r_union)), (0, 255, 255), 2)
        cv2.circle(overlay_img, (int(c_art.x), int(c_art.y)), 4, (0, 0, 255), -1)
        cv2.circle(overlay_img, (int(c_lum.x), int(c_lum.y)), 4, (255, 0, 0), -1)
        cv2.putText(
            overlay_img,
            f"A{artery_idx} AIFm:{aif_multi:.1f}% CI:{ci:.2f} ML:{ml:.2f}",
            (int(c_art.x), int(c_art.y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        for rec in aif_records:
            if rec["artery_idx"] != artery_idx:
                continue
            rec["aif_percent"] = float(aif_multi)
            rec["aif_class"] = aif_class
            rec["centroid_artery"] = (float(c_art.x * scale), float(c_art.y * scale))
            rec["centroid_lumen"] = (float(c_lum.x * scale), float(c_lum.y * scale))
            rec["r_outer"] = float(r_outer * scale)
            rec["r_adj"] = float(r_union * scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    aif_summary = summarize_aif_records(aif_records)
    return aif_records, aif_summary


def pick_default_input_paths():
    preferred_stems = ["grundium16", "grundium17", "grundium1"]
    for stem in preferred_stems:
        svs = DATASET_DIR / f"{stem}.svs"
        jsn = DATASET_DIR / f"{stem}.svs.json"
        if svs.exists() and jsn.exists():
            return svs, jsn
    raise FileNotFoundError(
        f"No matching .svs/.svs.json pair found in {DATASET_DIR}. "
        "Provide --svs and --json explicitly."
    )


def resolve_input_path(path_str: str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative
    return DATASET_DIR / path


def assert_matching_pair(svs_path: Path, json_path: Path):
    expected_json_name = f"{svs_path.stem}.svs.json"
    if json_path.name != expected_json_name:
        raise ValueError(
            "Annotation JSON does not match the SVS file.\n"
            f"SVS:  {svs_path.name}\n"
            f"JSON: {json_path.name}\n"
            f"Expected JSON name: {expected_json_name}\n"
            "Provide the matching .svs.json for this SVS."
        )


def main():
    parser = argparse.ArgumentParser(description="Artery-lumen homogeneity analysis pipeline")
    parser.add_argument("--svs", type=str, default=None, help="Path to SVS file")
    parser.add_argument("--json", type=str, default=None, help="Path to annotation JSON")
    parser.add_argument("--level", type=int, default=2, help="OpenSlide pyramid level")
    parser.add_argument(
        "--output-image",
        type=str,
        default=None,
        help="Output overlay image path",
    )
    parser.add_argument(
        "--output-aif-image",
        type=str,
        default=None,
        help="Output AIF overlay image path",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output analysis JSON path",
    )
    parser.add_argument(
        "--output-flags-csv",
        type=str,
        default=None,
        help="Output artery flag summary CSV path",
    )
    parser.add_argument(
        "--output-severity-csv",
        type=str,
        default=None,
        help="Output artery severity CSV path",
    )
    parser.add_argument(
        "--output-case-summary-json",
        type=str,
        default=None,
        help="Output case summary JSON path",
    )
    parser.add_argument(
        "--output-case-summary-pdf",
        type=str,
        default=None,
        help="Output case summary PDF path",
    )
    parser.add_argument(
        "--qupath-script",
        type=str,
        default="plot_lumen_and_centroids.groovy",
        help="Groovy script path for QuPath headless plotting",
    )
    parser.add_argument(
        "--qupath-aif-script",
        type=str,
        default="plot_aif_overlays.groovy",
        help="Groovy script path for AIF QuPath plotting",
    )
    parser.add_argument(
        "--no-qupath",
        action="store_true",
        help="Skip QuPath headless plotting",
    )
    args = parser.parse_args()

    if args.svs and args.json:
        svs_path = resolve_input_path(args.svs)
        json_path = resolve_input_path(args.json)
    else:
        svs_path, json_path = pick_default_input_paths()

    if not svs_path.exists():
        raise FileNotFoundError(f"SVS file not found: {svs_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    assert_matching_pair(svs_path, json_path)
    qupath_paths = build_qupath_paths(svs_path)

    stem = svs_path.stem
    output_stem_dir = OUTPUT_DIR / stem
    output_image = (
        resolve_input_path(args.output_image)
        if args.output_image
        else output_stem_dir / f"{stem}_artery_overlay.png"
    )
    output_aif_image = (
        resolve_input_path(args.output_aif_image)
        if args.output_aif_image
        else output_stem_dir / f"{stem}_aif_overlay.png"
    )
    output_json = (
        resolve_input_path(args.output_json)
        if args.output_json
        else output_stem_dir / f"{stem}_artery_analysis.json"
    )
    output_flags_csv = (
        resolve_input_path(args.output_flags_csv)
        if args.output_flags_csv
        else output_stem_dir / f"{stem}_artery_flags.csv"
    )
    output_severity_csv = (
        resolve_input_path(args.output_severity_csv)
        if args.output_severity_csv
        else output_stem_dir / f"{stem}_severity.csv"
    )
    output_case_summary_json = (
        resolve_input_path(args.output_case_summary_json)
        if args.output_case_summary_json
        else output_stem_dir / f"{stem}_case_summary.json"
    )
    output_case_summary_pdf = (
        resolve_input_path(args.output_case_summary_pdf)
        if args.output_case_summary_pdf
        else output_stem_dir / f"{stem}_case_summary.pdf"
    )
    qupath_script = resolve_input_path(args.qupath_script)
    qupath_aif_script = resolve_input_path(args.qupath_aif_script)

    print("PIPELINE STARTED")
    print(f"SVS:  {svs_path}")
    print(f"JSON: {json_path}")

    loader = AnnotationLoader(json_path)
    data = loader.load()
    artery_polygons = loader.extract_artery_polygons(data)
    artery_lumen_map = loader.extract_artery_lumen_mapping(data)
    updated_map = loader.check_homogeneity(
        artery_lumen_map,
        n_rays=72,
        tolerance=0.1,
        min_ratio=0.55,
    )
    print(f"Mapped arteries: {len(updated_map)}")

    slide_manager = SlideManager(svs_path, level=args.level)
    img_np = slide_manager.read_image()
    write_qupath_csvs(updated_map, qupath_paths["output_dir"])

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    aif_records, aif_summary = draw_aif_overlay(
        img_np,
        artery_polygons,
        gray,
        output_aif_image,
        slide_manager.scale,
    )
    if aif_records:
        write_aif_qupath_csvs(aif_records, qupath_paths["output_dir"], aif_summary=aif_summary)

    artery_flags = compute_artery_flags(updated_map, aif_records, aif_summary=aif_summary)
    write_artery_flags_csv(artery_flags, output_flags_csv)
    artery_severity = compute_artery_severity(artery_flags)
    write_artery_severity_csv(artery_severity, output_severity_csv)
    write_qupath_advanced_csvs(updated_map, artery_flags, artery_severity, qupath_paths["output_dir"])
    case_summary = build_case_summary(stem, updated_map, artery_flags, artery_severity)
    output_case_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with output_case_summary_json.open("w", encoding="utf-8") as f:
        json.dump(case_summary, f, indent=2)
    write_case_summary_pdf(
        case_summary,
        artery_flags,
        artery_severity,
        output_case_summary_pdf,
        output_image,
        output_aif_image,
    )
    draw_artery_on_slide(
        img_np,
        updated_map,
        output_image,
        slide_manager.scale,
        artery_flags=artery_flags,
        artery_severity=artery_severity,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in updated_map.items():
        key_int = _safe_int(key)
        flag_payload = artery_flags.get(key_int, {})
        severity_payload = artery_severity.get(key_int, {})
        serializable[str(key)] = {
            "artery": np.asarray(value.get("artery", [])).tolist(),
            "lumens": [np.asarray(x).tolist() for x in value.get("lumens", [])],
            "homogeneity": value.get("homogeneity"),
            "shape_similarity": value.get("shape_similarity"),
            "concentricity_index": value.get("concentricity_index"),
            "lumen_count": value.get("lumen_count"),
            "multi_lumen_complexity": value.get("multi_lumen_complexity"),
            "wall_thickness_min": value.get("wall_thickness_min"),
            "wall_thickness_mean": value.get("wall_thickness_mean"),
            "wall_thickness_max": value.get("wall_thickness_max"),
            "wall_thickness_cv": value.get("wall_thickness_cv"),
            "wall_thickness_iqr": value.get("wall_thickness_iqr"),
            "narrowing_index": value.get("narrowing_index"),
            "ray_valid_fraction": value.get("ray_valid_fraction"),
            "flag_level": flag_payload.get("flag_level"),
            "risk_score": flag_payload.get("risk_score"),
            "flag_reasons": flag_payload.get("reasons", []),
            "aif_percent": flag_payload.get("aif_percent"),
            "contour_quality_mean": flag_payload.get("contour_quality_mean"),
            "severity_grade": severity_payload.get("severity_grade"),
            "severity_label": severity_payload.get("severity_label"),
            "severity_reasons": severity_payload.get("severity_reasons", []),
        }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    if not args.no_qupath:
        if not qupath_script.exists():
            raise FileNotFoundError(f"Groovy script not found: {qupath_script}")
        ensure_qupath_project_template(qupath_paths["project_dir"])
        project_image_path = ensure_project_image(svs_path, qupath_paths["project_dir"])
        ensure_qupath_project_files(
            project_image_path,
            qupath_paths["project_file"],
            qupath_paths["project_dir"],
            qupath_paths["output_dir"],
        )
        run_qupath_headless(project_image_path.name, qupath_script, qupath_paths["project_file"])
        if qupath_aif_script.exists() and aif_records:
            run_qupath_headless(
                project_image_path.name,
                qupath_aif_script,
                qupath_paths["project_file"],
            )
        elif aif_records:
            print(f"Warning: AIF QuPath script not found: {qupath_aif_script}")

    print(f"Overlay saved:  {output_image}")
    print(f"AIF overlay saved:  {output_aif_image}")
    print(f"Analysis saved: {output_json}")
    print(f"Flag summary saved: {output_flags_csv}")
    print(f"Severity summary saved: {output_severity_csv}")
    print(f"Case summary JSON saved: {output_case_summary_json}")
    print(f"Case summary PDF saved: {output_case_summary_pdf}")
    print(f"QuPath project: {qupath_paths['project_dir']}")
    print("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()