"""
render_modelnet_previews.py
---------------------------
Render thumbnails for the exact ModelNet10 meshes selected by train_svm.py.

Run with ParaView's Python executable:

    pvpython scripts/render_modelnet_previews.py

Outputs are written to:

    output/modelnet_mesh_previews/

The script intentionally mirrors train_svm.py's file selection:
sorted .off files, first N_TRAIN_PER_CLASS from train and first
N_TEST_PER_CLASS from test for each class.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

import vtk

try:
    from paraview.simple import (
        Delete,
        GetActiveViewOrCreate,
        Hide,
        Render,
        ResetCamera,
        SaveScreenshot,
        Show,
        TrivialProducer,
    )
except ImportError as exc:
    raise SystemExit(
        "This script needs ParaView's Python environment. Run it with pvpython, "
        "for example:\n\n"
        "    pvpython scripts/render_modelnet_previews.py\n"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from path_config import roots_for

ROOTS = roots_for(__file__)
REPO_ROOT = ROOTS["REPO_ROOT"]
TEST_ROOT = ROOTS["TEST_ROOT"]

DEFAULT_CLASSES = ["bathtub", "chair", "toilet", "desk"]
MODELNET_ROOT = ROOTS["MODELNET_ROOT"]
DEFAULT_N_TRAIN_PER_CLASS = 30
DEFAULT_N_TEST_PER_CLASS = 5

DEFAULT_OUTPUT_DIR = ROOTS["OUTPUT_ROOT"] / "modelnet_mesh_previews"
IMAGE_SIZE = 360

CLASS_COLORS = {
    "bathtub": (0.20, 0.43, 0.82),
    "chair": (0.86, 0.35, 0.18),
    "toilet": (0.20, 0.62, 0.35),
    "desk": (0.66, 0.35, 0.73),
}


def load_train_svm_selection_config() -> tuple[list[str], int, int]:
    config = {
        "CLASSES": DEFAULT_CLASSES,
        "N_TRAIN_PER_CLASS": DEFAULT_N_TRAIN_PER_CLASS,
        "N_TEST_PER_CLASS": DEFAULT_N_TEST_PER_CLASS,
    }
    train_svm_path = SCRIPT_DIR / "train_svm.py"
    tree = ast.parse(train_svm_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in config:
                config[target.id] = ast.literal_eval(node.value)
    return (
        list(config["CLASSES"]),
        int(config["N_TRAIN_PER_CLASS"]),
        int(config["N_TEST_PER_CLASS"]),
    )


def selected_meshes() -> list[tuple[str, str, int, Path]]:
    meshes = []
    classes, n_train_per_class, n_test_per_class = load_train_svm_selection_config()
    for cls_name in classes:
        for split, n_per_class in (
            ("train", n_train_per_class),
            ("test", n_test_per_class),
        ):
            split_dir = MODELNET_ROOT / cls_name / split
            split_meshes = sorted(split_dir.glob("*.off"))[:n_per_class]
            for index, mesh_path in enumerate(split_meshes, start=1):
                meshes.append((split, cls_name, index, mesh_path))
    return meshes


def off_tokens(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in line.split():
                yield token


def read_off(path: Path) -> vtk.vtkPolyData:
    tokens = off_tokens(path)
    header = next(tokens)
    if header != "OFF":
        raise ValueError(f"{path} is not an OFF file; found header {header!r}")

    n_vertices = int(next(tokens))
    n_faces = int(next(tokens))
    _ = int(next(tokens))

    points = vtk.vtkPoints()
    for _ in range(n_vertices):
        x = float(next(tokens))
        y = float(next(tokens))
        z = float(next(tokens))
        points.InsertNextPoint(x, y, z)

    polygons = vtk.vtkCellArray()
    for _ in range(n_faces):
        n_indices = int(next(tokens))
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(n_indices)
        for point_index in range(n_indices):
            polygon.GetPointIds().SetId(point_index, int(next(tokens)))
        polygons.InsertNextCell(polygon)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(polygons)

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(mesh)
    triangle_filter.Update()

    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(triangle_filter.GetOutput())
    normal_filter.ConsistencyOn()
    normal_filter.AutoOrientNormalsOn()
    normal_filter.SplittingOff()
    normal_filter.Update()

    return normalize_polydata(normal_filter.GetOutput())


def normalize_polydata(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    bounds = polydata.GetBounds()
    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    extent = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1e-8,
    )

    normalized = vtk.vtkPolyData()
    normalized.DeepCopy(polydata)
    points = normalized.GetPoints()
    for point_id in range(points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(point_id)
        points.SetPoint(
            point_id,
            (x - center[0]) / extent,
            (y - center[1]) / extent,
            (z - center[2]) / extent,
        )
    points.Modified()
    normalized.Modified()
    return normalized


def render_mesh(
    mesh_path: Path,
    screenshot_path: Path,
    class_name: str,
    view,
    image_size: int,
) -> None:
    polydata = read_off(mesh_path)
    source = TrivialProducer(registrationName=mesh_path.stem)
    source.GetClientSideObject().SetOutput(polydata)

    display = Show(source, view)
    display.Representation = "Surface"
    display.DiffuseColor = CLASS_COLORS.get(class_name, (0.65, 0.65, 0.65))
    display.Ambient = 0.25
    display.Diffuse = 0.85
    display.Specular = 0.15
    display.SpecularPower = 20.0

    view.ViewSize = [image_size, image_size]
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0

    camera = view.GetActiveCamera()
    camera.SetFocalPoint(0.0, 0.0, 0.0)
    camera.SetPosition(2.0, -3.0, 1.6)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(0.72)
    ResetCamera(view)
    camera.SetFocalPoint(0.0, 0.0, 0.0)
    camera.SetPosition(2.0, -3.0, 1.6)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(0.72)
    camera.Elevation(20)
    camera.Azimuth(35)
    Render(view)

    SaveScreenshot(str(screenshot_path), view, ImageResolution=[image_size, image_size])

    Hide(source, view)
    Delete(source)


def write_gallery_html(output_dir: Path, rows: list[dict[str, str]]) -> None:
    html_path = output_dir / "index.html"
    items = []
    for row in rows:
        label = (
            f"{row['split']} | {row['class']} | "
            f"{row['index']} | {row['mesh_name']}"
        )
        items.append(
            "      <figure>\n"
            f"        <img src=\"{row['image_name']}\" alt=\"{label}\">\n"
            f"        <figcaption>{label}</figcaption>\n"
            "      </figure>"
        )

    html = (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <title>ModelNet SVM Mesh Previews</title>\n"
        "  <style>\n"
        "    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }\n"
        "    h1 { font-size: 22px; }\n"
        "    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 14px; }\n"
        "    figure { margin: 0; border: 1px solid #ddd; border-radius: 6px; padding: 8px; }\n"
        "    img { display: block; width: 100%; height: auto; }\n"
        "    figcaption { font-size: 11px; line-height: 1.35; word-break: break-word; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>ModelNet SVM Mesh Previews</h1>\n"
        "  <p>Same sorted train/test files selected by scripts/train_svm.py.</p>\n"
        "  <main class=\"grid\">\n"
        + "\n".join(items)
        + "\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )
    html_path.write_text(html, encoding="utf-8")


def write_contact_sheet(output_dir: Path, image_names: list[str], columns: int) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow is not available, so contact_sheet.png was skipped.")
        return

    thumb_size = 140
    label_height = 34
    padding = 8
    rows = (len(image_names) + columns - 1) // columns
    width = columns * (thumb_size + padding) + padding
    height = rows * (thumb_size + label_height + padding) + padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i, image_name in enumerate(image_names):
        image = Image.open(output_dir / image_name).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        col = i % columns
        row = i // columns
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + label_height + padding)
        sheet.paste(image, (x, y))
        draw.text((x, y + thumb_size + 4), image_name[:28], fill=(20, 20, 20), font=font)

    sheet.save(output_dir / "contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ParaView thumbnails for the same meshes used by train_svm.py."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--contact-sheet-columns", type=int, default=10)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meshes = selected_meshes()
    if not meshes:
        raise SystemExit(f"No .off meshes found under {MODELNET_ROOT}")

    view = GetActiveViewOrCreate("RenderView")
    rows = []

    print(f"Rendering {len(meshes)} meshes to {output_dir}")
    for split, cls_name, index, mesh_path in meshes:
        image_name = f"{split}_{cls_name}_{index:03d}_{mesh_path.stem}.png"
        screenshot_path = output_dir / image_name
        print(f"  {split:5s} {cls_name:7s} {index:03d} {mesh_path.name}")
        render_mesh(mesh_path, screenshot_path, cls_name, view, args.image_size)
        rows.append(
            {
                "split": split,
                "class": cls_name,
                "index": str(index),
                "mesh_name": mesh_path.name,
                "mesh_path": str(mesh_path),
                "image_name": image_name,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class", "index", "mesh_name", "mesh_path", "image_name"],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_gallery_html(output_dir, rows)
    write_contact_sheet(
        output_dir,
        [row["image_name"] for row in rows],
        columns=args.contact_sheet_columns,
    )

    print(f"Saved thumbnails: {output_dir}")
    print(f"Saved manifest  : {manifest_path}")
    print(f"Saved gallery   : {output_dir / 'index.html'}")
    if (output_dir / "contact_sheet.png").exists():
        print(f"Saved sheet     : {output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
