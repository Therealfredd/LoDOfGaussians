"""
LoD Trainer - a standalone launcher for training Gaussian splats with
"A LoD of Gaussians" (https://github.com/FelixWindisch/LoDOfGaussians)
on your own COLMAP datasets.

Runs on the system Python (tkinter only, no third-party imports) so it can
bootstrap the whole toolchain itself: clone the repo, build an isolated venv,
compile the CUDA extensions, then drive training from a GUI.
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_NAME = "LoD Trainer"
APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR / "repo"
VENV_DIR = APP_DIR / "venv"
# Keep JIT-compiled kernels inside the app folder rather than the user profile,
# so setup owns them and can rebuild deterministically.
TORCH_EXT_DIR = APP_DIR / "torch_extensions"
GSPLAT_PATCH_MARKER = "# lod_trainer: MSVC-safe flags"
VIEWGRAPH_PATCH_MARKER = "# lod_trainer: accept standard COLMAP pose lines"
SETTINGS_FILE = APP_DIR / "settings.json"
REPO_URL = "https://github.com/FelixWindisch/LoDOfGaussians.git"
FUSED_SSIM_URL = "git+https://github.com/rahul-goel/fused-ssim/"

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# train.py imports networkx but requirements.txt omits it. numpy is pinned
# below 2.0 because this codebase predates the numpy 2 ABI break.
CORE_PACKAGES = [
    "numpy<2",
    "networkx",
    "plyfile",
    "tqdm",
    "joblib",
    "exif",
    "scikit-learn",
    "opencv-python",
    "matplotlib",
    "scipy",
    "tensorboard",
    "torchviz",
    "psutil",
    "gsplat",
]

# Only used by the depth-preprocessing tools, not by train.py. Heavy and
# prone to resolver conflicts, so it is opt-in.
EXTRA_PACKAGES = ["timm==0.4.5", "gradio==4.29.0", "gradio_imageslider"]

# Mirrors configs/default.json from the repo so the UI works before cloning.
DEFAULT_CONFIG = {
    "iterations": 250000,
    "coarse_iterations": 60000,
    "lr_multiplier": 1.0,
    "SH_degree": 1,
    "position_lr_init": 2e-05,
    "position_lr_final": 2e-07,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 230000,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "exposure_lr_init": 0.001,
    "exposure_lr_final": 0.0001,
    "exposure_lr_delay_steps": 5000,
    "exposure_lr_delay_mult": 0.001,
    "percent_dense": 0.01,
    "lambda_dssim": 0.2,
    "densification_interval": 1000,
    "densification": "classic",
    "opacity_reset_interval": 100000000000.0,
    "densify_from_iter": 100,
    "densify_until_iter": 200000,
    "densify_grad_threshold": 5e-06,
    "depth_l1_weight_init": 1.0,
    "depth_l1_weight_final": 0.01,
    "noise_lr": 1e5,
    "lambda_scaling": 0.1,
    "lambda_opacity": 0.1,
    "densify_percent": 1.02,
    "cap_max": 100000000,
    "graph_view_select": True,
    "view_graph_k": 100,
    "use_bounding_spheres": False,
    "use_frustum_culling": True,
    "storage_device": "cpu",
    "SPT_root_volume": 25,
    "target_granularity_pixels": 3,
    "min_SPT_size": 256,
    "use_GPU_caching": True,
    "cache_size": 22000000,
    "cache_size_after_reduction": 18000000,
    "clear_cache_interval": 1000,
    "llff_hold": 100,
    "vary_distance_multiplier": True,
    "output_file_name": "result.dhier",
}

# (config key, label, widget kind, help text)
PARAM_SPEC = [
    ("Quality / length", [
        ("coarse_iterations", "Coarse iterations", "int",
         "Scaffold pass. Runs first and is comparatively cheap."),
        ("iterations", "Fine iterations", "int",
         "Hierarchical pass. The main cost driver."),
        ("SH_degree", "SH degree", "int",
         "Spherical harmonics degree. 1 keeps memory low; 3 captures more view-dependent colour."),
        ("lambda_dssim", "SSIM weight", "float",
         "Weight of the D-SSIM term against L1. 0.2 is standard."),
    ]),
    ("Densification", [
        ("densification", "Strategy", "choice:classic,MCMC",
         "'classic' grows by gradient threshold. 'MCMC' grows to a fixed budget (cap_max)."),
        ("cap_max", "Max Gaussians (MCMC)", "int",
         "Hard splat budget when using MCMC. Ignored by 'classic'."),
        ("densify_grad_threshold", "Densify grad threshold", "float",
         "Lower = denser and slower. Used by 'classic'."),
        ("densification_interval", "Densify every N iters", "int", ""),
        ("densify_from_iter", "Densify from iter", "int", ""),
        ("densify_until_iter", "Densify until iter", "int",
         "Should stay below the fine iteration count."),
        ("percent_dense", "Percent dense", "float", ""),
    ]),
    ("Level of detail / streaming", [
        ("cache_size", "GPU cache size", "int",
         "Gaussians held on the GPU. The main VRAM dial - lower this first if you hit OOM."),
        ("cache_size_after_reduction", "Cache size after reduction", "int",
         "Target after a cache flush. Keep below the cache size."),
        ("use_GPU_caching", "Use GPU caching", "bool", ""),
        ("clear_cache_interval", "Clear cache every N iters", "int", ""),
        ("storage_device", "Storage device", "choice:cpu,cuda",
         "Where the full model lives. 'cpu' streams from system RAM and is what makes large scenes fit."),
        ("target_granularity_pixels", "Target granularity (px)", "float",
         "LoD cut selection. Higher = coarser and faster."),
        ("SPT_root_volume", "SPT root volume", "float", ""),
        ("min_SPT_size", "Min SPT size", "int", ""),
        ("use_frustum_culling", "Frustum culling", "bool", ""),
        ("use_bounding_spheres", "Bounding spheres", "bool", ""),
    ]),
    ("Views and output", [
        ("graph_view_select", "Graph view selection", "bool",
         "Samples training views along a nearest-neighbour camera graph. Needs more than 'view graph k' images."),
        ("view_graph_k", "View graph k", "int",
         "Neighbours per camera. Note: the repo hardcodes k=100 internally."),
        ("llff_hold", "Hold out every Nth image", "int",
         "Test-set stride. Use -1 to train on every image."),
        ("vary_distance_multiplier", "Vary distance multiplier", "bool", ""),
        ("output_file_name", "Output hierarchy name", "str",
         "Final .dhier file written into the output folder."),
    ]),
    ("Learning rates", [
        ("lr_multiplier", "LR multiplier", "float", ""),
        ("position_lr_init", "Position LR init", "float", ""),
        ("position_lr_final", "Position LR final", "float", ""),
        ("position_lr_max_steps", "Position LR max steps", "float", ""),
        ("feature_lr", "Feature LR", "float", ""),
        ("opacity_lr", "Opacity LR", "float", ""),
        ("scaling_lr", "Scaling LR", "float", ""),
        ("rotation_lr", "Rotation LR", "float", ""),
    ]),
]

PRESETS = {
    "Default (paper settings)": {},
    "Fast preview": {
        "coarse_iterations": 15000,
        "iterations": 60000,
        "densify_until_iter": 45000,
        "position_lr_max_steps": 55000,
        "target_granularity_pixels": 4,
    },
    "High quality": {
        "coarse_iterations": 90000,
        "iterations": 350000,
        "densify_until_iter": 280000,
        "position_lr_max_steps": 330000,
        "SH_degree": 3,
        "target_granularity_pixels": 2,
    },
    "Low VRAM (8-12 GB)": {
        "cache_size": 8000000,
        "cache_size_after_reduction": 6000000,
        "SH_degree": 1,
        "target_granularity_pixels": 4,
    },
}


# --------------------------------------------------------------------------
# environment probing
# --------------------------------------------------------------------------

def _run_capture(cmd, **kwargs):
    """Run a command, return (returncode, combined output). Never raises."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace",
            creationflags=CREATE_NO_WINDOW, timeout=120, **kwargs)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:  # noqa: BLE001 - probing must never be fatal
        return 1, str(exc)


def find_python310():
    """Locate a real CPython 3.10/3.11 interpreter, avoiding the Store stub."""
    candidates = []
    if os.name == "nt":
        for ver in ("3.10", "3.11"):
            rc, out = _run_capture(["py", f"-{ver}", "-c",
                                    "import sys; print(sys.executable)"])
            if rc == 0 and out.strip():
                candidates.append(out.strip().splitlines()[-1])
    for name in ("python3.10", "python3.11", "python3", "python"):
        exe = shutil.which(name)
        if not exe:
            continue
        # The Windows Store alias resolves but produces no version output.
        if "WindowsApps" in exe:
            continue
        rc, out = _run_capture([exe, "-c",
                                "import sys; print('%d.%d' % sys.version_info[:2])"])
        if rc == 0 and out.strip().startswith(("3.10", "3.11")):
            candidates.append(exe)
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def find_cuda_toolkits():
    """Return [(version, root_path)] for every installed CUDA toolkit."""
    found = {}
    roots = []
    if os.name == "nt":
        roots.append(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"))
    roots.append(Path("/usr/local"))
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            m = re.fullmatch(r"v?(\d+)\.(\d+)", child.name.replace("cuda-", ""))
            if m and (child / "bin").is_dir():
                found[f"{m.group(1)}.{m.group(2)}"] = str(child)
    nvcc = shutil.which("nvcc")
    if nvcc:
        rc, out = _run_capture([nvcc, "--version"])
        m = re.search(r"release (\d+)\.(\d+)", out)
        if rc == 0 and m:
            found.setdefault(f"{m.group(1)}.{m.group(2)}",
                             str(Path(nvcc).parent.parent))
    return sorted(found.items(), key=lambda kv: [int(x) for x in kv[0].split(".")],
                  reverse=True)


def torch_index_for_cuda(cuda_version):
    """Map a CUDA toolkit version onto a PyTorch wheel index."""
    try:
        major, minor = (int(x) for x in cuda_version.split(".")[:2])
    except ValueError:
        return "cu124"
    if major >= 13:
        return "cu128"
    if major == 12:
        if minor >= 8:
            return "cu128"
        if minor >= 6:
            return "cu126"
        if minor >= 4:
            return "cu124"
        return "cu121"
    return "cu118"


def find_vs_build_env():
    """Return (vcvars_path, msvc_toolset_version) for the newest MSVC found."""
    if os.name != "nt":
        return None, None
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) \
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None, None
    rc, out = _run_capture([str(vswhere), "-latest", "-products", "*",
                            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                            "-property", "installationPath"])
    if rc != 0 or not out.strip():
        return None, None
    install = Path(out.strip().splitlines()[-1])
    vcvars = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    toolset = None
    msvc_dir = install / "VC" / "Tools" / "MSVC"
    if msvc_dir.is_dir():
        versions = sorted(p.name for p in msvc_dir.iterdir() if p.is_dir())
        if versions:
            toolset = versions[-1]
    return (str(vcvars) if vcvars.exists() else None), toolset


def needs_unsupported_compiler_flag(cuda_version, toolset):
    """
    nvcc hard-fails on MSVC toolsets newer than it knows about. CUDA 12.4 and
    earlier reject MSVC 14.40+ (_MSC_VER >= 1940), which is exactly the
    combination that ships on current VS installs.
    """
    if not cuda_version or not toolset:
        return False
    try:
        cmaj, cmin = (int(x) for x in cuda_version.split(".")[:2])
        tmaj, tmin = (int(x) for x in toolset.split(".")[:2])
    except ValueError:
        return False
    if (tmaj, tmin) < (14, 40):
        return False
    return (cmaj, cmin) < (12, 6)


def detect_gpu():
    rc, out = _run_capture(["nvidia-smi",
                            "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"])
    if rc == 0 and out.strip():
        return out.strip().splitlines()[0].strip()
    return None


def gpu_arch_list():
    """Compute capability of GPU 0, e.g. '8.6' - narrows and speeds up builds."""
    rc, out = _run_capture(["nvidia-smi", "--query-gpu=compute_cap",
                            "--format=csv,noheader"])
    if rc == 0 and out.strip():
        cap = out.strip().splitlines()[0].strip()
        if re.fullmatch(r"\d+\.\d+", cap):
            return cap
    return None


def venv_python(venv_dir=VENV_DIR):
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def load_vcvars_env(vcvars_path):
    """
    Run vcvars64.bat and capture the environment it sets, so subsequent
    builds (and gsplat's runtime JIT compile) can find cl.exe.
    """
    if not vcvars_path:
        return None
    try:
        p = subprocess.run(
            f'cmd /c "call "{vcvars_path}" >nul 2>&1 && set"',
            capture_output=True, text=True, errors="replace", shell=True,
            creationflags=CREATE_NO_WINDOW, timeout=180)
    except Exception:  # noqa: BLE001
        return None
    if p.returncode != 0:
        return None
    env = {}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    return env or None


# --------------------------------------------------------------------------
# dataset validation
# --------------------------------------------------------------------------

# Number of focal-length parameters each COLMAP model puts before cx, cy.
# Everything after cx, cy is distortion.
CAMERA_FOCAL_PARAMS = {
    "SIMPLE_PINHOLE": 1, "PINHOLE": 2, "SIMPLE_RADIAL": 1, "RADIAL": 1,
    "OPENCV": 2, "OPENCV_FISHEYE": 2, "FULL_OPENCV": 2, "FOV": 2,
    "SIMPLE_RADIAL_FISHEYE": 1, "RADIAL_FISHEYE": 1, "THIN_PRISM_FISHEYE": 2,
}
PINHOLE_MODEL_ID = 1
MODEL_STEMS = ("cameras", "images", "points3D")


def _parse_cameras_text(path):
    cams = []
    with open(path, "r", encoding="utf-8", errors="replace") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            e = line.split()
            cams.append({"id": int(e[0]), "model": e[1], "w": int(e[2]),
                         "h": int(e[3]), "params": [float(x) for x in e[4:]]})
    return cams


def _parse_cameras_binary(path):
    import struct
    id_to_name = {
        0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
        4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE",
    }
    n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
    cams = []
    with open(path, "rb") as fid:
        count = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(count):
            cid, model_id, w, h = struct.unpack("<iiQQ", fid.read(24))
            k = n_params.get(model_id, 4)
            params = list(struct.unpack("<" + "d" * k, fid.read(8 * k)))
            cams.append({"id": cid, "model": id_to_name.get(model_id, "UNKNOWN"),
                         "w": w, "h": h, "params": params})
    return cams


def _write_cameras_text(path, cams):
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("# Camera list with one line of data per camera:\n")
        fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fid.write(f"# Number of cameras: {len(cams)}\n")
        for c in cams:
            params = " ".join(repr(float(p)) for p in c["params"])
            fid.write(f"{c['id']} {c['model']} {c['w']} {c['h']} {params}\n")


def _write_cameras_binary(path, cams):
    import struct
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cams)))
        for c in cams:
            fid.write(struct.pack("<iiQQ", c["id"], PINHOLE_MODEL_ID,
                                  c["w"], c["h"]))
            fid.write(struct.pack("<4d", *[float(p) for p in c["params"][:4]]))


def to_pinhole(cam):
    """
    Convert a camera to PINHOLE. Only valid when distortion is zero - i.e. the
    images are already undistorted. Returns (new_cam, reason_if_refused).
    """
    model, params = cam["model"], cam["params"]
    if model == "PINHOLE" and len(params) == 4:
        return dict(cam), None
    n_focal = CAMERA_FOCAL_PARAMS.get(model)
    if n_focal is None or len(params) < n_focal + 2:
        return None, f"unrecognised camera model '{model}'"
    if n_focal == 1:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        dist = params[3:]
    else:
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
        dist = params[4:]
    if any(abs(d) > 1e-9 for d in dist):
        return None, (f"camera {cam['id']} ({model}) has non-zero distortion")
    out = dict(cam)
    out["model"] = "PINHOLE"
    out["params"] = [fx, fy, cx, cy]
    return out, None


def resolve_model_file(directory, stem, fmt):
    """Find e.g. points3D.txt regardless of how it was capitalised."""
    target = f"{stem}.{fmt}".lower()
    try:
        for p in directory.iterdir():
            if p.is_file() and p.name.lower() == target:
                return p
    except OSError:
        pass
    return None


def _model_format_in(directory):
    """Return 'bin'/'txt' if a directory holds a complete model, else None."""
    for fmt in ("bin", "txt"):
        if all(resolve_model_file(directory, s, fmt) for s in MODEL_STEMS):
            return fmt
    return None


def find_model_dir(root):
    """
    Locate the COLMAP model. Returns (dir, is_correct_location, fmt) where fmt
    is 'bin', 'txt' or None. Exports commonly drop the model straight into the
    project root, or into sparse/ without the '0' subfolder.
    """
    root = Path(root)
    for candidate, correct in ((root / "sparse" / "0", True),
                               (root / "sparse", False),
                               (root, False)):
        if not candidate.is_dir():
            continue
        fmt = _model_format_in(candidate)
        if fmt:
            return candidate, correct, fmt
    return None, False, None


def find_images_dir(root):
    """
    Locate the images. Returns (dir, is_correct_location, count). Captures often
    keep the images loose in the project root alongside the model files.
    """
    root = Path(root)
    sub = root / "images"
    if sub.is_dir():
        n = count_images(sub)
        if n:
            return sub, True, n
    loose = sum(1 for p in root.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if loose:
        return root, False, loose
    if sub.is_dir():
        return sub, True, 0
    return None, False, 0


def analyse_cameras(model_dir, fmt):
    """Returns (models_summary, needs_conversion, blocking_reason)."""
    src = resolve_model_file(Path(model_dir), "cameras", fmt)
    if src is None:
        return {}, False, "cameras file not found"
    try:
        cams = (_parse_cameras_binary(src) if fmt == "bin"
                else _parse_cameras_text(src))
    except Exception as exc:  # noqa: BLE001
        return {}, False, f"could not read cameras: {exc}"
    if not cams:
        return {}, False, "no cameras found"

    summary = {}
    for c in cams:
        summary[c["model"]] = summary.get(c["model"], 0) + 1

    # The text reader asserts PINHOLE exactly; the binary path also accepts
    # SIMPLE_PINHOLE further downstream.
    ok_models = {"PINHOLE"} if fmt == "txt" else {"PINHOLE", "SIMPLE_PINHOLE"}
    if set(summary) <= ok_models:
        return summary, False, None

    for c in cams:
        _, reason = to_pinhole(c)
        if reason:
            return summary, False, reason
    return summary, True, None


def prepare_dataset(root, log):
    """
    Reshape a real-world COLMAP export into the layout the trainer expects:
    images under images/, the model under sparse/0/, and PINHOLE cameras.
    Originals are kept alongside as .original.
    """
    root = Path(root)
    model_dir, model_ok, fmt = find_model_dir(root)
    if model_dir is None:
        return False, ("No COLMAP model found. Expected cameras, images and "
                       "points3D (.bin or .txt) in the project root, sparse/, "
                       "or sparse/0/.")

    did = []

    # 1. Images loose in the project root -> images/
    img_dir, img_ok, n_images = find_images_dir(root)
    if img_dir is not None and not img_ok and n_images:
        target = root / "images"
        target.mkdir(parents=True, exist_ok=True)
        log(f"Moving {n_images} images into images/", "step")
        moved = 0
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                shutil.move(str(p), str(target / p.name))
                moved += 1
        log(f"  moved {moved} images", "ok")
        did.append(f"moved {moved} images into images/")

    # 2. Model files -> sparse/0/, normalising capitalisation
    if not model_ok:
        target = root / "sparse" / "0"
        target.mkdir(parents=True, exist_ok=True)
        from_root = model_dir.resolve() == root.resolve()
        log("Moving COLMAP model into sparse/0/", "step")
        for stem in MODEL_STEMS:
            src = resolve_model_file(model_dir, stem, fmt)
            if src is None:
                continue
            dst = target / f"{stem}.{fmt}"
            shutil.move(str(src), str(dst))
            log(f"  {src.name} -> sparse/0/{dst.name}", "out")
        if not from_root:
            # A dedicated sparse/ folder: bring any stragglers along too.
            for p in list(model_dir.iterdir()):
                if p.is_file():
                    shutil.move(str(p), str(target / p.name))
                    log(f"  {p.name} -> sparse/0/{p.name}", "out")
        model_dir = target
        did.append("moved the model into sparse/0/")

    summary, needs_conv, blocker = analyse_cameras(model_dir, fmt)
    log(f"Camera models: {summary}", "info")
    if blocker:
        return False, (
            f"Cannot use this model: {blocker}.\n"
            "The images still carry lens distortion, so they must be "
            "undistorted first:\n"
            "    colmap image_undistorter --image_path images "
            "--input_path sparse/0 --output_path undistorted")
    if not needs_conv:
        log("Cameras are already PINHOLE.", "ok")
    else:
        src = resolve_model_file(model_dir, "cameras", fmt)
        backup = src.with_name(src.name + ".original")
        cams = (_parse_cameras_binary(src) if fmt == "bin"
                else _parse_cameras_text(src))
        converted = []
        for c in cams:
            new, reason = to_pinhole(c)
            if reason:
                return False, reason
            converted.append(new)
        if not backup.exists():
            shutil.copy2(src, backup)
            log(f"Backed up original -> {backup.name}", "ok")
        if fmt == "bin":
            _write_cameras_binary(src, converted)
        else:
            _write_cameras_text(src, converted)
        log(f"Converted {len(converted)} cameras to PINHOLE (distortion was "
            f"zero, so this is lossless).", "ok")
        did.append(f"converted {len(converted)} cameras to PINHOLE")

    if not did:
        return True, "Dataset was already in the expected layout."
    return True, "Prepared: " + ", ".join(did) + "."


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images(images_dir):
    """Count images, staying shallow when possible - these folders get large."""
    top = [p for p in images_dir.iterdir()
           if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if top:
        return len(top)
    return sum(1 for p in images_dir.rglob("*")
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def inspect_dataset(root):
    """
    Validate a COLMAP project folder.
    Returns (ok, image_count, messages, fixable) where `fixable` means the
    problems can be resolved by prepare_dataset().
    """
    msgs = []
    fixable = False
    root = Path(root)
    if not root.is_dir():
        return False, 0, [("error", "Folder does not exist.")], False

    ok = True
    # train.py shells out to GaussianHierarchyCreator with an unquoted command
    # string, so any space in the path silently splits into the wrong arguments.
    if " " in str(root):
        msgs.append(("error",
                     "Path contains spaces. The hierarchy step builds an unquoted "
                     "shell command and will fail. Move the dataset to a path "
                     "without spaces."))
        ok = False

    model_dir, correct_place, fmt = find_model_dir(root)
    if model_dir is None:
        msgs.append(("error",
                     "No complete COLMAP model found. Expected cameras, images "
                     "and points3D (.bin or .txt) in the project root, sparse/, "
                     "or sparse/0/."))
        return False, 0, msgs, False

    if not correct_place:
        where = ("the project root" if model_dir.resolve() == root.resolve()
                 else "sparse/")
        msgs.append(("warn",
                     f"Model is in {where} but the trainer reads sparse/0/ - "
                     f"'Prep' will move it."))
        ok = False
        fixable = True
    else:
        msgs.append(("ok", f"COLMAP model found ({fmt} format)."))

    summary, needs_conv, blocker = analyse_cameras(model_dir, fmt)
    if summary:
        msgs.append(("ok", "Cameras: " + ", ".join(
            f"{n}x {m}" for m, n in sorted(summary.items()))))
    if blocker:
        msgs.append(("error", f"Unusable cameras: {blocker}. The images need "
                              f"undistorting with 'colmap image_undistorter'."))
        ok = False
    elif needs_conv:
        msgs.append(("warn",
                     "Cameras are not PINHOLE, but carry no distortion - "
                     "'Prep' will convert them losslessly."))
        ok = False
        fixable = True

    img_dir, images_placed, count = find_images_dir(root)
    if img_dir is None or count == 0:
        msgs.append(("error", "No images found. Expected an images/ folder, or "
                              "image files in the project root."))
        ok = False
    elif not images_placed:
        msgs.append(("warn",
                     f"{count} images are loose in the project root - 'Prep' "
                     f"will move them into images/."))
        ok = False
        fixable = True
    else:
        msgs.append(("ok", f"{count} images found."))

    if (root / "depths").is_dir():
        msgs.append(("ok", "Optional depths/ folder detected - depth loss enabled."))
    if (root / "masks").is_dir():
        msgs.append(("ok", "Optional masks/ folder detected."))

    if not os.access(model_dir, os.W_OK):
        msgs.append(("warn", "sparse/0 is not writable; the view graph cannot "
                             "be cached there."))

    return ok, count, msgs, fixable


# --------------------------------------------------------------------------
# process runner
# --------------------------------------------------------------------------

class Runner:
    """Runs shell steps on a worker thread and streams output to a queue."""

    def __init__(self, log_queue):
        self.q = log_queue
        self.proc = None
        self.cancelled = False
        self._lock = threading.Lock()

    def log(self, text, tag="info"):
        self.q.put((tag, text))

    def stop(self):
        with self._lock:
            self.cancelled = True
            proc = self.proc
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True, creationflags=CREATE_NO_WINDOW)
                else:
                    proc.terminate()
            except Exception as exc:  # noqa: BLE001
                self.log(f"Could not stop process: {exc}", "warn")

    def run(self, cmd, cwd=None, env=None, label=None, check=True):
        """Stream a command's output. Returns the exit code."""
        if self.cancelled:
            return -1
        shown = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        self.log(f"\n$ {shown}", "cmd")
        if label:
            self.log(f"  ({label})", "dim")

        full_env = dict(os.environ)
        if env:
            full_env.update({k: str(v) for k, v in env.items()})
        # Unbuffered child output so the log scrolls live.
        full_env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd) if cwd else None, env=full_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1,
                shell=isinstance(cmd, str), creationflags=CREATE_NO_WINDOW)
        except FileNotFoundError as exc:
            self.log(f"Command not found: {exc}", "error")
            return 127
        with self._lock:
            self.proc = proc

        for line in proc.stdout:
            self.log(line.rstrip("\n"), "out")
        proc.wait()
        with self._lock:
            self.proc = None

        if proc.returncode != 0:
            tag = "error" if check else "warn"
            self.log(f"[exit code {proc.returncode}]", tag)
        return proc.returncode


# --------------------------------------------------------------------------
# install pipeline
# --------------------------------------------------------------------------

class Installer:
    def __init__(self, runner, opts):
        self.r = runner
        self.opts = opts
        self.build_env = {}
        self.cmake_env = {}

    def _fail(self, msg):
        self.r.log(msg, "error")
        return False

    def prepare_build_env(self):
        vc = {}
        vcvars = self.opts.get("vcvars")
        if vcvars:
            self.r.log("Loading MSVC build environment...", "step")
            vc = load_vcvars_env(vcvars) or {}
            if vc:
                self.r.log(f"  MSVC toolset {self.opts.get('toolset')} ready.", "ok")
            else:
                self.r.log("  Could not load vcvars64.bat; builds may fail.", "warn")

        cuda_root = self.opts.get("cuda_root")
        arch = self.opts.get("arch")

        def make(with_vcvars):
            env = dict(vc) if with_vcvars else {}
            if cuda_root:
                env["CUDA_HOME"] = cuda_root
                env["CUDA_PATH"] = cuda_root
                env["PATH"] = str(Path(cuda_root) / "bin") + os.pathsep + \
                    env.get("PATH", os.environ.get("PATH", ""))
            # MSVC + setuptools on Windows.
            env["DISTUTILS_USE_SDK"] = "1"
            TORCH_EXT_DIR.mkdir(parents=True, exist_ok=True)
            env["TORCH_EXTENSIONS_DIR"] = str(TORCH_EXT_DIR)
            if arch:
                env["TORCH_CUDA_ARCH_LIST"] = arch
            if self.opts.get("allow_unsupported"):
                env["NVCC_PREPEND_FLAGS"] = "-allow-unsupported-compiler"
            # Parallel compile without swamping the machine.
            env.setdefault("MAX_JOBS",
                           str(max(1, min(8, (os.cpu_count() or 4) // 2))))
            return env

        if self.opts.get("allow_unsupported"):
            self.r.log("  Using -allow-unsupported-compiler "
                       "(CUDA/MSVC version mismatch).", "warn")

        self.build_env = make(True)
        # The Visual Studio CMake generator fails to identify the C++ compiler
        # when a vcvars environment is injected into it - it sets up its own.
        # nvcc still needs CUDA_PATH and the unsupported-compiler flag.
        self.cmake_env = make(False)
        return True

    def clone_repo(self):
        if (REPO_DIR / "train.py").exists():
            self.r.log("Repository already present, updating submodules...", "step")
            rc = self.r.run(["git", "submodule", "update", "--init", "--recursive"],
                            cwd=REPO_DIR, check=False)
            return True
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR, ignore_errors=True)
        self.r.log("Cloning LoDOfGaussians (with submodules)...", "step")
        # core.longpaths defeats the 260-char MAX_PATH limit that the bundled
        # Eigen sources otherwise trip over on Windows.
        rc = self.r.run(["git", "-c", "core.longpaths=true", "clone", "--recursive",
                         REPO_URL, str(REPO_DIR)])
        if rc != 0:
            return self._fail("Clone failed.")
        return True

    def create_venv(self):
        py = self.opts["python"]
        vpy = venv_python()
        if vpy.exists():
            self.r.log("Virtual environment already exists.", "ok")
        else:
            self.r.log("Creating virtual environment...", "step")
            if self.r.run([py, "-m", "venv", str(VENV_DIR)]) != 0:
                return self._fail("Could not create the virtual environment.")
        self.r.log("Upgrading pip toolchain...", "step")
        # The bundled pip is far too old to resolve modern wheels.
        self.r.run([str(vpy), "-m", "pip", "install", "--upgrade",
                    "pip", "setuptools", "wheel"], check=False)
        return True

    def install_torch(self):
        vpy = venv_python()
        tag = self.opts["torch_index"]
        self.r.log(f"Installing PyTorch ({tag})...", "step")
        rc = self.r.run([str(vpy), "-m", "pip", "install", "torch", "torchvision",
                         "--index-url", f"https://download.pytorch.org/whl/{tag}"])
        if rc != 0:
            return self._fail("PyTorch install failed.")
        rc, out = _run_capture([str(vpy), "-c",
                                "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"])
        self.r.log(f"  torch: {out.strip()}", "ok" if rc == 0 else "error")
        return rc == 0

    def install_requirements(self):
        vpy = venv_python()
        self.r.log("Installing Python dependencies...", "step")
        if self.r.run([str(vpy), "-m", "pip", "install", *CORE_PACKAGES]) != 0:
            return self._fail("Dependency install failed.")
        if self.opts.get("extras"):
            self.r.log("Installing optional extras...", "step")
            self.r.run([str(vpy), "-m", "pip", "install", *EXTRA_PACKAGES], check=False)
        # cmake as a wheel avoids a separate system install.
        self.r.run([str(vpy), "-m", "pip", "install", "cmake"], check=False)
        return True

    def build_extensions(self):
        vpy = venv_python()
        steps = [
            ("simple-knn", [str(vpy), "-m", "pip", "install",
                            str(REPO_DIR / "submodules" / "simple-knn"),
                            "--no-build-isolation"]),
            ("gaussianhierarchy", [str(vpy), "-m", "pip", "install",
                                   str(REPO_DIR / "submodules" / "gaussianhierarchy"),
                                   "--no-build-isolation"]),
            ("fused-ssim", [str(vpy), "-m", "pip", "install", FUSED_SSIM_URL,
                            "--no-build-isolation"]),
        ]
        for name, cmd in steps:
            if self.r.cancelled:
                return False
            self.r.log(f"Building CUDA extension: {name} (this takes a while)...", "step")
            if self.r.run(cmd, env=self.build_env) != 0:
                return self._fail(
                    f"Failed to build {name}. Check that the CUDA toolkit and "
                    f"MSVC versions above are compatible.")
        return True

    def patch_gsplat(self):
        """
        gsplat's JIT build hardcodes GCC-only flags ('-Wno-attributes', '-O3')
        which cl.exe rejects outright: "invalid numeric argument". Rewrite them
        to MSVC equivalents on Windows.
        """
        if os.name != "nt":
            return True
        vpy = venv_python()
        rc, out = _run_capture([str(vpy), "-c",
                                "import importlib.util as u; s = u.find_spec('gsplat');"
                                " print(s.origin if s else '')"])
        origin = out.strip().splitlines()[-1] if rc == 0 and out.strip() else ""
        if not origin or not Path(origin).exists():
            self.r.log("  gsplat not found; skipping patch.", "warn")
            return True
        backend = Path(origin).parent / "cuda" / "_backend.py"
        if not backend.exists():
            self.r.log("  gsplat/cuda/_backend.py not found; skipping patch.", "warn")
            return True

        text = backend.read_text(encoding="utf-8")
        if GSPLAT_PATCH_MARKER in text:
            self.r.log("  gsplat already patched.", "ok")
            return True
        target = '        extra_cflags = [opt_level, "-Wno-attributes"]'
        if target not in text:
            self.r.log("  gsplat flag line not found - version may differ. "
                       "Continuing; the JIT build may fail.", "warn")
            return True
        replacement = (
            f"        {GSPLAT_PATCH_MARKER}\n"
            '        if os.name == "nt":\n'
            '            extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]\n'
            "        else:\n"
            '            extra_cflags = [opt_level, "-Wno-attributes"]'
        )
        backup = backend.with_suffix(".py.original")
        if not backup.exists():
            shutil.copy2(backend, backup)
        backend.write_text(text.replace(target, replacement), encoding="utf-8")
        self.r.log("  Patched gsplat to use MSVC-compatible flags.", "ok")
        # Any half-finished build from before the patch must go.
        shutil.rmtree(TORCH_EXT_DIR, ignore_errors=True)
        return True

    def patch_view_graph(self):
        """
        construct_distance_graph() only treats a line as a camera pose when it
        splits into 11 space-separated fields, but a standard COLMAP images.txt
        pose line has 10. On stock COLMAP data nothing matches, the name list
        stays empty, and it dies on an empty float index array. Accept both.
        """
        target_file = REPO_DIR / "utils" / "view_graph_utils.py"
        if not target_file.exists():
            self.r.log("  view_graph_utils.py not found; skipping.", "warn")
            return True
        text = target_file.read_text(encoding="utf-8")
        if VIEWGRAPH_PATCH_MARKER in text:
            self.r.log("  view graph parser already patched.", "ok")
            return True
        old_cond = '                if len(line.split(" ")) == 11:'
        old_split = '                    split = line.split(" ")'
        if old_cond not in text or old_split not in text:
            self.r.log("  view graph parser differs from the expected source; "
                       "skipping. Turn off 'Graph view selection' if training "
                       "fails there.", "warn")
            return True
        new_cond = (f"                {VIEWGRAPH_PATCH_MARKER}\n"
                    "                if len(line.split()) in (10, 11):")
        text = text.replace(old_cond, new_cond).replace(
            old_split, "                    split = line.split()")
        backup = target_file.with_suffix(".py.original")
        if not backup.exists():
            shutil.copy2(target_file, backup)
        target_file.write_text(text, encoding="utf-8")
        self.r.log("  Patched view graph parser for standard COLMAP files.", "ok")
        return True

    def apply_patches(self):
        return self.patch_gsplat() and self.patch_view_graph()

    def warm_gsplat(self):
        """
        Compile gsplat's CUDA kernels now. They are built on first use, so
        without this the first training run stalls - and any failure would
        surface minutes into training rather than here.
        """
        self.r.log("Compiling gsplat CUDA kernels (several minutes)...", "step")
        rc = self.r.run([str(venv_python()), "-c",
                         "from gsplat.cuda._backend import _C; print('gsplat kernels ready')"],
                        cwd=REPO_DIR, env=self.build_env)
        if rc != 0:
            return self._fail("gsplat kernel compilation failed.")
        return True

    def build_hierarchy_creator(self):
        """CMake build of GaussianHierarchyCreator, which train.py shells out to."""
        gh = REPO_DIR / "submodules" / "gaussianhierarchy"
        if not gh.is_dir():
            return self._fail("gaussianhierarchy submodule missing.")
        cmake = venv_python().parent / ("cmake.exe" if os.name == "nt" else "cmake")
        cmake_cmd = str(cmake) if cmake.exists() else "cmake"

        self.r.log("Configuring GaussianHierarchyCreator...", "step")
        # The bundled CMakeLists declares cmake_minimum_required(VERSION 3.0),
        # which CMake 4.x refuses outright. This policy flag restores the old
        # behaviour; older CMake versions simply ignore the unused variable.
        if self.r.run([cmake_cmd, ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                       "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"],
                      cwd=gh, env=self.cmake_env) != 0:
            return self._fail("CMake configure failed.")
        self.r.log("Compiling GaussianHierarchyCreator...", "step")
        if self.r.run([cmake_cmd, "--build", "build", "-j", "--config", "Release"],
                      cwd=gh, env=self.cmake_env) != 0:
            return self._fail("CMake build failed.")
        if not hierarchy_creator_path().exists():
            return self._fail(
                f"Build reported success but {hierarchy_creator_path()} is missing.")
        self.r.log("  GaussianHierarchyCreator built.", "ok")
        return True

    def verify(self):
        vpy = venv_python()
        self.r.log("Verifying installation...", "step")
        check = (
            "import torch;"
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available());"
            "import gsplat; print('gsplat ok');"
            "import simple_knn._C; print('simple_knn ok');"
            "from gaussian_hierarchy._C import expand_to_size; print('gaussian_hierarchy ok');"
            "from fused_ssim import fused_ssim; print('fused_ssim ok');"
            "import networkx, sklearn, plyfile, cv2, psutil, torchviz; print('deps ok')"
        )
        rc = self.r.run([str(vpy), "-c", check], cwd=REPO_DIR,
                        env=self.build_env, check=False)
        exe_ok = hierarchy_creator_path().exists()
        self.r.log(f"  GaussianHierarchyCreator: {'found' if exe_ok else 'MISSING'}",
                   "ok" if exe_ok else "error")
        return rc == 0 and exe_ok

    def run_all(self):
        stages = [
            ("Preparing build environment", self.prepare_build_env),
            ("Fetching source", self.clone_repo),
            ("Creating environment", self.create_venv),
            ("Installing PyTorch", self.install_torch),
            ("Installing dependencies", self.install_requirements),
            ("Building CUDA extensions", self.build_extensions),
            ("Applying compatibility patches", self.apply_patches),
            ("Building hierarchy tool", self.build_hierarchy_creator),
            ("Compiling gsplat kernels", self.warm_gsplat),
            ("Verifying", self.verify),
        ]
        for i, (name, fn) in enumerate(stages, 1):
            if self.r.cancelled:
                self.r.log("\nSetup cancelled.", "warn")
                return False
            self.r.log(f"\n=== [{i}/{len(stages)}] {name} ===", "head")
            self.q_progress(i - 1, len(stages))
            if not fn():
                self.r.log(f"\nSetup stopped at: {name}", "error")
                return False
        self.q_progress(len(stages), len(stages))
        return True

    def q_progress(self, done, total):
        self.r.q.put(("progress", (done, total)))


def hierarchy_creator_path():
    gh = REPO_DIR / "submodules" / "gaussianhierarchy" / "build"
    if os.name == "nt":
        return gh / "Release" / "GaussianHierarchyCreator.exe"
    return gh / "GaussianHierarchyCreator"


def existing_scaffold_iterations(dataset, output):
    """Iteration counts of scaffolds already sitting in the output folder."""
    out = Path(output) if output else Path(dataset) / "output"
    pc = out / "scaffold" / "point_cloud"
    if not pc.is_dir():
        return []
    iters = []
    for d in pc.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try:
                iters.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return sorted(iters)


def installation_state():
    """Returns (ready, list of missing pieces)."""
    missing = []
    if not (REPO_DIR / "train.py").exists():
        missing.append("source repository")
    if not venv_python().exists():
        missing.append("virtual environment")
    if not hierarchy_creator_path().exists():
        missing.append("GaussianHierarchyCreator")
    return (not missing), missing


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1080x760")
        self.minsize(940, 640)

        self.log_queue = queue.Queue()
        self.runner = Runner(self.log_queue)
        self.worker = None
        self.busy = False
        self.train_start = None

        self.env_info = {}
        self.vars = {}
        self.dataset_ok = False
        self.image_count = 0
        self.graph_too_small = 0
        self._validate_job = None

        self._build_style()
        self._build_widgets()
        self._probe_environment()
        self._load_settings()
        self._refresh_status()
        self.dataset_var.trace_add("write", lambda *_: self._schedule_validate())
        self.after(60, self._drain_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout ------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Head.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Sub.TLabel", foreground="#555")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 9, "bold"))

    def _build_widgets(self):
        header = ttk.Frame(self, padding=(14, 10, 14, 6))
        header.pack(fill="x")
        ttk.Label(header, text="LoD of Gaussians - Trainer",
                  style="Head.TLabel").pack(anchor="w")
        self.status_label = ttk.Label(header, text="Checking environment...",
                                      style="Sub.TLabel")
        self.status_label.pack(anchor="w")

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=14, pady=(4, 10))

        self.nb = ttk.Notebook(paned)
        paned.add(self.nb, weight=3)

        self._build_setup_tab()
        self._build_train_tab()
        self._build_params_tab()

        # log console
        logframe = ttk.LabelFrame(paned, text="Output", padding=6)
        paned.add(logframe, weight=2)
        self.log = tk.Text(logframe, wrap="none", height=12,
                           background="#101418", foreground="#d8dee9",
                           insertbackground="#d8dee9",
                           font=("Consolas", 9), relief="flat")
        yscroll = ttk.Scrollbar(logframe, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        for tag, colour in (("error", "#ff6b6b"), ("warn", "#ffc857"),
                            ("ok", "#8ecf7a"), ("cmd", "#6fb3d2"),
                            ("head", "#c792ea"), ("step", "#8be9fd"),
                            ("dim", "#7f8c99"), ("out", "#d8dee9"),
                            ("info", "#d8dee9")):
            self.log.tag_configure(tag, foreground=colour)
        self.log.tag_configure("head", font=("Consolas", 9, "bold"))

        bottom = ttk.Frame(self, padding=(14, 0, 14, 12))
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.stop_btn = ttk.Button(bottom, text="Stop", command=self._stop,
                                   state="disabled")
        self.stop_btn.pack(side="right")
        ttk.Button(bottom, text="Clear log",
                   command=lambda: self.log.delete("1.0", "end")).pack(
            side="right", padx=6)

    def _build_setup_tab(self):
        tab = ttk.Frame(self.nb, padding=14)
        self.nb.add(tab, text="  1. Setup  ")

        info = ttk.LabelFrame(tab, text="Detected environment", padding=10,
                              style="Section.TLabelframe")
        info.pack(fill="x")
        self.env_tree = ttk.Treeview(info, columns=("value",), show="tree headings",
                                     height=7)
        self.env_tree.heading("#0", text="Component")
        self.env_tree.heading("value", text="Status")
        self.env_tree.column("#0", width=210, stretch=False)
        self.env_tree.column("value", width=640)
        self.env_tree.pack(fill="x")

        opts = ttk.LabelFrame(tab, text="Build options", padding=10,
                              style="Section.TLabelframe")
        opts.pack(fill="x", pady=(12, 0))

        row = ttk.Frame(opts)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="CUDA toolkit:", width=18).pack(side="left")
        self.cuda_var = tk.StringVar()
        self.cuda_combo = ttk.Combobox(row, textvariable=self.cuda_var,
                                       state="readonly", width=42)
        self.cuda_combo.pack(side="left")
        self.cuda_combo.bind("<<ComboboxSelected>>", lambda e: self._on_cuda_change())
        self.torch_label = ttk.Label(row, text="", style="Sub.TLabel")
        self.torch_label.pack(side="left", padx=10)

        self.allow_unsupported = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts, variable=self.allow_unsupported,
            text="Allow unsupported host compiler (needed when MSVC is newer than CUDA)"
        ).pack(anchor="w", pady=3)

        self.extras_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts, variable=self.extras_var,
            text="Also install optional extras (gradio/timm - only for depth preprocessing)"
        ).pack(anchor="w", pady=3)

        btns = ttk.Frame(tab)
        btns.pack(fill="x", pady=(14, 0))
        self.install_btn = ttk.Button(btns, text="Install / Repair everything",
                                      command=self._start_install)
        self.install_btn.pack(side="left")
        ttk.Button(btns, text="Re-check environment",
                   command=self._recheck).pack(side="left", padx=8)
        ttk.Button(btns, text="Verify installation",
                   command=self._start_verify).pack(side="left")

        ttk.Label(
            tab, style="Sub.TLabel", justify="left",
            text=("Setup clones the repository, creates an isolated Python "
                  "environment, and compiles the CUDA extensions.\n"
                  "Expect 15-40 minutes on first run - the CUDA compiles are the "
                  "slow part. Nothing outside this folder is modified.")
        ).pack(anchor="w", pady=(12, 0))

    def _build_train_tab(self):
        tab = ttk.Frame(self.nb, padding=14)
        self.nb.add(tab, text="  2. Train  ")

        ds = ttk.LabelFrame(tab, text="COLMAP dataset", padding=10,
                            style="Section.TLabelframe")
        ds.pack(fill="x")

        row = ttk.Frame(ds)
        row.pack(fill="x")
        ttk.Label(row, text="Project folder:", width=16).pack(side="left")
        self.dataset_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.dataset_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self._pick_dataset).pack(
            side="left", padx=(8, 0))
        self.prepare_btn = ttk.Button(row, text="Prep",
                                      command=self._prepare_dataset,
                                      state="disabled", width=8)
        self.prepare_btn.pack(side="left", padx=(6, 0))

        self.ds_status = tk.Text(ds, height=7, wrap="word", relief="flat",
                                 background=self.cget("background"),
                                 font=("Segoe UI", 9))
        self.ds_status.pack(fill="x", pady=(8, 0))
        self.ds_status.tag_configure("error", foreground="#c0392b")
        self.ds_status.tag_configure("warn", foreground="#b8860b")
        self.ds_status.tag_configure("ok", foreground="#2d7a2d")
        self.ds_status.configure(state="disabled")

        ttk.Label(ds, style="Sub.TLabel", justify="left",
                  text="Expected layout:   <folder>/sparse/0/{cameras,images,points3D}.bin"
                       "   +   <folder>/images/    (optional: depths/, masks/)"
                  ).pack(anchor="w", pady=(4, 0))

        out = ttk.LabelFrame(tab, text="Output", padding=10,
                             style="Section.TLabelframe")
        out.pack(fill="x", pady=(12, 0))
        row = ttk.Frame(out)
        row.pack(fill="x")
        ttk.Label(row, text="Output folder:", width=16).pack(side="left")
        self.output_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.output_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self._pick_output).pack(
            side="left", padx=(8, 0))
        ttk.Label(out, style="Sub.TLabel",
                  text="Leave empty to use <project folder>/output"
                  ).pack(anchor="w", pady=(4, 0))

        self.skip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            out, variable=self.skip_var,
            text="Resume: skip the coarse pass if a scaffold already exists"
        ).pack(anchor="w", pady=(6, 0))

        pre = ttk.LabelFrame(tab, text="Preset", padding=10,
                             style="Section.TLabelframe")
        pre.pack(fill="x", pady=(12, 0))
        self.preset_var = tk.StringVar(value="Default (paper settings)")
        row = ttk.Frame(pre)
        row.pack(fill="x")
        combo = ttk.Combobox(row, textvariable=self.preset_var, state="readonly",
                             values=list(PRESETS), width=30)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        ttk.Label(row, style="Sub.TLabel",
                  text="   Applies a set of parameters; fine-tune them on the "
                       "Parameters tab.").pack(side="left")

        actions = ttk.Frame(tab)
        actions.pack(fill="x", pady=(16, 0))
        self.train_btn = ttk.Button(actions, text="Start training",
                                    command=self._start_training)
        self.train_btn.pack(side="left")
        ttk.Button(actions, text="Open output folder",
                   command=self._open_output).pack(side="left", padx=8)
        self.elapsed_label = ttk.Label(actions, text="", style="Sub.TLabel")
        self.elapsed_label.pack(side="left", padx=12)

    def _build_params_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  3. Parameters  ")

        canvas = tk.Canvas(tab, highlightthickness=0)
        scroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=14)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        for section, params in PARAM_SPEC:
            box = ttk.LabelFrame(inner, text=section, padding=10,
                                 style="Section.TLabelframe")
            box.pack(fill="x", pady=(0, 12))
            box.columnconfigure(2, weight=1)
            for r, (key, label, kind, helptext) in enumerate(params):
                ttk.Label(box, text=label).grid(row=r, column=0, sticky="w",
                                                padx=(0, 10), pady=3)
                default = DEFAULT_CONFIG.get(key)
                if kind == "bool":
                    var = tk.BooleanVar(value=bool(default))
                    ttk.Checkbutton(box, variable=var).grid(row=r, column=1,
                                                            sticky="w", pady=3)
                elif kind.startswith("choice:"):
                    var = tk.StringVar(value=str(default))
                    ttk.Combobox(box, textvariable=var, state="readonly", width=16,
                                 values=kind.split(":", 1)[1].split(",")).grid(
                        row=r, column=1, sticky="w", pady=3)
                else:
                    var = tk.StringVar(value=self._fmt(default))
                    ttk.Entry(box, textvariable=var, width=18).grid(
                        row=r, column=1, sticky="w", pady=3)
                self.vars[key] = (var, kind)
                if helptext:
                    ttk.Label(box, text=helptext, style="Sub.TLabel",
                              wraplength=560, justify="left").grid(
                        row=r, column=2, sticky="w", padx=(12, 0))

        btns = ttk.Frame(inner)
        btns.pack(fill="x")
        ttk.Button(btns, text="Reset to defaults",
                   command=lambda: self._apply_config(DEFAULT_CONFIG)).pack(side="left")

    @staticmethod
    def _fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return str(value)
            if value and (abs(value) < 1e-4 or abs(value) >= 1e10):
                return f"{value:g}"
            return str(value)
        return str(value)

    # -- environment -------------------------------------------------------
    def _probe_environment(self):
        e = self.env_info
        e["python"] = find_python310()
        e["git"] = shutil.which("git")
        e["gpu"] = detect_gpu()
        e["arch"] = gpu_arch_list()
        e["cuda"] = find_cuda_toolkits()
        e["vcvars"], e["toolset"] = find_vs_build_env()

        rows = []
        rows.append(("Python 3.10/3.11",
                     e["python"] or "NOT FOUND - install Python 3.10 from python.org",
                     bool(e["python"])))
        rows.append(("Git", e["git"] or "NOT FOUND - install Git for Windows",
                     bool(e["git"])))
        rows.append(("GPU", e["gpu"] or "No NVIDIA GPU detected", bool(e["gpu"])))
        if e["cuda"]:
            rows.append(("CUDA toolkit",
                         ", ".join(v for v, _ in e["cuda"]), True))
        else:
            rows.append(("CUDA toolkit",
                         "NOT FOUND - install the NVIDIA CUDA Toolkit (12.6 recommended)",
                         False))
        if e["toolset"]:
            rows.append(("MSVC compiler", f"toolset {e['toolset']}", True))
        else:
            rows.append(("MSVC compiler",
                         "NOT FOUND - install Visual Studio Build Tools with "
                         "'Desktop development with C++'", False))
        ready, missing = installation_state()
        rows.append(("Installation",
                     "Ready" if ready else "Missing: " + ", ".join(missing), ready))
        if " " in str(APP_DIR):
            rows.append(("Install path",
                         f"'{APP_DIR}' contains spaces - the hierarchy build step "
                         f"will fail. Move this folder somewhere without spaces.",
                         False))

        self.env_tree.delete(*self.env_tree.get_children())
        for name, value, ok in rows:
            self.env_tree.insert("", "end", text=name, values=(value,),
                                 tags=("ok" if ok else "bad",))
        self.env_tree.tag_configure("bad", foreground="#c0392b")
        self.env_tree.tag_configure("ok", foreground="#1d6f1d")

        if e["cuda"]:
            self.cuda_combo["values"] = [f"CUDA {v}   ({p})" for v, p in e["cuda"]]
            self.cuda_combo.current(0)
            self._on_cuda_change()
        else:
            self.cuda_combo["values"] = []

    def _selected_cuda(self):
        idx = self.cuda_combo.current()
        cuda = self.env_info.get("cuda") or []
        if 0 <= idx < len(cuda):
            return cuda[idx]
        return (None, None)

    def _on_cuda_change(self):
        version, _ = self._selected_cuda()
        if not version:
            return
        tag = torch_index_for_cuda(version)
        self.torch_label.configure(text=f"-> PyTorch wheel: {tag}")
        need = needs_unsupported_compiler_flag(version, self.env_info.get("toolset"))
        self.allow_unsupported.set(need)

    def _recheck(self):
        self._probe_environment()
        self._refresh_status()
        self._log_line(("info", "Environment re-checked."))

    def _refresh_status(self):
        ready, missing = installation_state()
        e = self.env_info
        if ready:
            self.status_label.configure(
                text=f"Ready.   {e.get('gpu') or 'GPU unknown'}")
            self.train_btn.state(["!disabled"])
            self.install_btn.configure(text="Reinstall / Repair")
        else:
            self.status_label.configure(
                text="Setup required - missing: " + ", ".join(missing))
            self.train_btn.state(["disabled"])

    # -- settings ----------------------------------------------------------
    def _apply_config(self, cfg):
        for key, (var, kind) in self.vars.items():
            if key not in cfg:
                continue
            if kind == "bool":
                var.set(bool(cfg[key]))
            else:
                var.set(self._fmt(cfg[key]))

    def _apply_preset(self):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(PRESETS.get(self.preset_var.get(), {}))
        self._apply_config(cfg)
        self._log_line(("info", f"Applied preset: {self.preset_var.get()}"))

    def collect_config(self):
        """Merge UI values over the defaults; raise ValueError on bad input."""
        cfg = dict(DEFAULT_CONFIG)
        for key, (var, kind) in self.vars.items():
            raw = var.get()
            if kind == "bool":
                cfg[key] = bool(raw)
            elif kind == "int":
                try:
                    cfg[key] = int(float(str(raw).strip()))
                except ValueError:
                    raise ValueError(f"'{key}' must be a whole number (got '{raw}').")
            elif kind == "float":
                try:
                    cfg[key] = float(str(raw).strip())
                except ValueError:
                    raise ValueError(f"'{key}' must be a number (got '{raw}').")
            else:
                cfg[key] = str(raw).strip()
        return cfg

    def _load_settings(self):
        self._apply_config(DEFAULT_CONFIG)
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt settings file is not fatal
            return
        self.dataset_var.set(data.get("dataset", ""))
        self.output_var.set(data.get("output", ""))
        self.preset_var.set(data.get("preset", "Default (paper settings)"))
        self.skip_var.set(data.get("skip_if_exists", True))
        self.extras_var.set(data.get("extras", False))
        self._apply_config(data.get("config", {}))
        if self.dataset_var.get():
            self._validate_dataset()

    def _save_settings(self):
        try:
            cfg = self.collect_config()
        except ValueError:
            cfg = {}
        data = {
            "dataset": self.dataset_var.get(),
            "output": self.output_var.get(),
            "preset": self.preset_var.get(),
            "skip_if_exists": bool(self.skip_var.get()),
            "extras": bool(self.extras_var.get()),
            "config": cfg,
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # -- dataset -----------------------------------------------------------
    def _pick_dataset(self):
        initial = self.dataset_var.get() or str(Path.home())
        folder = filedialog.askdirectory(
            title="Select the folder containing your COLMAP data",
            initialdir=initial)
        if folder:
            self.dataset_var.set(os.path.normpath(folder))
            self._validate_dataset()

    def _pick_output(self):
        folder = filedialog.askdirectory(title="Select an output folder")
        if folder:
            self.output_var.set(os.path.normpath(folder))

    def _schedule_validate(self):
        """Debounce validation so typing a path does not re-scan on every key."""
        if self._validate_job is not None:
            try:
                self.after_cancel(self._validate_job)
            except tk.TclError:
                pass
        self._validate_job = self.after(500, self._validate_dataset)

    def _validate_dataset(self):
        self._validate_job = None
        path = self.dataset_var.get().strip()
        self.ds_status.configure(state="normal")
        self.ds_status.delete("1.0", "end")
        if not path:
            self.ds_status.configure(state="disabled")
            return
        ok, count, msgs, fixable = inspect_dataset(path)
        self.dataset_ok, self.image_count = ok, count
        self.prepare_btn.state(["!disabled" if fixable else "disabled"])

        if ok:
            self.ds_status.insert("end", "Dataset looks valid.\n", "ok")
        for tag, msg in msgs:
            self.ds_status.insert("end", f"  - {msg}\n", tag)
        if fixable:
            self.ds_status.insert(
                "end", "  -> Press 'Prepare dataset' to fix this.\n", "warn")

        # graph_view_select builds a k-NN graph over cameras; sklearn needs
        # strictly more images than neighbours.
        if ok and self.vars["graph_view_select"][0].get():
            k = 100  # the repo overrides view_graph_k with a hardcoded 100
            usable = count
            hold = self.vars["llff_hold"][0].get()
            try:
                hold = int(float(hold))
            except ValueError:
                hold = -1
            if 0 < hold < 1_000_000:
                usable -= count // hold
            if usable <= k:
                self.ds_status.insert(
                    "end",
                    f"  - Only {usable} usable images, but graph view selection "
                    f"needs more than {k}. Starting training will offer to turn "
                    f"it off.\n", "warn")
                self.graph_too_small = usable
            else:
                self.graph_too_small = 0
        else:
            self.graph_too_small = 0
        self.ds_status.configure(state="disabled")

    @staticmethod
    def _prep_plan(path):
        """Human-readable list of what Prep would change, for confirmation."""
        root = Path(path)
        plan = []
        img_dir, images_placed, count = find_images_dir(root)
        if img_dir is not None and count and not images_placed:
            plan.append(f"move {count} images into images/")
        model_dir, model_ok, fmt = find_model_dir(root)
        if model_dir is not None and not model_ok:
            where = ("the project root" if model_dir.resolve() == root.resolve()
                     else "sparse/")
            plan.append(f"move the COLMAP model from {where} into sparse/0/")
        if model_dir is not None:
            summary, needs_conv, blocker = analyse_cameras(model_dir, fmt)
            if needs_conv and not blocker:
                n = sum(summary.values())
                models = "/".join(sorted(summary))
                plan.append(f"rewrite {n} {models} cameras as PINHOLE "
                            f"(lossless - they carry no distortion)")
        return plan

    def _prepare_dataset(self):
        path = self.dataset_var.get().strip()
        if not path:
            return
        plan = self._prep_plan(path)
        if not plan:
            messagebox.showinfo(APP_NAME, "Nothing to fix - this dataset is "
                                          "already in the expected layout.")
            return
        if not messagebox.askyesno(
                APP_NAME,
                "Prep will make these changes inside the dataset folder:\n\n"
                + "\n".join(f"  • {p}" for p in plan)
                + "\n\nFiles are moved, not copied, and the original camera "
                  "file is kept alongside as '.original'.\n\nProceed?"):
            return
        ok, message = prepare_dataset(path, lambda m, t="info": self._log_line((t, m)))
        self._log_line(("ok" if ok else "error", message))
        self._validate_dataset()
        if not ok:
            messagebox.showerror(APP_NAME, message)

    # -- actions -----------------------------------------------------------
    def _set_busy(self, busy, progress_mode="determinate"):
        self.busy = busy
        state = "disabled" if busy else "!disabled"
        for btn in (self.install_btn, self.train_btn):
            btn.state([state])
        self.stop_btn.state(["!disabled" if busy else "disabled"])
        if busy:
            self.progress.configure(mode=progress_mode)
            if progress_mode == "indeterminate":
                self.progress.start(15)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self._refresh_status()

    def _spawn(self, fn, progress_mode="determinate"):
        if self.busy:
            return
        self.runner.cancelled = False
        self._set_busy(True, progress_mode)

        def wrapper():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - surface worker crashes
                self.log_queue.put(("error", f"Unexpected error: {exc}"))
            finally:
                self.log_queue.put(("__done__", None))

        self.worker = threading.Thread(target=wrapper, daemon=True)
        self.worker.start()

    def _installer_opts(self):
        version, root = self._selected_cuda()
        return {
            "python": self.env_info.get("python"),
            "cuda_root": root,
            "cuda_version": version,
            "torch_index": torch_index_for_cuda(version) if version else "cu124",
            "vcvars": self.env_info.get("vcvars"),
            "toolset": self.env_info.get("toolset"),
            "arch": self.env_info.get("arch"),
            "allow_unsupported": bool(self.allow_unsupported.get()),
            "extras": bool(self.extras_var.get()),
        }

    def _start_install(self):
        if not self.env_info.get("python"):
            messagebox.showerror(
                APP_NAME,
                "No suitable Python was found.\n\n"
                "Install Python 3.10 (64-bit) from python.org, then press "
                "'Re-check environment'.")
            return
        if not self.env_info.get("git"):
            messagebox.showerror(APP_NAME, "Git is required. Install Git for Windows "
                                           "and press 'Re-check environment'.")
            return
        if not self.env_info.get("cuda"):
            if not messagebox.askyesno(
                    APP_NAME,
                    "No CUDA toolkit was detected. The CUDA extensions cannot be "
                    "compiled without it.\n\nContinue anyway?"):
                return
        self._save_settings()
        self.nb.select(0)
        opts = self._installer_opts()
        self._log_line(("head", f"\n=== Setting up in {APP_DIR} ==="))

        def job():
            ok = Installer(self.runner, opts).run_all()
            self.log_queue.put(("ok" if ok else "error",
                                "\nSetup complete. Switch to the Train tab."
                                if ok else "\nSetup failed - see the messages above."))

        self._spawn(job)

    def _start_verify(self):
        if not venv_python().exists():
            messagebox.showinfo(APP_NAME, "Nothing installed yet - run setup first.")
            return
        opts = self._installer_opts()

        def job():
            inst = Installer(self.runner, opts)
            inst.prepare_build_env()
            ok = inst.verify()
            self.log_queue.put(("ok" if ok else "error",
                                "\nVerification passed." if ok
                                else "\nVerification failed."))

        self._spawn(job, progress_mode="indeterminate")

    def _start_training(self):
        ready, missing = installation_state()
        if not ready:
            messagebox.showerror(APP_NAME,
                                 "Setup is incomplete: " + ", ".join(missing))
            return
        dataset = self.dataset_var.get().strip()
        if not dataset:
            messagebox.showerror(APP_NAME, "Select the folder containing your "
                                           "COLMAP data first.")
            return
        self._validate_dataset()
        if not self.dataset_ok:
            if not messagebox.askyesno(
                    APP_NAME,
                    "The dataset has problems (see the Train tab).\n\n"
                    "Start training anyway?"):
                return
        try:
            cfg = self.collect_config()
        except ValueError as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return

        # The view graph fits a 100-nearest-neighbour graph over the cameras,
        # so a small capture cannot supply enough points and training would die
        # partway through building it.
        if self.graph_too_small and cfg.get("graph_view_select"):
            choice = messagebox.askyesnocancel(
                APP_NAME,
                f"Graph view selection needs more than 100 usable images, but "
                f"this capture has {self.graph_too_small}.\n\n"
                f"Yes  - turn it off and train normally\n"
                f"No   - leave it on (training will fail while building the graph)\n"
                f"Cancel - go back")
            if choice is None:
                return
            if choice:
                cfg["graph_view_select"] = False
                self.vars["graph_view_select"][0].set(False)
                self._log_line(("warn", "Graph view selection turned off - "
                                        "too few images for a k=100 graph."))

        if cfg["densify_until_iter"] > cfg["iterations"]:
            messagebox.showwarning(
                APP_NAME,
                "'Densify until iter' is larger than the fine iteration count; "
                "densification will simply run to the end.")

        output = self.output_var.get().strip()
        skip = bool(self.skip_var.get())

        # Resuming silently reuses whatever scaffold is on disk. If that
        # scaffold is shorter than the run you just asked for, the coarse stage
        # you configured never happens - easy to miss, expensive to discover.
        if skip:
            iters = existing_scaffold_iterations(dataset, output)
            if iters and max(iters) < cfg["coarse_iterations"]:
                choice = messagebox.askyesnocancel(
                    APP_NAME,
                    f"An existing scaffold trained for {max(iters):,} iterations "
                    f"was found, but you asked for {cfg['coarse_iterations']:,}.\n\n"
                    f"Yes  - reuse the shorter scaffold (fast, lower quality)\n"
                    f"No   - retrain the coarse pass from scratch\n"
                    f"Cancel - go back")
                if choice is None:
                    return
                skip = choice

        self._save_settings()
        self.nb.select(1)
        opts = self._installer_opts()
        self.train_start = time.time()

        def job():
            self._run_training(cfg, dataset, output, skip, opts)

        self._spawn(job, progress_mode="determinate")
        self._tick_elapsed()

    def _run_training(self, cfg, dataset, output, skip, opts):
        r = self.runner
        # train.py opens the config as configs/<name>, relative to its own cwd.
        cfg_name = "lod_trainer.json"
        cfg_path = REPO_DIR / "configs" / cfg_name
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
        r.log(f"Wrote config -> {cfg_path}", "ok")

        inst = Installer(r, opts)
        inst.prepare_build_env()
        env = dict(inst.build_env)
        # gsplat compiles its kernels on first use, so the compiler environment
        # has to be present at run time too, not just during setup.
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [str(venv_python()), "train.py",
               "--project_dir", dataset,
               "--config", cfg_name]
        if output:
            cmd += ["--output_dir", output]
        if skip:
            cmd.append("--skip_if_exists")

        r.log(f"\nTraining started. Total iterations: coarse "
              f"{cfg['coarse_iterations']:,} + fine {cfg['iterations']:,}", "head")
        r.log("This runs for hours. The window stays responsive; press Stop to abort.",
              "dim")

        rc = self.runner.run(cmd, cwd=REPO_DIR, env=env, check=True)
        out_dir = output or str(Path(dataset) / "output")
        if rc == 0:
            r.log(f"\nTraining finished. Results in: {out_dir}", "ok")
            r.log(f"Hierarchy file: {cfg['output_file_name']}", "ok")
        elif r.cancelled:
            r.log("\nTraining stopped by user.", "warn")
        else:
            r.log("\nTraining failed - see the errors above.", "error")

    def _tick_elapsed(self):
        if self.busy and self.train_start:
            secs = int(time.time() - self.train_start)
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            self.elapsed_label.configure(text=f"Elapsed {h:02d}:{m:02d}:{s:02d}")
            self.after(1000, self._tick_elapsed)
        elif not self.busy:
            self.train_start = None

    def _open_output(self):
        target = self.output_var.get().strip() or (
            str(Path(self.dataset_var.get().strip()) / "output")
            if self.dataset_var.get().strip() else "")
        if not target or not Path(target).is_dir():
            messagebox.showinfo(APP_NAME, "No output folder yet.")
            return
        if os.name == "nt":
            os.startfile(target)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", target])

    def _stop(self):
        if not self.busy:
            return
        if messagebox.askyesno(APP_NAME, "Stop the running task?"):
            self._log_line(("warn", "Stopping..."))
            self.runner.stop()

    # -- log plumbing ------------------------------------------------------
    def _log_line(self, item):
        tag, text = item
        self.log.insert("end", str(text) + "\n", tag)
        # Keep the widget bounded; training emits a lot of lines.
        if int(self.log.index("end-1c").split(".")[0]) > 4000:
            self.log.delete("1.0", "1500.0")
        self.log.see("end")

    def _drain_log(self):
        try:
            for _ in range(400):
                tag, payload = self.log_queue.get_nowait()
                if tag == "__done__":
                    self._set_busy(False)
                elif tag == "progress":
                    done, total = payload
                    self.progress.configure(value=done * 100 / max(1, total))
                else:
                    self._log_line((tag, payload))
                    if tag == "out":
                        self._maybe_progress(str(payload))
        except queue.Empty:
            pass
        self.after(60, self._drain_log)

    def _maybe_progress(self, line):
        """Pick training progress out of the tqdm bar."""
        m = re.search(r"(\d+)%\|", line)
        # ttk returns option values as Tcl_Obj, which never equals a str.
        if m and str(self.progress["mode"]) == "determinate":
            try:
                self.progress.configure(value=float(m.group(1)))
            except (ValueError, tk.TclError):
                pass

    def _on_close(self):
        if self.busy:
            if not messagebox.askyesno(
                    APP_NAME, "A task is still running. Stop it and quit?"):
                return
            self.runner.stop()
        self._save_settings()
        self.destroy()


def main():
    if sys.version_info < (3, 8):
        print("This launcher needs Python 3.8 or newer.")
        return 1
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
