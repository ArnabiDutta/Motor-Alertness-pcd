import open3d as o3d
import numpy as np
import cv2
from ultralytics import YOLO
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser(description="Batch LiDAR-human segmentation using YOLO and projection")
parser.add_argument('--ply_dir', type=str, required=True, help='Directory containing PLY frames')
parser.add_argument('--rgb_dir', type=str, required=True, help='Directory containing RGB frames')
parser.add_argument('--out_dir', type=str, required=True, help='Directory to save output human point clouds')
parser.add_argument('--max_depth', type=float, default=4.0, help='Max depth for point selection (meters)')
parser.add_argument('--box_shift_x', type=int, default=-200, help='Shift YOLO bbox along x (pixels)')
parser.add_argument('--pad', type=int, default=70, help='Padding for YOLO bbox (pixels)')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

fx = 1392.99665
fy = 1393.79659
cx = 990.890479
cy = 546.485531

T = np.array([
    [ 0.99647,  -0.0180473,  0.0819873, -0.278098],
    [-0.0816003, 0.0212275,  0.996439,  -0.316289],
    [-0.0197234, -0.999612,  0.0196799,  0.0227856],
    [0.0,        0.0,        0.0,        1.0]
])

def transform_lidar_to_camera(points):
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])
    points_cam = (T @ points_hom.T).T
    return points_cam[:, :3]

def project_to_image_plane(points_cam):
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    z = np.where(z == 0, 1e-6, z)
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    return np.stack([u, v], axis=1).astype(np.int32)

model = YOLO("yolov8n.pt")

for ply_path in sorted(glob(os.path.join(args.ply_dir, "frame_*.ply"))):
    frame_id = os.path.splitext(os.path.basename(ply_path))[0].split("_")[1]
    rgb_path = os.path.join(args.rgb_dir, f"frame_{frame_id}.png")

    if not os.path.exists(rgb_path):
        print(f"Skipping {frame_id}: RGB file not found.")
        continue

    print(f"Processing frame {frame_id}...")

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    rgb_img = cv2.imread(rgb_path)
    img_h, img_w = rgb_img.shape[:2]

    points_cam = transform_lidar_to_camera(points)
    pixel_coords = project_to_image_plane(points_cam)

    results = model(rgb_img)[0]
    boxes = [box for box in results.boxes.data.cpu().numpy() if int(box[5]) == 0]

    if not boxes:
        print(f"No humans detected in frame {frame_id}.")
        continue

    x1, y1, x2, y2 = boxes[0][:4].astype(int)
    x1 = np.clip(x1 - args.pad + args.box_shift_x, 0, img_w - 1)
    y1 = np.clip(y1 - args.pad, 0, img_h - 1)
    x2 = np.clip(x2 + args.pad + args.box_shift_x, 0, img_w - 1)
    y2 = np.clip(y2 + args.pad, 0, img_h - 1)

    in_bbox_mask = (
        (pixel_coords[:, 0] >= x1) & (pixel_coords[:, 0] <= x2) &
        (pixel_coords[:, 1] >= y1) & (pixel_coords[:, 1] <= y2)
    )
    masked_lidar_points = points[in_bbox_mask]

    depths = np.linalg.norm(masked_lidar_points, axis=1)
    depth_mask = depths < args.max_depth
    human_points = masked_lidar_points[depth_mask]

    if len(human_points) == 0:
        print(f"No 3D human points in frame {frame_id}.")
        continue

    human_pcd = o3d.geometry.PointCloud()
    human_pcd.points = o3d.utility.Vector3dVector(human_points)
    pcd_out_path = os.path.join(args.out_dir, f"frame_{frame_id}_human.pcd")
    o3d.io.write_point_cloud(pcd_out_path, human_pcd)

    # Optional debug image
    # dbg_img = rgb_img.copy()
    # cv2.rectangle(dbg_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # img_out_path = os.path.join(args.out_dir, f"frame_{frame_id}_bbox.png")
    # cv2.imwrite(img_out_path, dbg_img)

    print(f"Saved: {pcd_out_path}")
