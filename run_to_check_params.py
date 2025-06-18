import open3d as o3d
import numpy as np
import cv2
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="LiDAR-human segmentation using YOLO and projection")
parser.add_argument('--pcd_path', type=str, required=True, help='Path to input PCD file')
parser.add_argument('--rgb_path', type=str, required=True, help='Path to input RGB image')
parser.add_argument('--output_pcd', type=str, default='human_cluster.pcd', help='Output PCD file path')
parser.add_argument('--max_depth', type=float, default=4.0, help='Max depth for point selection (meters)')
parser.add_argument('--box_shift_x', type=int, default=-200, help='Shift YOLO bbox along x (pixels)')
parser.add_argument('--pad', type=int, default=70, help='Padding for YOLO bbox (pixels)')
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.pcd_path)
points = np.asarray(pcd.points)
rgb_img = cv2.imread(args.rgb_path)
img_h, img_w = rgb_img.shape[:2]

Intrinsic Matrix
fx = 1392.99665
fy = 1393.79659
cx = 990.890479
cy = 546.485531

Extrinsic Matrix (LiDAR → Camera)
T = np.array([
    [ 0.99647,  -0.0180473,  0.0819873, -0.278098],
    [-0.0816003, 0.0212275,  0.996439,  -0.316289],
    [-0.0197234, -0.999612,  0.0196799,  0.0227856],
    [0.0,        0.0,        0.0,        1.0]
])

def transform_lidar_to_camera(points):
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])  # [N, 4]
    points_cam = (T @ points_hom.T).T       # [N, 4]
    return points_cam[:, :3]                # Drop homogeneous component


def project_to_image_plane(points_cam):
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    z = np.where(z == 0, 1e-6, z)
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    return np.stack([u, v], axis=1).astype(np.int32)

points_cam = transform_lidar_to_camera(points)
pixel_coords = project_to_image_plane(points_cam)

model = YOLO("yolov8n.pt")
results = model(rgb_img)[0]
boxes = [box for box in results.boxes.data.cpu().numpy() if int(box[5]) == 0]

if not boxes:
    print("No humans detected.")
    exit()


x1, y1, x2, y2 = boxes[0][:4].astype(int)

# Apply padding and shift
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
    print("No 3D points projected inside the bounding box.")
else:
    human_pcd = o3d.geometry.PointCloud()
    human_pcd.points = o3d.utility.Vector3dVector(human_points)
    print(f"Segmented {len(human_points)} human points inside bounding box.")
    o3d.visualization.draw_geometries([human_pcd])
    o3d.io.write_point_cloud(args.output_pcd, human_pcd)

    dbg_img = rgb_img.copy()
    cv2.rectangle(dbg_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("YOLO Human Box", dbg_img)

    while True:
        if cv2.getWindowProperty("YOLO Human Box", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    exit(0)
