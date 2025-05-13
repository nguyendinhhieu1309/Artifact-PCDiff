import numpy as np
import open3d as o3d
import argparse
import os

def visualize_point_cloud(npy_file_path):
    """
    Loads and visualizes a point cloud from an .npy file.
    """
    if not os.path.exists(npy_file_path):
        print(f"Error: File not found at {npy_file_path}")
        return

    # Load the point cloud data from the .npy file
    # Expected shape: (batch_size, num_points, 3) or (num_points, 3)
    try:
        point_cloud_data = np.load(npy_file_path)
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    points_to_visualize = None
    # If the data has a batch dimension, take the first sample
    if point_cloud_data.ndim == 3:
        print(f"Data has shape {point_cloud_data.shape}. Visualizing the first sample from the batch.")
        points_to_visualize = point_cloud_data[0]
    elif point_cloud_data.ndim == 2 and point_cloud_data.shape[1] == 3:
        print(f"Data has shape {point_cloud_data.shape}. Visualizing.")
        points_to_visualize = point_cloud_data
    else:
        print(f"Error: Unexpected data shape {point_cloud_data.shape}. Expected (batch_size, N, 3) or (N, 3).")
        return

    if points_to_visualize.shape[0] == 0:
        print("No points to visualize.")
        return

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_to_visualize)

    # Add some color for better visibility if there isn't any
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray color

    # Save the point cloud as a .ply file
    base_name = os.path.splitext(npy_file_path)[0]
    ply_file_path = base_name + ".ply"
    try:
        o3d.io.write_point_cloud(ply_file_path, pcd)
        print(f"Successfully saved point cloud to: {ply_file_path}")
    except Exception as e:
        print(f"Error saving .ply file: {e}")

    # Attempt to visualize the point cloud interactively
    print(f"Attempting to visualize point cloud from: {npy_file_path}")
    try:
        o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud: {os.path.basename(npy_file_path)}")
    except Exception as e:
        print(f"Failed to open interactive visualization window: {e}")
        print("This is common in headless environments. You can view the saved .ply file instead.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point cloud from an .npy file.")
    parser.add_argument("npy_file", type=str, help="Path to the .npy file containing the point cloud.")
    args = parser.parse_args()

    visualize_point_cloud(args.npy_file)