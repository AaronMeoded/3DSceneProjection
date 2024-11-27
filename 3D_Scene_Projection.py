

"""
Steps:
(1) Convert 3d_scan.txt to PLY
(2) Extract projection matrix from txt file
(3) Read 3d banner coordinates from png file using OCR
(4) Matrix multiply facade coordinates by the projection matrix to get the 2d coordinates of banner
(5) Project banner over image given the above 2d coordinates
(6) Find the furthest 3d point from the camera (as a way of finding a 'corner' & isolating one facade)
(7) Convert furthest 3d point to 2d via matrix multiplication, projection matrix
(8) Crop image to only include one facade based on the x-coordinate extracted above
"""

import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
import open3d as o3d


# Step (1): Convert 3d_scan.txt to PLY
def txt_to_ply(path='3d_scan'):
    # Load your .txt file
    df = pd.read_csv(path, header=None)
    df.columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'NX', 'NY', 'NZ']

    # PLY header
    ply_header = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property float nx
    property float ny
    property float nz
    end_header
    '''.format(len(df))

    # Save to PLY file
    with open(str(path + '.ply'), 'w') as file:
        file.write(ply_header)
        df.to_csv(file, sep=' ', index=False, header=False)


# (2) Extract projection matrix from txt file
def get_projection_matrix(path):
    # takes the projection matrix file name, outputs numpy matrix
    try:
        # Read the file and convert the lines into a numpy array
        with open(path, 'r') as file:
            matrix = np.array([list(map(float, line.split())) for line in file.readlines()])

        # Check if the matrix is 3x4
        if matrix.shape != (3, 4):
            raise ValueError("The file does not contain a valid 3x4 projection matrix.")

        return matrix
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# (3) Read 3d banner coordinates from png file using OCR
def get_3d_banner_coordinates(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(gray_image)
    # print(f'OCR text:{text}')

    # Split the text by lines and extract coordinates
    lines = text.split('\n')
    coordinates = []
    # Use regular expression to find all patterns of coordinates
    matches = re.findall(r'\(([^)]+)', text)

    for match in matches:
        try:
            # Split the match by ',' and convert each to float
            coords = [float(x.strip()) for x in match.split(',')]
            if len(coords) == 3:  # Ensure there are 3 components for 3D coordinates
                coordinates.append(coords)
        except ValueError as e:
            # Log the error and skip this match
            print(f"Error converting coordinates: {e}, input text was: {match}")
            continue

    return np.array(coordinates)


# (4): Matrix multiply facade coordinates by the projection matrix to get the 2d coordinates of banner
def multiply_projection_matrix_by_coordinates(projection_matrix, coordinates):
    # Projects 3D points to 2D using a 3x4 projection matrix.

    # print(f'input to np.hstack:{coordinates}')
    # Convert the 3D points to homogeneous coordinates (adding 1 as the fourth component)
    points_3d_homogeneous = np.hstack([coordinates, np.ones((len(coordinates), 1))])

    # Multiply with the projection matrix
    points_2d_homogeneous = points_3d_homogeneous @ projection_matrix.T

    # Convert back to non-homogeneous coordinates (divide by the last component)
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, [2]]

    return points_2d
    # returns [x1, y1], [x2, y2], [x3, y3], [x4, y4]


# (5) Project banner over image given the above 2d coordinates
def project_banner(two_d_coords, facade_path, banner_path):
    # facade_image: The image where the banner will be overlaid.
    # transformed_banner: The image of the banner that has been transformed and is ready to be placed on the facade.
    # two_d_coords: A numpy array of shape (4, 2) containing the x,y coordinates of the corners where the banner will be placed.

    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = two_d_coords
    two_d_coords = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])

    facade_image = cv2.imread(facade_path)
    banner_image = cv2.imread(banner_path)

    # Coordinates on the banner image (source points)
    pts_banner = np.array([[0, 0], [banner_image.shape[1], 0], [banner_image.shape[1], banner_image.shape[0]],
                           [0, banner_image.shape[0]]], dtype="float32")

    # Coordinates on the facade image where the banner will be placed (destination points)
    pts_facade = np.array([two_d_coords], dtype="float32")

    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(pts_banner, pts_facade)

    transformed_banner = cv2.warpPerspective(banner_image, matrix, (facade_image.shape[1], facade_image.shape[0]))

    # Create an empty mask with the same dimensions as the facade image
    mask = np.zeros(facade_image.shape[:2], dtype=np.uint8)

    # Use the 2D coordinates to define the polygonal area for the mask
    # The coordinates need to be in a shape (1, -1, 2)
    pts = np.array([two_d_coords], dtype=np.int32)

    # Fill the polygon defined by the 2D coordinates with white (255)
    cv2.fillPoly(mask, pts, 255)

    # Invert the mask to clear the area where the banner will be placed
    mask_inv = cv2.bitwise_not(mask)

    # Clear the area in the facade image where the banner will be placed
    facade_area_ready = cv2.bitwise_and(facade_image, facade_image, mask=mask_inv)

    # Overlay the banner onto the facade image by using the original mask
    # This assumes that transformed_banner is the same size as the facade image
    banner_area = cv2.bitwise_and(transformed_banner, transformed_banner, mask=mask)

    # Combine the facade image with the banner area
    facade_image = cv2.add(facade_area_ready, banner_area)
    cv2.imwrite('pre-crop.jpg', facade_image)


# retired: failed attempt at finding crop point
def find_most_populated_cell(path):
    point_cloud = o3d.io.read_point_cloud(path)
    point_cloud = np.asarray(point_cloud.points)

    # Ignore the z-axis
    points_2d = point_cloud[:, :2]

    # Find the extreme points: lowest x, highest x, lowest y, highest y
    min_x_point = points_2d[points_2d[:, 0].argmin()]
    max_x_point = points_2d[points_2d[:, 0].argmax()]
    min_y_point = points_2d[points_2d[:, 1].argmin()]
    max_y_point = points_2d[points_2d[:, 1].argmax()]

    # Calculate distances between each pair of the four extreme points
    distances = np.zeros((4, 4))
    extreme_points = [min_x_point, max_x_point, min_y_point, max_y_point]
    print(f'extreme_points:\n{extreme_points}')
    for i in range(4):
        for j in range(4):
             if i < j:  # Since distance is symmetric, no need to calculate twice
                distances[i, j] = np.linalg.norm(extreme_points[i] - extreme_points[j])
    print(f'distances_between_extreme_pts:\n{distances}')

    # Find the indices of the maximum distance
    max_dist_indices = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
    point1, point2 = extreme_points[max_dist_indices[0]], extreme_points[max_dist_indices[1]]

    # Define a line between the two furthest points
    p1, p2 = point1, point2
    print(f'p1,p2:{p1,p2}')
    line_direction = p2 - p1
    line_direction /= np.linalg.norm(line_direction)  # Normalize the line vector

    # Calculate perpendicular distance from each point in the cloud to the line
    # The line equation can be given as p1 + t * line_direction. Any point on the line can be represented for some t.
    # The vector from p1 to a point in points_2d is given by (points_2d - p1).
    # The cross product of two vectors is a vector perpendicular to both and the magnitude is the area of the parallelogram.
    # The distance from the point to the line is the height of the parallelogram (area divided by base, which is line_direction here).
    distances_to_line = np.cross(np.repeat([line_direction], points_2d.shape[0], axis=0), (points_2d - p1))
    distances_to_line = np.abs(distances_to_line) / np.linalg.norm(line_direction)

    # The index of the maximum distance gives the point that is furthest from the line
    furthest_index = np.argmax(distances_to_line)
    intersection_point = points_2d[furthest_index]

    # Determine an arbitrary z-coordinate
    z_value = np.mean(point_cloud[:, 2])  # Or a different method to find a suitable z-coordinate

    # The center point of the most populated cell in 3D
    center_point = np.append(intersection_point, z_value)
    return center_point


# (6) Find the furthest 3d point from the camera (as a way of finding a 'corner' & isolating one facade)
def find_furthest_point_from_camera(path):
    # Load the point cloud from the .ply file
    pcd = o3d.io.read_point_cloud(path)

    # Assume the camera is at the origin (0, 0, 0)
    camera_location = np.array([0, 0, 0])

    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points)

    # Calculate distances from the camera to all points
    distances = np.linalg.norm(points - camera_location, axis=1)

    # Find the index of the furthest point
    furthest_point_index = np.argmax(distances)

    # Retrieve the furthest point coordinates
    furthest_point = points[furthest_point_index]

    return furthest_point


# (8) Crop image to only include one facade based on the x-coordinate extracted above
def crop_unused_facade(path, x_crop_pt, two_d_coords):
    # Load the original image
    image = cv2.imread(path)

    # Determine which side of x_crop_pt to keep based on the banner's position
    banner_xs = [coord[0] for coord in two_d_coords]
    keep_right = x_crop_pt < min(banner_xs)
    x_crop_pt = int(x_crop_pt)

    # Crop the image
    if keep_right:
        # If the banner is to the right, keep the right side
        cropped_image = image[:, x_crop_pt:]
    else:
        # If the banner is to the left, keep the left side
        cropped_image = image[:, :x_crop_pt]

    cv2.imwrite('result.jpg', cropped_image)


def execute_project(scan_path, proj_mat_path, coordinates_path, image_path, banner_path):
    # run all steps
    txt_to_ply(scan_path)  # step (1)
    projection_matrix = get_projection_matrix(proj_mat_path)  # step (2)
    coordinates = get_3d_banner_coordinates(coordinates_path)  # step (3)
    two_d_coords = multiply_projection_matrix_by_coordinates(projection_matrix, coordinates)  # step (4)
    project_banner(two_d_coords, image_path, banner_path)  # step (5)
    # crop image
    crop_pt = find_furthest_point_from_camera(scan_path+'.ply')  # step (6)
    crop_pt = multiply_projection_matrix_by_coordinates(projection_matrix, np.array([crop_pt]))  # step (7)
    x_crop_pt = crop_pt[0][0]
    crop_unused_facade('pre-crop.jpg', x_crop_pt, two_d_coords)  # step (8) [final step]


def main():
    execute_project(scan_path='inputs/3d_scan.txt', proj_mat_path='inputs/projMat.txt',
                    coordinates_path='inputs/coordinated.png', image_path='inputs/image.jpg',
                    banner_path='inputs/banner.jpg')


if __name__ == '__main__':
    main()