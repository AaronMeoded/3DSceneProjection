# 3D Scene Projection

## Description
**3D Scene Projection** is a project that integrates advanced 3D processing, computer vision, and image transformation techniques to map real-world 3D data onto 2D images. This project automates the process of isolating building facades, projecting banners or other overlays with precise perspective, and cropping the scene for final visualization.

## Key Features
1. **3D Data Conversion**:
   - Processes raw 3D scan data into usable formats (e.g., PLY).
   - Extracts relevant spatial features for facade isolation.
2. **Projection Transformation**:
   - Maps 3D world coordinates to 2D image space using a calibrated **3x4 projection matrix**.
   - Ensures accurate alignment between the 3D scan and the image.
3. **Banner Placement**:
   - Reads banner coordinates and applies transformations for perspective-correct overlay.
   - Handles real-world camera calibration for accurate placement.
4. **Facade Isolation**:
   - Identifies and isolates a specific facade from the scene based on 3D spatial relationships.
   - Crops the final image to focus on the target area.


## Visualization
3D Facade: <img width="661" alt="Screenshot 2024-01-22 at 7 13 27 PM" src="https://github.com/user-attachments/assets/8a8f1c35-8c99-49cf-baca-bdb8cc8a319d">
2D (image) representation of the facade: ![image](https://github.com/user-attachments/assets/1eb9fe55-b63e-40a3-95be-7b70838a48ee)
Banner to project onto the facade: ![banner](https://github.com/user-attachments/assets/1681fca5-4f68-4048-962a-aca2a90be18b)
### Results
Pre-cropped, and post-cropped images:
![pre-crop](https://github.com/user-attachments/assets/c83686b3-5a87-4cfc-8919-c4435f12b49e)
![result](https://github.com/user-attachments/assets/fa4d6a40-ccc3-4164-838d-0b8c5fd969d3)





## Applications
- **Augmented Reality**: Simulate virtual signage and overlays in real-world environments.
- **Architectural Design**: Visualize proposed changes to buildings or structures.
- **Projection Mapping**: Test and implement virtual content placement on real-world surfaces.

## Getting Started

### Prerequisites
Install the required dependencies:
```bash pip install opencv-python numpy open3d pandas pytesseract```

## Input Files
Place the following files in the `inputs` directory:

- **image.jpg**: Input scene image.
- **3d_scan.txt**: 3D point cloud data with color and normals.
- **projMat.txt**: 3x4 camera projection matrix.
- **coordinates.png**: Banner's 3D coordinates (extracted via OCR).
- **banner.jpg**: Banner to be projected.

---

## Workflow
1. **Convert 3D Scan**: Transform `3d_scan.txt` to PLY for easier handling.
2. **Extract Projection Matrix**: Load the provided 3x4 projection matrix.
3. **Read Banner Coordinates**: Use OCR to extract 3D banner coordinates from `coordinates.png`.
4. **Transform Coordinates**: Map 3D points to 2D using matrix multiplication.
5. **Overlay Banner**: Project the banner onto the image with proper perspective alignment.
6. **Crop Facade**: Identify and crop the relevant facade containing the banner.
7. **Save Final Image**: Output the cropped facade with the projected banner.

---

## Usage
Run the full pipeline with:
```bash python main.py```

