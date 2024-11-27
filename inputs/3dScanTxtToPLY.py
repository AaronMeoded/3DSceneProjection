import pandas as pd

# Load your .txt file
df = pd.read_csv('inputs/3d_scan.txt', header=None)
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
with open('output_file.ply', 'w') as file:
    file.write(ply_header)
    df.to_csv(file, sep=' ', index=False, header=False)
