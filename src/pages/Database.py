import streamlit as st
import os
import subprocess

st.set_page_config(page_title="Database Management", page_icon="📊")

# Title for the app
st.title("📂 Database Viewer")

# Specify the path to the folder
folder_path = "../Dataset/FaceData/raw"  # Replace with your folder path

# Check if the folder exists
if os.path.exists(folder_path):
    # Get subfolders and their contents
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    # Display summary of the folder
    st.write(f"📁 **Folder Path:** `{folder_path}`")
    st.write(f"📂 **Number of Subfolders:** {len(folders)}")

    # Loop through each folder and display contents
    for folder in folders:
        subfolder_path = os.path.join(folder_path, folder)
        files = os.listdir(subfolder_path)
        num_files = len(files)

        # Display each folder in an expandable section
        with st.expander(f"📂 {folder} ({num_files} files)"):
            # Show the file names in a neat grid
            cols = st.columns(3)  # Adjust the number of columns for layout
            for i, file in enumerate(files):
                file_path = os.path.join(subfolder_path, file)
                # Show file names or preview images (if they're images)
                with cols[i % 3]:  # Distribute files across columns
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        st.image(file_path, caption=file, use_column_width=True)
                    else:
                        st.write(f"📄 {file}")
else:
    st.error(f"❌ Folder `{folder_path}` does not exist.")

# Data preprocessing section with button to execute the command
st.write("### Data Preprocessing: Crop Faces from the Raw Images")

# Button to trigger the preprocessing step
if st.button("Start Face Cropping and Classification"):
    st.write("Processing started... Please wait.")
    
    # Run the shell command using subprocess
    try:
        # First command: Crop faces
        subprocess.run([
            "python", "../src/align_dataset_mtcnn.py", 
            "../Dataset/FaceData/raw", "../Dataset/FaceData/processed", 
            "--image_size", "160", 
            "--margin", "32", 
            "--random_order", 
            "--gpu_memory_fraction", "0.25"
        ], check=True)
        
        # Second command: Train classifier
        subprocess.run([
            "python", "../src/classifier.py", 
            "TRAIN", "../Dataset/FaceData/processed", 
            "../Models/20180402-114759.pb", 
            "../Models/facemodel.pkl", 
            "--batch_size", "1000"
        ], check=True)
        
        st.success("✅ Face cropping and classification completed successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"❌ An error occurred: {e}")