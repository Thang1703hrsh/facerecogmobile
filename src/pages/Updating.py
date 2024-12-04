import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import shutil
import tempfile

# st.set_page_config(page_title="Updating Data", page_icon="🔄")
st.set_page_config(layout="wide", page_title="Face Recognition App")

# Title and Description
st.title("👤 Face Recognition App")
st.write("Easily manage your face recognition database with adding, deleting, and adjusting options.")

# Sidebar Menu
menu = ["Adding", "Deleting", "Adjusting"]
choice = st.sidebar.radio("📋 Options", menu)

# Folder Path
save_folder = "../Dataset/FaceData/raw"

# Session state to store captured images
if "captured_images" not in st.session_state:
    st.session_state.captured_images = []

# Function to create user folder
def create_user_folder(name):
    user_folder = os.path.join(save_folder, name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

# Adding New Faces
if choice == "Adding":
    st.header("📸 Add New Faces to the Dataset")
    st.info("Provide a name and capture images using webcam or upload files.")

    name = st.text_input("Name", placeholder="Enter name")
    min_images = st.number_input("Minimum Number of Images", min_value=1, step=1)
    upload = st.radio("📤 Upload Image, 📷 Use Webcam, or 🎥 Upload Video", ["Upload Image", "Webcam", "Upload Video"])

    if upload == "Upload Image":
        uploaded_images = st.file_uploader("Upload Images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

        if uploaded_images:
            st.write("Uploaded Images Preview:")
            cols = st.columns(len(uploaded_images))
            for i, img_file in enumerate(uploaded_images):
                with cols[i % len(cols)]:
                    img = Image.open(img_file)
                    st.image(img, use_column_width=True, caption=img_file.name)

            submit_btn = st.button("Submit")
            if submit_btn:
                if name == "":
                    st.error("⚠️ Please enter a name.")
                elif len(uploaded_images) < min_images:
                    st.error(f"⚠️ Upload at least {min_images} images. You uploaded {len(uploaded_images)}.")
                else:
                    user_folder = create_user_folder(name)
                    for img_file in uploaded_images:
                        img = Image.open(img_file)
                        img.save(os.path.join(user_folder, img_file.name))
                    st.success(f"✅ {len(uploaded_images)} images saved to {user_folder}!")

    elif upload == "Webcam":
        img_file_buffer = st.camera_input("Take a Picture")

        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            st.session_state.captured_images.append(cv2_img)
            st.success(f"📸 Picture {len(st.session_state.captured_images)} captured!")

        submit_btn = st.button("Submit")
        if submit_btn:
            if name == "":
                st.error("⚠️ Please enter a name.")
            elif len(st.session_state.captured_images) < int(min_images):  # Ensure min_images is treated as an integer
                st.error(f"⚠️ Capture at least {min_images} images. You captured {len(st.session_state.captured_images)}.")
            else:
                # Create a folder for the user within the save folder
                user_folder = create_user_folder(name)
                for i, img in enumerate(st.session_state.captured_images):
                    file_path = os.path.join(user_folder, f"{name}_{i+1}.jpg")
                    cv2.imwrite(file_path, img)
                st.success(f"✅ {len(st.session_state.captured_images)} images saved to {user_folder}!")

                # Display captured images horizontally
                st.write("Captured Images:")
                cols = st.columns(len(st.session_state.captured_images))  # Create a column for each image
                for col, img in zip(cols, st.session_state.captured_images):
                    col.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

                # Clear session state after submission
                st.session_state.captured_images = []

    # Upload Video Section with Frame Extraction
    elif upload == "Upload Video":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

        if uploaded_video:
            # Show video preview
            st.write("Video Preview:")
            st.video(uploaded_video)

            submit_btn = st.button("Submit Video")
            if submit_btn:
                if name == "":
                    st.error("⚠️ Please enter a name.")
                else:
                    # Create user folder
                    user_folder = create_user_folder(name)

                    # Save the uploaded video to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                        temp_file.write(uploaded_video.getbuffer())
                        temp_video_path = temp_file.name  # Get the path to the temporary video file
                    
                    st.write(f"Processing video: {temp_video_path}")

                    # Open the video using OpenCV
                    video_capture = cv2.VideoCapture(temp_video_path)

                    # Check if the video opened correctly
                    if not video_capture.isOpened():
                        st.error("❌ Failed to open the video file.")
                    else:
                        st.write("Extracting frames from the video...")

                        frame_count = 0
                        while True:
                            ret, frame = video_capture.read()  # Read the next frame
                            if not ret:  # If no frame is returned, end of video
                                break

                            frame_count += 1
                            # Save each frame as an image (e.g., jpg)
                            frame_filename = os.path.join(user_folder, f"frame_{frame_count:04d}.jpg")
                            cv2.imwrite(frame_filename, frame)

                        video_capture.release()  # Release the video capture object

                        # Delete the temporary video file
                        os.remove(temp_video_path)

                        st.success(f"✅ {frame_count} frames saved to {user_folder}!")

# Data Deletion Section
elif choice == "Deleting":
    st.header("🗑️ Delete Faces from the Dataset")
    st.warning("This action is irreversible. Please proceed with caution.")

    # List all folders in the database
    if os.path.exists(save_folder):
        folders = [f for f in os.listdir(save_folder) if os.path.isdir(os.path.join(save_folder, f))]
        
        if len(folders) > 0:
            # Allow the user to select a folder to delete
            folder_to_delete = st.selectbox("Select Folder to Delete", folders)

            # Display the selected folder contents for confirmation
            if folder_to_delete:
                subsave_folder = os.path.join(save_folder, folder_to_delete)
                files = os.listdir(subsave_folder)
                st.write(f"Contents of {folder_to_delete}:")
                for file in files:
                    st.write(f"📄 {file}")

                # Confirm deletion
                confirm_delete = st.button(f"Delete {folder_to_delete}")

                if confirm_delete:
                    try:
                        # Delete the folder and its contents
                        shutil.rmtree(subsave_folder)
                        st.success(f"✅ Successfully deleted the folder: {folder_to_delete}")
                    except Exception as e:
                        st.error(f"❌ Error deleting folder: {e}")
        else:
            st.write("No folders available to delete.")
    else:
        st.error(f"❌ Folder `{save_folder}` does not exist.")

# Adjusting Faces (Editing Data)
elif choice == "Adjusting":
    st.header("⚙️ Adjust Face Data")
    # st.info("Update the name, ID, or image for an existing entry.")

    # List all folders in the database
    if os.path.exists(save_folder):
        folders = [f for f in os.listdir(save_folder) if os.path.isdir(os.path.join(save_folder, f))]

        if len(folders) > 0:
            # Allow the user to select a folder to edit
            folder_to_edit = st.selectbox("Select Folder to Edit", folders)
            
            # Display the selected folder contents for confirmation
            if folder_to_edit:
                user_folder = os.path.join(save_folder, folder_to_edit)
                files = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                st.write(f"Current Images in folder `{folder_to_edit}`:")

                if files:
                    cols = st.columns(3)
                    for i, file in enumerate(files):
                        with cols[i % 3]:
                            img_path = os.path.join(user_folder, file)
                            st.image(img_path, caption=file)
                
                    # Delete Image Section
                    delete_image = st.selectbox("Select Image to Delete", files)
                    if delete_image:
                        delete_btn = st.button(f"Delete {delete_image}")
                        if delete_btn:
                            try:
                                os.remove(os.path.join(user_folder, delete_image))
                                st.success(f"✅ Successfully deleted image: {delete_image}")
                            except Exception as e:
                                st.error(f"❌ Error deleting image: {e}")
                else:
                    st.write("No images available in this folder.")

                # Form to adjust the folder's name/ID or upload a new image
                with st.form("Adjust Form"):
                    new_name = st.text_input("New Folder Name", value=folder_to_edit)
                    new_image = st.file_uploader("Upload New Image", type=['jpg', 'png', 'jpeg'])
                    submit_adjust_btn = st.form_submit_button("Submit Adjustments")

                    if submit_adjust_btn:
                        # Rename folder (if name is changed)
                        if new_name != folder_to_edit:  # Change the folder name
                            new_user_folder = os.path.join(save_folder, new_name)
                            if not os.path.exists(new_user_folder):
                                os.rename(user_folder, new_user_folder)
                                st.success(f"✅ Folder renamed to: {new_name}")
                            else:
                                st.error(f"❌ Folder with name `{new_name}` already exists.")
                        
                        # Upload new image
                        if new_image:
                            image = Image.open(new_image)
                            image_path = os.path.join(user_folder, new_image.name)
                            image.save(image_path)
                            st.success(f"✅ New image saved as: {new_image.name}")
                        
                        st.success("✅ Adjustments saved successfully.")
        else:
            st.write("No folders available to edit.")
    else:
        st.error(f"❌ Folder `{save_folder}` does not exist.")