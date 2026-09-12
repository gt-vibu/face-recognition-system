import os
import uuid

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src import config, database
from src.embeddings import detect_faces

st.set_page_config(page_title="Enroll Person", page_icon="🧑")
st.title("Enroll a Person")

name = st.text_input("Name")
files = st.file_uploader(
    f"Upload {config.MIN_ENROLLMENT_IMAGES}-{config.MAX_ENROLLMENT_IMAGES} clear, "
    "front-facing photos containing ONE person each",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if files:
    if len(files) > config.MAX_ENROLLMENT_IMAGES:
        st.warning(f"Only the first {config.MAX_ENROLLMENT_IMAGES} images will be used.")
        files = files[: config.MAX_ENROLLMENT_IMAGES]

    accepted_embeddings = []
    first_good_image = None
    preview_cols = st.columns(len(files))

    for i, file in enumerate(files):
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        faces = detect_faces(image_bgr)

        with preview_cols[i]:
            st.image(image, use_container_width=True)
            if len(faces) == 0:
                st.error("No face detected")
            elif len(faces) > 1:
                st.error("Multiple faces — use a single-person photo")
            else:
                st.success("Face detected ✓")
                accepted_embeddings.append(faces[0].embedding)
                if first_good_image is None:
                    first_good_image = image

    st.divider()
    st.write(f"**Valid samples:** {len(accepted_embeddings)} / {len(files)}")

    name_clean = name.strip()
    ready = name_clean != "" and len(accepted_embeddings) >= config.MIN_ENROLLMENT_IMAGES

    if not ready:
        if name_clean == "":
            st.info("Enter a name to continue.")
        else:
            st.info(f"Need at least {config.MIN_ENROLLMENT_IMAGES} valid single-face images.")

    if name_clean and database.name_exists(name_clean):
        st.warning("This name is already enrolled. Saving will UPDATE their stored data.")

    if st.button("Save Person", disabled=not ready):
        avg_embedding = np.mean(np.stack(accepted_embeddings), axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)

        thumbnail_path = None
        if first_good_image is not None:
            thumbnail_path = os.path.join(config.THUMB_DIR, f"{uuid.uuid4().hex}.jpg")
            first_good_image.save(thumbnail_path)

        database.add_or_update_person(
            name_clean, avg_embedding, len(accepted_embeddings), thumbnail_path
        )
        st.success(f"'{name_clean}' enrolled successfully with {len(accepted_embeddings)} samples.")
        st.balloons()
