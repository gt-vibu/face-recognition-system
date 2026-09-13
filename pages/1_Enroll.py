import os
import uuid

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src import config, database
from src.embeddings import detect_faces
from src.enrollment import embedding_sum, reference_from_sum

st.set_page_config(page_title="Enroll Person", page_icon="🧑")
st.title("Enroll a Person")

name = st.text_input("Name")

# An existing name needs an explicit choice: add photos to their enrollment, or replace it.
ADD, REPLACE = "Add these photos to their enrollment", "Replace their enrollment"
existing_id = database.get_person_id(name.strip()) if name.strip() else None
mode = "new"
if existing_id is not None:
    if existing_id in database.ids_supporting_add_photos():
        choice = st.radio("This person is already enrolled.", [ADD, REPLACE])
        mode = "add" if choice == ADD else "replace"
    else:
        st.info("This person was enrolled before adding photos was supported — saving will replace their enrollment.")
        mode = "replace"
min_photos = 1 if mode == "add" else config.MIN_ENROLLMENT_IMAGES
files = st.file_uploader(
    f"Upload {config.MIN_ENROLLMENT_IMAGES}-{config.MAX_ENROLLMENT_IMAGES} clear, "
    "front-facing photos containing ONE person each",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
st.caption("For best results, add 3 photos with slightly different angles or lighting.")

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
    ready = name_clean != "" and len(accepted_embeddings) >= min_photos

    if not ready:
        if name_clean == "":
            st.info("Enter a name to continue.")
        else:
            st.info(f"Need at least {min_photos} valid single-face image{'s' if min_photos > 1 else ''}.")

    if mode == "replace":
        st.warning("Saving will REPLACE this person's stored reference with one built from these photos only.")

    label = {"new": "Save Person", "add": "Add Photos", "replace": "Replace Enrollment"}[mode]
    if st.button(label, disabled=not ready):
        total = embedding_sum(accepted_embeddings)
        if mode == "add":
            # Exact running update of the stored sum; the photos themselves are not kept.
            count = database.add_photos_to_person(existing_id, total, len(accepted_embeddings))
            st.success(f"Added {len(accepted_embeddings)} photo(s) to '{name_clean}' — {count} samples in total.")
        else:
            thumbnail_path = None
            if first_good_image is not None:
                thumbnail_path = os.path.join(config.THUMB_DIR, f"{uuid.uuid4().hex}.jpg")
                first_good_image.save(thumbnail_path)

            database.add_or_update_person(
                name_clean, reference_from_sum(total), len(accepted_embeddings), thumbnail_path,
                embedding_sum=total,
            )
            verb = "re-enrolled" if mode == "replace" else "enrolled"
            st.success(f"'{name_clean}' {verb} successfully with {len(accepted_embeddings)} samples.")
        st.balloons()
