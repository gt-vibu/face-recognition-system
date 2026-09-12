import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src import config, database
from src.embeddings import detect_faces
from src.matching import MatchStatus, best_match

st.set_page_config(page_title="Identify Person", page_icon="🔍")
st.title("Identify a Face")

if "identify_attempt" not in st.session_state:
    st.session_state.identify_attempt = 0
if "identify_locked" not in st.session_state:
    st.session_state.identify_locked = False
# Streamlit reruns the whole script with the same upload still present, so an
# attempt is only counted once per distinct upload (not once per rerun).
if "identify_last_file_id" not in st.session_state:
    st.session_state.identify_last_file_id = None


def reset_session():
    st.session_state.identify_attempt = 0
    st.session_state.identify_locked = False
    st.session_state.identify_last_file_id = None


if st.session_state.identify_locked:
    st.error("**Identity could not be confidently verified.**")
    st.caption(f"Maximum attempts reached ({config.MAX_IDENTIFICATION_ATTEMPTS}).")
    col1, col2 = st.columns(2)
    if col1.button("Try Again"):
        reset_session()
        st.rerun()
    if col2.button("Go to Enrollment"):
        reset_session()
        st.switch_page("pages/1_Enroll.py")
    st.stop()

st.caption(f"Attempt {st.session_state.identify_attempt + 1} of {config.MAX_IDENTIFICATION_ATTEMPTS}")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces = detect_faces(image_bgr)

    if len(faces) == 0:
        st.error("No face detected. Please upload a clearer image.")
    else:
        enrolled = database.get_person_embeddings_for_matching()
        annotated = image_np.copy()

        # Draw all boxes first so the image renders once with every face marked.
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 2)
        st.image(annotated, use_container_width=True)

        any_uncertain = False

        for idx, face in enumerate(faces):
            result = best_match(face.embedding, enrolled)
            st.markdown(f"### Face {idx + 1}")

            if result.status == MatchStatus.CONFIRMED:
                st.success(f"✓ Identity Confirmed — **{result.person_name}**")
                st.write(f"Similarity: `{result.similarity:.2f}`")

            elif result.status == MatchStatus.UNCERTAIN:
                any_uncertain = True
                st.warning(
                    f"⚠ Identity Not Confirmed — closest candidate **{result.person_name}**\n\n"
                    f"Similarity: `{result.similarity:.2f}` "
                    f"(confirmed threshold: `{config.CONFIRMED_THRESHOLD}`)\n\n"
                    "This result is too close to confidently identify the person. "
                    "This candidate name is shown for context only and is **not** a confirmed match."
                )

            else:  # UNKNOWN
                st.error("✕ No Enrolled Identity Matched")
                st.write(f"Best similarity: `{result.similarity:.2f}`")
                st.caption("This face could not be matched with any enrolled person.")
                if st.button("Enroll This Person Instead", key=f"enroll_prompt_{idx}"):
                    reset_session()
                    st.switch_page("pages/1_Enroll.py")

        is_new_upload = uploaded_file.file_id != st.session_state.identify_last_file_id
        st.session_state.identify_last_file_id = uploaded_file.file_id

        if any_uncertain:
            if is_new_upload:
                st.session_state.identify_attempt += 1
            if st.session_state.identify_attempt >= config.MAX_IDENTIFICATION_ATTEMPTS:
                st.session_state.identify_locked = True
                st.rerun()
            else:
                remaining = config.MAX_IDENTIFICATION_ATTEMPTS - st.session_state.identify_attempt
                st.info(f"Please upload another clear image. ({remaining} attempt(s) remaining.)")
        else:
            st.session_state.identify_attempt = 0
