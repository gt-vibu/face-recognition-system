import streamlit as st

from src import config, database

st.set_page_config(page_title="Face Recognition System", page_icon="🧠", layout="centered")

st.title("Face Recognition Identification System")
st.caption("Local, offline face enrollment and identification — ArcFace embeddings + cosine similarity.")

people = database.get_all_persons()

col1, col2 = st.columns(2)
col1.metric("Enrolled People", len(people))
col2.metric("Confirmed-match threshold", f"{config.CONFIRMED_THRESHOLD:.2f}")

st.divider()

st.subheader("System status")
st.success("Recognition engine ready — all processing happens locally, no external API calls.")

st.subheader("Quick actions")
st.page_link("pages/1_Enroll.py", label="Enroll a Person", icon="➕")
st.page_link("pages/2_Identify.py", label="Identify a Face", icon="🔍")
st.page_link("pages/3_People.py", label="View Enrolled People", icon="📋")

with st.expander("Technical details"):
    st.markdown(
        f"""
- **Detection + embedding model:** InsightFace `{config.MODEL_PACK}` (ArcFace, ONNX Runtime, CPU)
- **Similarity metric:** Cosine similarity (not a calibrated confidence percentage)
- **Confirmed match threshold:** `{config.CONFIRMED_THRESHOLD}`
- **Uncertain match range:** `{config.UNCERTAIN_LOWER_BOUND}` – `{config.CONFIRMED_THRESHOLD}`
- **Max identification attempts (bounded retry):** {config.MAX_IDENTIFICATION_ATTEMPTS}
- **Storage:** SQLite, local file only, never committed to git
        """
    )
