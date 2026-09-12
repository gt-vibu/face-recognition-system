import streamlit as st

from src import database

st.set_page_config(page_title="Enrolled People", page_icon="📋")
st.title("Enrolled People")

people = database.get_all_persons()

if not people:
    st.info("No people have been enrolled yet.")
    st.page_link("pages/1_Enroll.py", label="Enroll your first person →")
else:
    header = st.columns([1, 3, 2, 3, 1])
    header[0].markdown("**Photo**")
    header[1].markdown("**Name**")
    header[2].markdown("**Samples**")
    header[3].markdown("**Enrolled on**")
    header[4].markdown("**Action**")

    for person_id, name, embedding, num_samples, created_at, thumbnail_path in people:
        cols = st.columns([1, 3, 2, 3, 1])
        if thumbnail_path:
            cols[0].image(thumbnail_path, width=60)
        cols[1].write(name)
        cols[2].write(f"{num_samples}")
        cols[3].write(created_at[:19].replace("T", " "))
        if cols[4].button("Delete", key=f"delete_{person_id}"):
            database.delete_person(person_id)
            st.rerun()
