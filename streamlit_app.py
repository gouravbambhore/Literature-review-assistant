import asyncio
import streamlit as st
from litrev_backend import run_litrev

st.set_page_config(page_title="Literature Review Assistant", page_icon="📚")
st.title("📚 Literature Review Assistant")

query = st.text_input("Enter research topic")
n_papers = st.slider("Number of papers", 1, 10, 5)

if st.button("Search") and query:

    async def _runner():
        container = st.container()
        async for frame in run_litrev(query, n_papers):
            role, *rest = frame.split(":", 1)
            content = rest[0].strip()
            with container:
                with st.chat_message("assistant"):
                    st.markdown(f"**{role}**: {content}")

    try:
        asyncio.run(_runner())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_runner())
