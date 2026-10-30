import sys
import subprocess
import streamlit as st

st.title("Joblib Installation Test")

st.write("Python version:", sys.version)

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "joblib==1.4.2"],
    capture_output=True,
    text=True
)

st.write("Installation output:")
st.code(result.stdout)

st.write("Installation errors:")
st.code(result.stderr)

try:
    import joblib
    st.success("Joblib installed successfully!")
    st.write("Version:", joblib.__version__)
except Exception as e:
    st.error(str(e))
