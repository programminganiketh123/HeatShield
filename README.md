# HEATSHIELD SIH26083 V13

Self-contained Streamlit prototype for Hyderabad heat-risk intelligence.

## Run in VS Code PowerShell

Open THIS folder (the folder containing `app.py`). Then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Or double-click `run_heatshield.bat`.

## Important

Do not open a parent folder and expect `data/` to be found elsewhere. V13 uses an absolute path based on the location of `app.py`, and the required `data/sample_hotspots.csv` is included in this package.

The hotspot CSV contains prototype/demo values. Replace with validated Hyderabad ward data before claiming live hyperlocal measurements.
