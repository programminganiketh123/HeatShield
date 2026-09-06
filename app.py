import html
import inspect
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from src.thermal import heat_index_celsius, risk_category, wbgt_proxy
from src.risk_engine import calculate_risk

st.set_page_config(page_title="HEATSHIELD | Hyderabad", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Robust dashboard styling
# -----------------------------
st.markdown(
    """
<style>
:root{--bg:#030b15;--panel:#071625;--panel2:#0b1b2d;--line:#1d3652;--muted:#8ea4bd;--text:#eef6ff;--blue:#309dff;--cyan:#63d5ff;--orange:#ff9b3d;--red:#ff4f43;--green:#35d889;--yellow:#ffd13b;--purple:#9b74ff}
html,body,[class*="css"]{font-family:Inter,Segoe UI,Arial,sans-serif}
header, [data-testid="stHeader"]{height:0!important;min-height:0!important;visibility:hidden!important;display:none!important}
[data-testid="stToolbar"], #MainMenu, footer{display:none!important;visibility:hidden!important}
.stApp{background:radial-gradient(circle at 20% 0%,rgba(33,106,178,.15),transparent 26%),radial-gradient(circle at 95% 2%,rgba(255,82,58,.08),transparent 22%),var(--bg);color:var(--text)}
.block-container{max-width:none!important;padding:.35rem .55rem .5rem .55rem!important;margin:0!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#04101d,#020914);border-right:1px solid rgba(112,145,180,.25)!important;width:220px!important;min-width:220px!important;max-width:220px!important;flex:0 0 220px!important}
section[data-testid="stSidebar"]>div:first-child{width:220px!important}
section[data-testid="stSidebar"]>div{padding:.65rem .65rem 1rem!important}
[data-testid="stSidebar"] .stRadio label{font-size:13px!important}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.42rem!important}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:3px 2px 10px 3px;border-bottom:1px solid rgba(81,119,157,.18);margin-bottom:8px}
.brand{display:flex;align-items:center;gap:11px}.brand-fire{font-size:30px;line-height:1}.brand-title{font-size:28px;font-weight:900;letter-spacing:-.7px}.brand-sub{font-size:12px;color:var(--muted);margin-top:3px}.top-actions{display:flex;align-items:center;gap:12px}.live{padding:6px 10px;border-radius:999px;border:1px solid rgba(53,216,137,.45);background:rgba(53,216,137,.08);color:#70ef9e;font-size:10px;font-weight:900}.updated{font-size:9px;color:#8298b3}.loc{padding:8px 12px;border-radius:9px;border:1px solid rgba(110,149,190,.24);background:rgba(9,25,42,.84);font-size:10px;color:#eef5ff}
.panel{background:linear-gradient(145deg,rgba(7,22,38,.98),rgba(3,12,23,.98));border:1px solid rgba(98,136,175,.32);border-radius:13px;padding:10px;box-shadow:0 8px 24px rgba(0,0,0,.20);box-sizing:border-box}.head{display:flex;align-items:center;gap:8px;font-weight:850;font-size:15px;margin-bottom:8px}.num{width:27px;height:27px;border-radius:8px;background:linear-gradient(145deg,#ff5a36,#d63b2e);display:flex;align-items:center;justify-content:center;font-weight:900}.num.blue{background:linear-gradient(145deg,#6b83f0,#5a45bb)}.num.cyan{background:linear-gradient(145deg,#2f9fff,#176bd0)}.num.green{background:linear-gradient(145deg,#45d47e,#148650)}
.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}.metric{background:rgba(16,37,61,.86);border:1px solid rgba(108,148,195,.2);border-radius:9px;padding:8px;min-height:72px}.mlabel{font-size:8.5px;color:#94aac2}.mvalue{font-size:17px;font-weight:900;margin-top:5px}.mhint{font-size:8px;color:#53b7ff;margin-top:3px}
.risk{margin-top:8px;border:1px solid rgba(255,206,43,.62);border-radius:10px;background:linear-gradient(120deg,rgba(12,31,48,.98),rgba(5,16,28,.98));padding:10px}.riskgrid{display:grid;grid-template-columns:1.15fr .85fr;gap:10px;align-items:center}.eyebrow{font-size:9px;color:#ffd33b;font-weight:900}.score{font-size:36px;line-height:.9;font-weight:950;color:#ffc92e}.score-unit{font-size:15px;color:#8ea4bd}.rlabel{font-size:9px;color:#92a8c0}.rlevel{font-size:17px;font-weight:950;color:#ffd33b}.copy{font-size:9px;color:#b6c5d5;line-height:1.45;margin-top:7px}.tip{font-size:9px;color:#62d9ff;margin-top:6px}.gauge{height:62px;display:flex;align-items:flex-end;justify-content:center;position:relative}.arc{width:112px;height:56px;border-radius:120px 120px 0 0;border:9px solid #38d98a;border-bottom:0;position:relative;filter:drop-shadow(0 0 6px rgba(255,208,51,.20))}.needle{position:absolute;width:3px;height:44px;background:white;bottom:-2px;left:58px;transform-origin:bottom center;transform:rotate(42deg);border-radius:2px}
.section-title{font-size:10px;font-weight:900;margin:10px 0 4px}.forecast{width:100%;height:82px;background:#050e18;border:1px solid rgba(85,120,157,.18);border-radius:7px;overflow:hidden}.forecast svg{width:100%;height:100%}.forecast-meta{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:6px}.meta-box{padding:7px;border-radius:8px;background:rgba(15,35,58,.72);border:1px solid rgba(101,139,181,.16)}.meta-k{font-size:7px;color:#8198b3}.meta-v{font-size:12px;font-weight:900;margin-top:2px}
.selector-label{font-size:8px;color:#8fa5bf;margin-bottom:3px}.map-note{font-size:7.5px;color:#7f96b0;margin-top:5px}.map-wrap{border:1px solid rgba(75,118,157,.28);border-radius:9px;overflow:hidden;background:#08131f}.map-legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}.legend{font-size:7.5px;color:#a4b6c8}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:3px}.hotspot{margin-top:7px;padding:8px;border:1px solid rgba(103,143,184,.18);border-radius:8px;background:rgba(18,38,62,.74)}.hot-top{display:flex;justify-content:space-between;align-items:center}.hot-name{font-size:10px;font-weight:900}.hot-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:5px}.hot-k{font-size:7px;color:#879db7}.hot-v{font-size:10px;font-weight:900;margin-top:2px}
.alert-main{padding:9px;border-radius:9px;background:linear-gradient(90deg,rgba(134,34,29,.50),rgba(66,22,25,.32));border:1px solid rgba(255,79,67,.64)}.alert-title{font-size:12px;font-weight:950}.alert-sub{font-size:8px;color:#d5a3a0;margin-top:3px}.alert-row{display:grid;grid-template-columns:25px 1fr auto;gap:7px;align-items:center;padding:7px 0;border-bottom:1px solid rgba(104,137,172,.13)}.alert-row:last-child{border-bottom:0}.aicon{font-size:17px}.aname{font-size:9px;font-weight:900}.acopy{font-size:7.5px;color:#8ca1ba;line-height:1.35;margin-top:2px}.priority{font-size:7px;font-weight:900;white-space:nowrap}.priority.red{color:#ff7166}.priority.gold{color:#ffc43a}.priority.blue{color:#67baff}.channels{font-size:7.5px;color:#879cb6;margin:5px 0}.action{margin-top:5px;padding:7px;border-radius:8px;border:1px solid rgba(83,135,183,.2);background:rgba(15,34,56,.78);text-align:center;font-size:9px;font-weight:900}
.explain{display:grid;grid-template-columns:1.25fr .75fr;gap:8px}.driver{margin:8px 0}.drow{display:flex;justify-content:space-between;font-size:8px}.drow span:last-child{color:#adc0d2}.bar{height:5px;border-radius:99px;background:#14273d;overflow:hidden;margin-top:4px}.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#2f9fff,#ffd13b,#ff4e43)}.insight{padding:9px;border-radius:8px;border:1px solid rgba(60,214,133,.22);background:rgba(7,32,32,.48)}.it{font-size:10px;color:#67ea98;font-weight:900}.ic{font-size:8px;color:#b0c1d1;line-height:1.48;margin-top:5px}
.prec-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}.prec{min-height:78px;padding:8px;border-radius:8px;background:rgba(17,36,59,.72);border:1px solid rgba(111,149,191,.15)}.pi{font-size:16px}.pt{font-size:8.5px;font-weight:900;margin-top:3px}.pc{font-size:7.4px;color:#8da3bc;line-height:1.32;margin-top:2px}
.water-grid{display:grid;grid-template-columns:.72fr 1.28fr .82fr;gap:7px}.water{padding:9px;border-radius:9px;border:1px solid rgba(55,143,238,.28);background:linear-gradient(145deg,rgba(7,39,70,.94),rgba(6,21,39,.98));min-height:124px}.ring{width:68px;height:68px;border-radius:50%;border:5px solid #2a93ff;display:flex;flex-direction:column;justify-content:center;align-items:center;margin:0 auto 7px}.rn{font-size:20px;font-weight:950;line-height:1}.ru{font-size:7px;color:#90a8c4}.glasses{font-size:16px;letter-spacing:1px}.wtotal{font-size:18px;font-weight:900;color:#66bcff}.small{font-size:7.5px;color:#8ca2bc}.day-card{padding:7px;border:1px solid rgba(103,143,185,.13);border-radius:8px;background:rgba(15,34,55,.62);text-align:center}.day{font-size:7.5px;color:#8ea6c0}.dscore{font-size:12px;font-weight:900;color:#ffd13b;margin-top:2px}.dtemp{font-size:8px;color:#e9f2fc;margin-top:1px}
.footer{padding:7px 2px;color:#66809b;font-size:8px}
.stButton>button{border-radius:8px;min-height:31px}.stTextInput input,.stSelectbox div[data-baseweb="select"]>div,.stNumberInput input{border-radius:8px!important}.stSelectbox label,.stRadio label,.stSlider label,.stToggle label,.stCheckbox label{font-size:8.5px!important}
@media(max-width:1150px){.metric-grid{grid-template-columns:repeat(2,1fr)}.riskgrid,.explain,.water-grid{grid-template-columns:1fr}.prec-grid{grid-template-columns:repeat(2,1fr)}.top-actions{gap:6px}.updated{display:none}}
</style>
""",
    unsafe_allow_html=True,
)

LAT, LON = 17.3850, 78.4867

# Load sample hotspot data from the app folder. If the CSV is accidentally moved,
# keep the dashboard usable with a tiny built-in fallback instead of crashing.
hotspot_path = BASE_DIR / "data" / "sample_hotspots.csv"
if hotspot_path.exists():
    HOTSPOTS = pd.read_csv(hotspot_path)
else:
    HOTSPOTS = pd.DataFrame([
        ["Gachibowli",17.4401,78.3489,31,25.1,67,"MODERATE"],
        ["Madhapur",17.4483,78.3915,38,25.7,70,"MODERATE"],
        ["Kondapur",17.4580,78.3600,35,25.4,69,"MODERATE"],
        ["Kukatpally",17.4849,78.4138,42,25.8,72,"HIGH"],
        ["Secunderabad",17.4399,78.4983,46,26.1,74,"HIGH"],
        ["Begumpet",17.4447,78.4660,39,25.6,71,"MODERATE"],
        ["Charminar",17.3616,78.4747,51,26.4,77,"HIGH"],
        ["Mehdipatnam",17.3940,78.4399,44,26.0,74,"HIGH"],
        ["LB Nagar",17.3499,78.5570,56,26.8,80,"HIGH"],
        ["Uppal",17.4058,78.5591,48,26.2,75,"HIGH"],
        ["HITEC City",17.4483,78.3790,41,25.9,72,"HIGH"],
        ["Shamshabad",17.2403,78.4294,29,24.9,65,"MODERATE"],
    ], columns=["name","lat","lon","risk","temperature","humidity","category"])


RISK_COLORS = {
    "LOW": [53, 216, 137, 230],
    "MODERATE": [255, 209, 59, 235],
    "HIGH": [255, 155, 61, 238],
    "SEVERE": [255, 79, 67, 242],
    "EXTREME": [172, 80, 225, 245],
}


def badge(level):
    c = {"LOW":"#35d889","MODERATE":"#ffd13b","HIGH":"#ff9b3d","SEVERE":"#ff5c52","EXTREME":"#b05de7"}[level]
    return f'<span style="display:inline-block;padding:3px 6px;border-radius:5px;color:{c};background:{c}18;border:1px solid {c}55;font-size:7px;font-weight:900">{level}</span>'


def svg_forecast(times, temps, feels, threshold):
    w, h = 700, 120
    left, right, top, bottom = 30, 14, 8, 30
    vals = np.array(list(temps) + list(feels) + [threshold], dtype=float)
    ymin, ymax = float(vals.min() - 1), float(vals.max() + 2)
    def xy(i, v):
        x = left + (w-left-right) * i / max(1, len(temps)-1)
        y = h-bottom - (h-top-bottom) * (v-ymin) / max(1e-6, ymax-ymin)
        return x,y
    p1 = " ".join(f"{xy(i,temps[i])[0]:.1f},{xy(i,temps[i])[1]:.1f}" for i in range(len(temps)))
    p2 = " ".join(f"{xy(i,feels[i])[0]:.1f},{xy(i,feels[i])[1]:.1f}" for i in range(len(feels)))
    th_y = xy(0, threshold)[1]
    peak_i = int(np.argmax(feels))
    px, py = xy(peak_i, feels[peak_i])
    labels = [(0,times[0]), (len(times)//3,times[len(times)//3]), (len(times)*2//3,times[len(times)*2//3]), (len(times)-1,times[-1])]
    label_svg = "".join(f'<text x="{xy(i, ymin)[0]:.1f}" y="112" fill="#8da2ba" font-size="8" text-anchor="middle">{html.escape(str(t))}</text>' for i,t in labels)
    return f'''<div class="forecast"><svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">
      <line x1="30" x2="686" y1="{xy(0,round((ymin+ymax)/2,1))[1]:.1f}" y2="{xy(0,round((ymin+ymax)/2,1))[1]:.1f}" stroke="#1c3045"/>
      <line x1="30" x2="686" y1="{th_y:.1f}" y2="{th_y:.1f}" stroke="#ff4f43" stroke-dasharray="5 4" opacity=".9"/>
      <polyline points="{p1}" fill="none" stroke="#1e8eff" stroke-width="2.8"/>
      <polyline points="{p2}" fill="none" stroke="#71cdfc" stroke-width="2.2" stroke-dasharray="5 4"/>
      <circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#ff4f43" stroke="#fff" stroke-width="1.2"/>
      <text x="{min(px+8,620):.1f}" y="{max(py-7,14):.1f}" fill="#ff756c" font-size="8" font-weight="700">Peak</text>
      {label_svg}
    </svg></div>'''


def weather_label(code):
    return {0:"Clear",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",51:"Drizzle",53:"Drizzle",61:"Rain",63:"Rain",65:"Heavy rain",80:"Showers",95:"Thunderstorm"}.get(int(code), "Current")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_weather():
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT, "longitude": LON,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code",
            "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,shortwave_radiation,precipitation_probability",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 5, "timezone": "Asia/Kolkata",
        }, timeout=12,
    )
    r.raise_for_status()
    return r.json(), None


def get_weather():
    try:
        return fetch_weather()
    except Exception as exc:
        return None, str(exc)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown('<div style="text-align:center;padding-top:2px"><div style="font-size:48px;line-height:1">🛡️</div><div style="font-size:19px;font-weight:900">HEATSHIELD</div><div style="color:#ff5936;font-size:10px;font-weight:900">SIH26083</div></div>', unsafe_allow_html=True)
    st.markdown('<hr style="border:0;border-top:1px solid rgba(110,145,180,.24);margin:18px 0">', unsafe_allow_html=True)
    st.markdown("**Navigation**")
    page = st.radio("Navigation", ["Dashboard","Analytics","History","Reports","Settings","About"], index=0, label_visibility="collapsed")
    st.markdown('<hr style="border:0;border-top:1px solid rgba(110,145,180,.24);margin:10px 0 12px">', unsafe_allow_html=True)
    st.markdown("**System Status**")
    st.success("● All systems operational")
    st.markdown("**Demo Controls**")
    sim_humidity = st.slider("Humidity %", 30, 95, 68)
    sim_radiation = st.slider("Solar radiation W/m²", 0, 1000, 850, 25)
    sim_wind = st.slider("Wind km/h", 0, 20, 6)
    st.caption("Demo simulator: change humidity, radiation or wind to demonstrate the prototype risk. The main dashboard score stays tied to live weather.")
    if st.button("Refresh live weather", use_container_width=True):
        fetch_weather.clear()
        st.rerun()

weather, weather_error = get_weather()
if weather is None:
    now = datetime.now()
    current = {"temperature_2m": 39.5,"relative_humidity_2m": 68.0,"apparent_temperature":42.0,"wind_speed_10m":6.0,"weather_code":1}
    hourly = {"time": [f"{now.date()} {h:02d}:00" for h in range(24)], "temperature_2m": [39,39,38,38,37,37,37,38,39,40,41,42,42,41,40,39,38,37,36,36,35,35,35,34], "relative_humidity_2m":[68]*24,"apparent_temperature":[40,40,39,39,38,39,40,41,42,43,44,45,45,44,43,42,40,39,38,37,36,36,35,35],"wind_speed_10m":[6]*24,"shortwave_radiation":[0,0,0,0,0,0,30,150,350,600,800,850,900,850,700,500,250,80,0,0,0,0,0,0]}
    daily = {"time":[str(now.date())],"temperature_2m_max":[42],"temperature_2m_min":[34],"precipitation_probability_max":[15]}
    data = {"current":current,"hourly":hourly,"daily":daily}
else:
    data = weather

current = data["current"]
hourly = data["hourly"]
daily = data.get("daily", {})

temp = float(current["temperature_2m"])
rh = float(current["relative_humidity_2m"])
feels = float(current["apparent_temperature"])
wind = float(current["wind_speed_10m"])
rad_now = float(next((x for x in hourly.get("shortwave_radiation", []) if x is not None), 0.0))

baseline = calculate_risk(temp, rh, wind, max(rad_now, 250), 16000, 18, 32, 72)
sim = calculate_risk(temp, sim_humidity, sim_wind, sim_radiation, 16000, 18, 32, 72)
score = float(baseline["overall_risk"])
level = baseline["risk_category"]
demo_delta = float(sim["overall_risk"]) - float(baseline["overall_risk"])
delta_color = "#72e7a1" if demo_delta <= 0 else "#ffb45b"
hi = heat_index_celsius(temp, rh)
wbgt = wbgt_proxy(temp, rh, wind, max(rad_now, 250))

# header
updated = datetime.now().strftime("%I:%M %p • %d %b %Y")
st.markdown(f'''<div class="topbar"><div class="brand"><div class="brand-fire">🔥</div><div><div class="brand-title">HEATSHIELD</div><div class="brand-sub">Hyderabad Heat Intelligence Platform</div></div></div><div class="top-actions"><span class="live">● LIVE</span><span class="updated">Last updated: {updated} ↻</span><span class="loc">⌖ Hyderabad, Telangana ▾</span></div></div>''', unsafe_allow_html=True)

if weather_error:
    st.warning("Live weather connection is temporarily unavailable. Showing a cached/demo-safe fallback for the dashboard.")

if page != "Dashboard":
    st.markdown(f'<div class="panel"><div class="head"><div class="num">{page[0]}</div><div>{page}</div></div>', unsafe_allow_html=True)
    if page == "Analytics":
        st.write("### 5-day outlook")
        if daily.get("time"):
            cards = st.columns(min(5, len(daily["time"])))
            for i, c in enumerate(cards):
                day = pd.to_datetime(daily["time"][i]).strftime("%a %d")
                c.metric(day, f"{daily['temperature_2m_max'][i]:.0f}°C", f"min {daily['temperature_2m_min'][i]:.0f}°C")
        st.write("### Risk drivers")
        st.write(f"Current prototype screening score: **{score:.0f}/100 ({level})**. Humidity **{sim_humidity}%**, radiation **{sim_radiation} W/m²**, wind **{sim_wind} km/h**.")
    elif page == "History":
        hist = pd.DataFrame({"Day":["-4d","-3d","-2d","-1d","Today"],"Risk":[58,61,56,64,round(score)],"Category":[risk_category(x) for x in [58,61,56,64,score]]})
        st.dataframe(hist, use_container_width=True, hide_index=True)
    elif page == "Reports":
        st.write("### Action-ready report")
        st.write(f"Hyderabad • Risk {score:.0f}/100 • {level}\n\nPeak risk should be reviewed against the forecast window. Recommended actions include hydration, shade, work-hour changes and preparedness for vulnerable groups.")
        st.download_button("Export report summary", f"HEATSHIELD Hyderabad\nRisk: {score:.0f}/100\nLevel: {level}\n", file_name="heatshield_report.txt")
    elif page == "Settings":
        st.checkbox("Auto refresh every 60 seconds", value=True)
        st.checkbox("Use fallback if live feed fails", value=True)
        st.selectbox("Risk model", ["Prototype HTSS normalized score", "Validated ML model (future)"])
    else:
        st.markdown("**HEATSHIELD SIH26083** — Impact-based heat-health early warning for Hyderabad.")
        st.write("Forecast → Thermal Stress → Vulnerability → AI Risk → Hyperlocal GIS → Actionable Warning")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Main three-column command center
c1, c2, c3 = st.columns([1.45, 1.45, 1.0])

with c1:
    st.markdown('<div class="panel"><div class="head"><div class="num">1</div><div>Temperature &amp; Heat Risk</div></div>', unsafe_allow_html=True)
    st.markdown(f'''<div class="metric-grid">
      <div class="metric"><div class="mlabel">🌡 Temperature</div><div class="mvalue">{temp:.1f}°C</div><div class="mhint">Live</div></div>
      <div class="metric"><div class="mlabel">💧 Humidity</div><div class="mvalue">{sim_humidity:.0f}%</div><div class="mhint">Heat load</div></div>
      <div class="metric"><div class="mlabel">🌬 Wind Speed</div><div class="mvalue">{sim_wind:.1f} km/h</div><div class="mhint">Cooling</div></div>
      <div class="metric"><div class="mlabel">🌤 Feels Like</div><div class="mvalue">{feels:.1f}°C</div><div class="mhint">{weather_label(current['weather_code'])}</div></div>
    </div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="risk"><div class="riskgrid"><div><div class="eyebrow">AI HEAT RISK INDEX</div><div style="margin-top:6px"><span class="score">{score:.0f}</span> <span class="score-unit">/ 100</span></div></div><div><div class="rlabel">Risk Level</div><div class="rlevel">{level}</div><div class="gauge"><div class="arc"><div class="needle"></div></div></div></div></div><div class="copy">AI assesses current environmental heat-stress potential. Main drivers: humidity, temperature, wind cooling and solar load.</div><div class="tip">✦ Stay hydrated, use shade and schedule cooling breaks.</div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Next 24 Hours Forecast</div>', unsafe_allow_html=True)
    tvals = [float(x) for x in hourly.get("temperature_2m",[])[:24]]
    fvals = [float(x) for x in hourly.get("apparent_temperature",[])[:24]]
    if tvals and fvals:
        ts = pd.to_datetime(hourly["time"][:24]).strftime("%a %H:%M").tolist()
        st.markdown(svg_forecast(ts, tvals, fvals, 40), unsafe_allow_html=True)
    if tvals and fvals:
        peak_i = int(np.argmax(fvals))
        st.markdown(f'''<div class="forecast-meta"><div class="meta-box"><div class="meta-k">Peak Heat Stress Time</div><div class="meta-v">{ts[peak_i]}</div></div><div class="meta-box"><div class="meta-k">Forecast Max</div><div class="meta-v">{max(tvals):.1f}°C</div></div></div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="panel"><div class="head"><div class="num blue">2</div><div>Hyderabad Real-time Map</div></div>', unsafe_allow_html=True)
    tabs = st.radio("Forecast window", ["Today","Tomorrow","+2 Days","+3 Days","+4 Days"], index=0, horizontal=True, label_visibility="collapsed")
    day_shift = {"Today":0,"Tomorrow":1,"+2 Days":2,"+3 Days":3,"+4 Days":4}[tabs]
    frame = HOTSPOTS.copy()
    frame["risk_view"] = np.clip(frame["risk"] + day_shift * 2.0, 0, 100)
    frame["level_view"] = frame["risk_view"].apply(risk_category)

    # Stable PyDeck map: use meter-based radii and a selector for hotspot inspection.
    # This avoids the oversized WebGL circles seen in earlier builds while keeping
    # the map pan/zoom/hover interactive across Streamlit versions.
    layers = []
    for level_name, color in RISK_COLORS.items():
        part = frame[frame["level_view"] == level_name].copy()
        if len(part):
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                id=f"hotspots-{level_name.lower()}",
                data=part,
                pickable=True,
                auto_highlight=True,
                get_position="[lon,lat]",
                get_radius=450,
                radius_units="meters",
                radius_min_pixels=5,
                radius_max_pixels=11,
                get_fill_color=color,
                get_line_color=[245,250,255,220],
                line_width_min_pixels=1,
            ))

    center_df = pd.DataFrame([[LAT, LON, "Hyderabad"]], columns=["lat","lon","name"])
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        id="city-center",
        data=center_df,
        pickable=True,
        get_position="[lon,lat]",
        get_radius=250,
        radius_units="meters",
        radius_min_pixels=4,
        radius_max_pixels=8,
        get_fill_color=[80,200,255,240],
        get_line_color=[240,248,255,255],
        line_width_min_pixels=1,
    ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=17.40, longitude=78.47, zoom=9.75, pitch=0, bearing=0, controller=True
        ),
        map_style=None,
        tooltip={
            "html":"<b>{name}</b><br/>Risk: {risk_view}/100<br/>Temperature: {temp}°C<br/>Humidity: {humidity}%<br/>Status: {level_view}",
            "style":{"backgroundColor":"#071625","color":"#ffffff"},
        },
    )

    # Streamlit/PyDeck API changed over time. Use the stable call first and only
    # enable selection when the installed version explicitly exposes it.
    chart_kwargs = {}
    try:
        params = inspect.signature(st.pydeck_chart).parameters
        if "height" in params:
            chart_kwargs["height"] = 320
        if "width" in params:
            chart_kwargs["width"] = "stretch"
        if "selection_mode" in params and "on_select" in params:
            chart_kwargs["selection_mode"] = "single-object"
            chart_kwargs["on_select"] = "rerun"
            chart_kwargs["key"] = "heatshield-map"
    except Exception:
        pass

    try:
        event = st.pydeck_chart(deck, **chart_kwargs)
    except TypeError:
        # Compatibility fallback for older Streamlit versions.
        event = st.pydeck_chart(deck)

    selected_name = st.session_state.get("selected_hotspot", "LB Nagar")
    # If the installed Streamlit exposes selection events, use them; otherwise the
    # selectbox below remains the reliable interaction mechanism.
    try:
        sel = event.selection.objects if event else {}
        if isinstance(sel, dict):
            for layer_id, items in sel.items():
                if items and layer_id.startswith("hotspots-"):
                    selected_name = items[0].get("name", selected_name)
                    break
    except Exception:
        pass

    options = frame["name"].tolist()
    default_index = options.index(selected_name) if selected_name in options else 0
    chosen = st.selectbox("Inspect hotspot", options, index=default_index, key="hotspot-inspector")
    st.session_state["selected_hotspot"] = chosen
    row = frame[frame["name"] == chosen].iloc[0]

    st.markdown('<div class="map-legend"><span class="legend"><span class="dot" style="background:#35d889"></span>Low 0–20</span><span class="legend"><span class="dot" style="background:#ffd13b"></span>Moderate 20–40</span><span class="legend"><span class="dot" style="background:#ff9b3d"></span>High 40–60</span><span class="legend"><span class="dot" style="background:#ff4f43"></span>Severe 60–80</span><span class="legend"><span class="dot" style="background:#b05de7"></span>Extreme 80–100</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hotspot"><div class="hot-top"><div><span style="color:#ff5147">●</span> <span class="hot-name">{html.escape(str(row["name"]))}</span></div><div>{badge(row["level_view"])}</div></div><div class="hot-grid"><div><div class="hot-k">Risk Score</div><div class="hot-v">{row["risk_view"]:.0f}/100</div></div><div><div class="hot-k">Temperature</div><div class="hot-v">{row["temperature"]:.1f}°C</div></div><div><div class="hot-k">Humidity</div><div class="hot-v">{row["humidity"]:.0f}%</div></div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="map-note">Pan and zoom the map, hover a hotspot for details, or use the hotspot selector to inspect a location. Values are prototype/sample points for the SIH demo, not live ward observations.</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="panel"><div class="head"><div class="num">3</div><div>Alert Centre</div></div>', unsafe_allow_html=True)
    headline = "EXTREME HEAT RISK ALERT" if score >= 60 else ("HIGH HEAT RISK WATCH" if score >= 40 else "MONITOR CONDITIONS")
    st.markdown(f'<div class="alert-main"><div class="alert-title">⚠ {headline}</div><div class="alert-sub">Live city risk: {score:.0f}/100 • Review conditions regularly</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="alert-row"><div class="aicon">👷</div><div><div class="aname">Outdoor Workers</div><div class="acopy">Avoid intensive outdoor work during peak heat. Shift schedules when risk rises.</div></div><div class="priority red">High Priority</div></div><div class="alert-row"><div class="aicon">👥</div><div><div class="aname">Vulnerable Groups</div><div class="acopy">Check elderly people, children and at-risk residents.</div></div><div class="priority gold">Medium Priority</div></div><div class="alert-row"><div class="aicon">🏛</div><div><div class="aname">Authorities</div><div class="acopy">Prepare advisories, water points and cooling support.</div></div><div class="priority blue">Action Required</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="channels">Notification Channels &nbsp; ☐ SMS &nbsp; ✉ Email &nbsp; 🔔 Push &nbsp; ◌ WhatsApp</div>', unsafe_allow_html=True)
    if st.button("View All Alerts →", use_container_width=True):
        st.info("Prototype alert centre: production delivery can connect authorized SMS, WhatsApp or notification services.")
    if st.button("ACTIVATE HEAT ACTION PLAN", use_container_width=True):
        st.success("Demo action plan activated: prioritize cooling centres, drinking-water points, work-hour changes and public advisories.")
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom row
b1, b2, b3 = st.columns([1.15, 1.0, 1.15])
with b1:
    st.markdown('<div class="panel"><div class="head"><div class="num cyan">4</div><div>Why the Risk? <span style="font-size:9px;color:#8ea4bd">ⓘ</span></div></div>', unsafe_allow_html=True)
    drivers = [("💧 Humidity", sim_humidity), ("🌡 Temperature", np.clip((temp-20)/30,0,1)*100), ("☀ Solar Radiation", np.clip(sim_radiation/1000,0,1)*100), ("🌬 Wind Speed", (1-np.clip(sim_wind/20,0,1))*100)]
    driver_html = ''.join(f'<div class="driver"><div class="drow"><span>{n}</span><span>{v:.0f}%</span></div><div class="bar"><div class="fill" style="width:{v:.0f}%"></div></div></div>' for n,v in drivers)
    insight = "High humidity is increasing evaporative difficulty, making it harder for the body to cool down. Lower wind and strong solar load add to thermal stress." if sim_humidity >= 65 else "Humidity is currently lower, so evaporative cooling is less constrained; radiation and temperature remain important drivers."
    st.markdown(f'<div class="explain"><div>{driver_html}</div><div class="insight"><div class="it">🧠 AI Insight</div><div class="ic">{insight}</div></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small" style="margin-top:5px">Established metrics shown separately: Heat Index <b>{hi:.1f}°C</b> • prototype WBGT proxy <b>{wbgt:.1f}°C</b>. These are not an official medical index.</div></div>', unsafe_allow_html=True)

with b2:
    st.markdown('<div class="panel"><div class="head"><div class="num green">5</div><div>Precautions</div></div>', unsafe_allow_html=True)
    st.markdown('''<div class="prec-grid"><div class="prec"><div class="pi">💧</div><div class="pt">Stay Hydrated</div><div class="pc">Drink water regularly even if not thirsty.</div></div><div class="prec"><div class="pi">☀️</div><div class="pt">Avoid Peak Sun</div><div class="pc">Stay indoors between 12 PM – 4 PM.</div></div><div class="prec"><div class="pi">👕</div><div class="pt">Wear Light Clothes</div><div class="pc">Loose, light-coloured and breathable.</div></div><div class="prec"><div class="pi">🌳</div><div class="pt">Use Shade</div><div class="pc">Take breaks in shade or cool places.</div></div><div class="prec"><div class="pi">🧴</div><div class="pt">Sunscreen</div><div class="pc">Use SPF 30+ outdoors.</div></div><div class="prec"><div class="pi">👨‍👩‍👧</div><div class="pt">Check on Others</div><div class="pc">Help elderly, kids and vulnerable people.</div></div></div>''', unsafe_allow_html=True)
    st.button("View Detailed Guidelines →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with b3:
    st.markdown('<div class="panel"><div class="head"><div class="num blue">6</div><div>Smart Water Reminder</div></div>', unsafe_allow_html=True)
    if "water_liters" not in st.session_state: st.session_state.water_liters = 1.5
    if "glasses" not in st.session_state: st.session_state.glasses = 6
    if "water_interval" not in st.session_state: st.session_state.water_interval = 45
    st.session_state.water_interval = st.number_input("Reminder interval (minutes)", 15, 180, st.session_state.water_interval, 15)
    water_progress = min(st.session_state.water_liters/2.5, 1.0)
    wc1,wc2,wc3 = st.columns(3)
    with wc1:
        st.markdown(f'<div class="water"><div class="small" style="text-align:center">Next Reminder In</div><div class="ring"><div class="rn">{st.session_state.water_interval}</div><div class="ru">min</div></div><div class="small" style="text-align:center">Last drink logged: 9:41 AM</div></div>', unsafe_allow_html=True)
    with wc2:
        st.markdown(f'<div class="water"><div class="small">Hydration Progress</div><div class="glasses">🥛🥛🥛🥛🥛◻◻</div><div class="wtotal">{st.session_state.glasses} / 8 <span class="small">Glasses</span></div></div>', unsafe_allow_html=True)
        st.progress(water_progress)
        if st.button("💧 I Drank Water", use_container_width=True):
            st.session_state.water_liters = min(2.5, st.session_state.water_liters + .25)
            st.session_state.glasses = min(8, st.session_state.glasses + 1)
            st.rerun()
    with wc3:
        st.markdown(f'<div class="water"><div class="small">Today\'s Intake</div><div class="wtotal">💧 {st.session_state.water_liters:.2f} L</div><div class="small">of 2.5 L goal</div><div style="margin-top:14px;font-size:23px;font-weight:950;color:#2f9fff;text-align:center">{water_progress*100:.0f}%</div><div class="small" style="text-align:center;margin-top:5px">Daily Goal: 2.5 Liters</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">HeatShield SIH26083 • AI-powered heat-risk intelligence • Live weather input from Open-Meteo • Prototype screening score, not a clinical diagnosis</div>', unsafe_allow_html=True)
