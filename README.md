# Urban Land Effect – Urban Heat Island Analysis and Mitigation

This project provides a comprehensive web application for analyzing, predicting, and mitigating Urban Heat Island (UHI) effects in metropolitan areas. Built with Streamlit, it integrates weather data, geospatial analysis, machine learning, and interactive data visualizations to help users detect temperature hotspots, predict future hotspots, evaluate cooling methods, and collect feedback for urban planning.

## Features

- **Hotspot Detection:**  
  Detects urban hotspots using real-time and historical temperature data from Open-Meteo, visualizing them on interactive maps.

- **Hotspot Prediction:**  
  Predicts upcoming heat hotspots using a Gradient Boosting regression model, allowing for proactive mitigation.

- **Analysis:**  
  Provides annual and monthly trends, heat island intensity analysis, and model performance metrics with visualizations.

- **Cooling Methods:**  
  Educates users about various cooling strategies (green roofs, cool pavements, urban forestry, etc.), their effectiveness, and allows users to rate and suggest new methods.

- **Monitoring & Feedback:**  
  Tracks temperature trends, evaluates cooling method effectiveness, and collects community feedback to inform future urban cooling strategies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sri-Krishnan007/Urban-Land-Effect.git
   cd Urban-Land-Effect
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy plotly requests geopy scikit-learn
   ```

3. **Run the app:**
   ```bash
   streamlit run finalpy.py
   ```

## Usage

- Open your browser to the Streamlit app URL provided in your terminal.
- Use the sidebar to navigate between the following sections:
  - **Hotspot Detection:** Enter an address and date range to visualize current hotspots.
  - **Hotspot Prediction:** Predict future hotspots for a given location and date.
  - **Analysis:** Explore trends and performance metrics.
  - **Cooling Methods:** Learn about and rate cooling strategies.
  - **Monitoring:** View historical trends and submit feedback.

## Data Sources

- **Weather Data:** Open-Meteo Archive API
- **Geocoding:** Nominatim (OpenStreetMap)

## Main Dependencies

- Python 3.x
- Streamlit
- pandas, numpy
- plotly
- scikit-learn
- geopy
- requests

## Screenshots

*Add screenshots of the main app pages here.*

## Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss any major changes before submitting a PR.

## License

*Specify your license here (e.g., MIT License).*

## Acknowledgements

- EPA Heat Island Effect ([link](https://www.epa.gov/heatislands))
- Open-Meteo ([link](https://open-meteo.com/))
- Nominatim Geocoding ([link](https://nominatim.openstreetmap.org/))

---

Let me know if you need a more detailed section, a specific license, or example screenshots!
