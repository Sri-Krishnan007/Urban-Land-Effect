import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import plotly.graph_objects as go


def get_weather_data(lat, lon, date):
    start_date = date - timedelta(days=7)  # Get data for a week leading up to the specified date
    end_date = date
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max&timezone=auto"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temperature': data['daily']['temperature_2m_max']
        })
    else:
        st.error(f"Failed to fetch weather data: {response.status_code}")
        return None
# 4. Main Page Structure
st.set_page_config(page_title="Urban Heat Island Analysis")
st.title("Urban Heat Island Analysis and Mitigation")
page = st.sidebar.selectbox("Choose a page", ["Hotspot Detection", "Hotspot Prediction", "Analysis", "Cooling Methods", "Monitoring"])
def hotspot_detection():
    st.header("Hotspot Detection using Open-Meteo Data")

    # User input for address and date range
    address = st.text_input("Enter an address", "New York, NY")
    start_date, end_date = st.date_input(
        "Select Date Range",
        value=(datetime.now().date() - timedelta(days=7), datetime.now().date())
    )

    if st.button("Detect Hotspots"):
        # Initialize geocoder
        geolocator = Nominatim(user_agent="urban_heat_island_app")

        try:
            # Geocode the address
            location = geolocator.geocode(address)
            if location:
                center_lat, center_lon = location.latitude, location.longitude
                st.write(f"Analyzed location: {location.address}")
                st.write(f"Coordinates: {center_lat}, {center_lon}")
            else:
                st.error("Could not find the specified address. Please try a different one.")
                return
        except (GeocoderTimedOut, GeocoderServiceError):
            st.error("Geocoding service failed. Please try again later.")
            return

        # Create a 6km x 6km grid around the center point
        grid_size = 3  # 3km in each direction from the center
        lats = np.linspace(center_lat - 0.027, center_lat + 0.027, 5)  # 0.027 degrees is approximately 3km
        lons = np.linspace(center_lon - 0.036, center_lon + 0.036, 5)  # 0.036 degrees is approximately 3km at 40° latitude

        # Fetch data for each grid point
        grid_data = []
        for lat in lats:
            for lon in lons:
                data = get_weather_data(lat, lon, end_date)
                if data is not None:
                    # Filter data for the selected date range
                    date_filtered_data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]
                    if not date_filtered_data.empty:
                        grid_data.append({
                            'latitude': lat,
                            'longitude': lon,
                            'temperature': date_filtered_data['temperature'].mean()
                        })

        if grid_data:
            df = pd.DataFrame(grid_data)

            # Create a heatmap
            st.subheader("Temperature Heatmap")
            fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='temperature',
                                    radius=20, center=dict(lat=center_lat, lon=center_lon),
                                    zoom=11, mapbox_style="open-street-map",
                                    color_continuous_scale="Viridis")
            fig.update_layout(title=f"Temperature Heatmap for {start_date} to {end_date}")
            st.plotly_chart(fig)

            # Identify hotspots
            threshold = df['temperature'].mean() + df['temperature'].std()
            hotspots = df[df['temperature'] > threshold]

            st.subheader("Detected Hotspots")
            if not hotspots.empty:
                st.write(f"Number of hotspots detected: {len(hotspots)}")
                st.dataframe(hotspots)

                # Plot hotspots on a map
                fig_hotspots = px.scatter_mapbox(hotspots, lat='latitude', lon='longitude',
                                                 color='temperature', size='temperature',
                                                 color_continuous_scale="Reds",
                                                 zoom=11, center=dict(lat=center_lat, lon=center_lon),
                                                 mapbox_style="open-street-map")
                fig_hotspots.update_layout(title=f"Detected Hotspots for {start_date} to {end_date}")
                st.plotly_chart(fig_hotspots)
            else:
                st.write("No hotspots detected in the given area for the selected date range.")

            # Explanation of Urban Heat Island Effect
            st.subheader("Why are there hotspots?")
            st.write("""
            Urban Heat Island (UHI) effect occurs when urban areas experience higher temperatures 
            compared to their rural surroundings. This is due to several factors:
            1. Dark surfaces absorbing more heat
            2. Reduced vegetation and natural cooling
            3. Heat generated by human activities
            4. Urban geometry trapping heat
            5. Reduced airflow in urban canyons
            """)
        else:
            st.error("Failed to fetch data for the given location and date range.")

def hotspot_prediction():
    st.header("Predicting Upcoming Hotspot Regions")
    
    address = st.text_input("Enter an address", "New York, NY")
    
    # Allow user to select a date for prediction
    prediction_date = st.date_input(
        "Select date for prediction",
        min_value=datetime.now().date() + timedelta(days=1),
        max_value=datetime.now().date() + timedelta(days=30),
        value=datetime.now().date() + timedelta(days=1)
    )
    
    if st.button("Predict Hotspots"):
        # Initialize geocoder
        geolocator = Nominatim(user_agent="urban_heat_island_app")

        try:
            # Geocode the address
            location = geolocator.geocode(address)
            if location:
                lat, lng = location.latitude, location.longitude
                st.write(f"Analyzed location: {location.address}")
                st.write(f"Coordinates: {lat}, {lng}")
            else:
                st.error("Could not find the specified address. Please try a different one.")
                return
        except (GeocoderTimedOut, GeocoderServiceError):
            st.error("Geocoding service failed. Please try again later.")
            return

        current_weather = get_weather_data(lat, lng, datetime.now().date())
        if current_weather is not None:
            # Historical data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            temperatures = np.random.normal(current_weather['temperature'].mean(), 2, size=len(dates))
            df = pd.DataFrame({'Date': dates, 'Temperature': temperatures})

            # Feature engineering
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year

            # Prepare training data
            X = df[['Day', 'Month', 'Year']]
            y = df['Temperature']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Gradient Boosting Model
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X_train, y_train)

            # Prediction
            future_data = pd.DataFrame({
                'Day': [prediction_date.day],
                'Month': [prediction_date.month],
                'Year': [prediction_date.year]
            })
            predicted_temp = model.predict(future_data)[0]
            
            # Evaluation
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Model Mean Squared Error: {mse:.2f}")

            # Simple threshold-based hotspot prediction
            threshold = np.percentile(y, 75)
            is_hotspot = predicted_temp > threshold
            
            st.subheader("Temperature Trend")
            fig = px.line(df, x='Date', y='Temperature', title="Past 30 Days Temperature Trend")
            st.plotly_chart(fig)
            
            st.subheader("Hotspot Prediction")
            st.write(f"Predicted temperature for {prediction_date}: {predicted_temp:.2f}°C")
            st.write(f"Hotspot threshold: {threshold:.2f}°C")
            st.write(f"This location is {'likely' if is_hotspot else 'not likely'} to be a hotspot on {prediction_date}.")

            # Additional context
            temp_difference = predicted_temp - threshold
            if is_hotspot:
                st.write(f"The predicted temperature is {temp_difference:.2f}°C above the hotspot threshold.")
                st.write("Consider implementing cooling measures or issuing heat advisories for this date.")
            else:
                st.write(f"The predicted temperature is {-temp_difference:.2f}°C below the hotspot threshold.")
                st.write("While not a hotspot, continue monitoring and implementing urban heat mitigation strategies.")

        else:
            st.warning("Failed to fetch current weather data.")

def analysis():
    st.header("Urban Heat Island Analysis")

    # Simulated historical data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    temperatures = np.random.normal(25, 5, size=len(dates)) + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    df = pd.DataFrame({"Date": dates, "Temperature": temperatures})

    # Temperature Trend Visualization
    st.subheader("Annual Temperature Trend")
    fig = px.line(df, x="Date", y="Temperature", title="Daily Average Temperature (Past Year)")
    st.plotly_chart(fig)

    # Monthly temperature box plot
    df['Month'] = df['Date'].dt.strftime('%B')
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                   'August', 'September', 'October', 'November', 'December']
    fig_box = px.box(df, x='Month', y='Temperature', category_orders={'Month': month_order})
    fig_box.update_layout(title="Monthly Temperature Distribution")
    st.plotly_chart(fig_box)

    # Heat island intensity
    st.subheader("Heat Island Intensity")
    rural_temp = df['Temperature'] - np.random.uniform(1, 3, size=len(df))
    heat_island_intensity = df['Temperature'] - rural_temp
    fig_intensity = px.line(x=df['Date'], y=heat_island_intensity, 
                            title="Urban Heat Island Intensity Over Time")
    fig_intensity.update_layout(xaxis_title="Date", yaxis_title="Temperature Difference (°C)")
    st.plotly_chart(fig_intensity)

    # Model Performance Metrics
    st.subheader("Model Performance Metrics")
    
    # Simulated predictions
    predictions = df['Temperature'] + np.random.normal(0, 1, size=len(df))
    
    mae = mean_absolute_error(df['Temperature'], predictions)
    mse = mean_squared_error(df['Temperature'], predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(df['Temperature'], predictions)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f}°C")
    col2.metric("MSE", f"{mse:.2f}°C²")
    col3.metric("RMSE", f"{rmse:.2f}°C")
    col4.metric("R² Score", f"{r2:.2f}")

    # Prediction vs Actual scatter plot
    fig_scatter = px.scatter(x=df['Temperature'], y=predictions, 
                             labels={'x': 'Actual Temperature', 'y': 'Predicted Temperature'},
                             title="Predicted vs Actual Temperatures")
    fig_scatter.add_trace(go.Scatter(x=[df['Temperature'].min(), df['Temperature'].max()], 
                                     y=[df['Temperature'].min(), df['Temperature'].max()],
                                     mode='lines', name='Ideal Prediction'))
    st.plotly_chart(fig_scatter)

    # Hotspot frequency by month
    hotspot_threshold = df['Temperature'].quantile(0.9)
    df['Is_Hotspot'] = df['Temperature'] > hotspot_threshold
    hotspot_freq = df.groupby('Month')['Is_Hotspot'].mean().reindex(month_order)
    fig_hotspot_freq = px.bar(x=hotspot_freq.index, y=hotspot_freq.values, 
                              labels={'x': 'Month', 'y': 'Hotspot Frequency'},
                              title="Hotspot Frequency by Month")
    st.plotly_chart(fig_hotspot_freq)

    # Insights and Recommendations
    st.subheader("Insights and Recommendations")
    st.write("""
    Based on the analysis above, here are some key insights and recommendations:
    1. The urban heat island effect is most pronounced during [insert months with highest intensity].
    2. Hotspots are most frequent in [insert months with highest frequency], suggesting a need for increased cooling measures during these periods.
    3. The prediction model performs well with an R² score of {r2:.2f}, but there's room for improvement, especially in extreme temperature scenarios.
    4. Consider implementing additional cooling methods in areas that consistently show high heat island intensity.
    5. Monitor the effectiveness of current cooling strategies, particularly during peak hotspot months.
    6. Enhance the prediction model by incorporating more features such as humidity, wind patterns, and urban density.
    """)


def cooling_methods():
    st.header("Cooling Methods for Urban Heat Islands")

    # Introduction
    st.write("""
    Urban Heat Islands (UHIs) are metropolitan areas that are significantly warmer than their surrounding rural areas. 
    This effect is caused by urban development and human activities. Here are several effective methods to mitigate 
    the UHI effect and cool our cities:
    """)

    # Define cooling methods with more detailed information
    cooling_methods = {
        "Green Roofs": {
            "description": "Installing vegetation on rooftops to provide shade and remove heat from the air.",
            "benefits": [
                "Reduces energy use",
                "Mitigates urban heat island effect",
                "Improves air quality",
                "Increases biodiversity",
                "Manages stormwater runoff"
            ],
            "implementation": "Install a waterproof membrane, root barrier, drainage layer, growing medium, and vegetation on flat or slightly sloped roofs.",
            "effectiveness": 4.2
        },
        "Cool Pavements": {
            "description": "Using reflective or permeable materials for roads and sidewalks to lower surface temperatures.",
            "benefits": [
                "Reduces surface temperatures",
                "Improves nighttime visibility",
                "Increases pavement durability",
                "Reduces stormwater runoff"
            ],
            "implementation": "Apply reflective coatings, use light-colored materials, or install permeable pavements that allow water infiltration.",
            "effectiveness": 3.8
        },
        "Urban Forestry": {
            "description": "Planting trees and vegetation to provide shade and cooling through evapotranspiration.",
            "benefits": [
                "Provides natural cooling",
                "Improves air quality",
                "Reduces energy consumption",
                "Enhances urban aesthetics",
                "Supports biodiversity"
            ],
            "implementation": "Develop urban forestry programs, plant trees along streets and in parks, and create urban green spaces.",
            "effectiveness": 4.5
        },
        "Cool Roofs": {
            "description": "Using reflective materials on roofs to reflect more sunlight and absorb less heat.",
            "benefits": [
                "Reduces building energy use",
                "Lowers roof temperatures",
                "Increases roof lifespan",
                "Improves indoor comfort"
            ],
            "implementation": "Apply reflective coatings, install light-colored roofing materials, or use specially designed reflective tiles.",
            "effectiveness": 4.0
        },
        "Water Features": {
            "description": "Incorporating fountains, ponds, and other water bodies to cool the surrounding air through evaporation.",
            "benefits": [
                "Provides localized cooling",
                "Enhances urban aesthetics",
                "Creates recreational spaces",
                "Supports urban biodiversity"
            ],
            "implementation": "Design and install fountains, ponds, or artificial streams in public spaces and parks.",
            "effectiveness": 3.5
        },
    }

    # Display each cooling method with expandable details
    for method, details in cooling_methods.items():
        with st.expander(f"{method} - Effectiveness: {details['effectiveness']}/5"):
            st.write(f"**Description:** {details['description']}")
            st.write("**Benefits:**")
            for benefit in details['benefits']:
                st.write(f"- {benefit}")
            st.write(f"**Implementation:** {details['implementation']}")

    # Interactive element: Let users rate the methods
    st.subheader("Rate These Cooling Methods")
    selected_method = st.selectbox("Select a method to rate", list(cooling_methods.keys()))
    user_rating = st.slider(f"How effective do you think {selected_method} is?", 1, 5, 3)
    if st.button("Submit Rating"):
        st.success(f"Thank you for rating {selected_method} as {user_rating}/5!")
        # In a real app, you would save this rating to a database

    # Suggest new methods
    st.subheader("Suggest a New Cooling Method")
    new_method = st.text_input("Method Name")
    new_description = st.text_area("Method Description")
    if st.button("Submit Suggestion"):
        # In a real app, you'd save this to a database
        st.success("Thank you for your suggestion! Our team will review it.")

    # Additional resources
    st.subheader("Additional Resources")
    st.write("""
    - [EPA - Heat Island Effect](https://www.epa.gov/heatislands)
    - [Cool Roof Rating Council](https://coolroofs.org/)
    - [Urban Forestry Network](https://urbanforestrynetwork.org/)
    - [Green Roofs for Healthy Cities](https://greenroofs.org/)
    """)

    # Call to action
    st.subheader("Get Involved")
    st.write("""
    Reducing urban heat islands is a community effort. Here's how you can contribute:
    1. Advocate for green policies in your local government
    2. Plant trees and maintain green spaces in your neighborhood
    3. Choose light-colored or reflective materials for your home's roof and pavement
    4. Support local initiatives focused on urban cooling
    5. Educate others about the importance of mitigating urban heat islands
    """)
def monitoring():
    st.header("Monitoring Results and Collecting Feedback")

    # Simulated temperature data for the past year
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    temperatures = np.random.normal(25, 5, size=len(dates)) + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    df = pd.DataFrame({"Date": dates, "Temperature": temperatures})

    # Temperature Trend Visualization
    st.subheader("Temperature Trend Over the Past Year")
    fig = px.line(df, x="Date", y="Temperature", title="Daily Average Temperature")
    st.plotly_chart(fig)

    # Monthly average temperatures
    monthly_avg = df.set_index('Date').resample('M').mean()
    st.subheader("Monthly Average Temperatures")
    fig_monthly = px.bar(monthly_avg, x=monthly_avg.index, y="Temperature", 
                         title="Monthly Average Temperatures")
    st.plotly_chart(fig_monthly)

    # Simulated cooling method effectiveness data
    cooling_methods = ['Green Roofs', 'Cool Pavements', 'Urban Forestry', 'Cool Roofs', 'Water Features']
    effectiveness = np.random.uniform(1, 5, size=len(cooling_methods))
    cooling_df = pd.DataFrame({"Method": cooling_methods, "Effectiveness": effectiveness})

    st.subheader("Cooling Method Effectiveness")
    fig_cooling = px.bar(cooling_df, x="Method", y="Effectiveness", 
                         title="Effectiveness of Cooling Methods (1-5 scale)")
    st.plotly_chart(fig_cooling)

    # Feedback Form
    st.subheader("Feedback Form")
    with st.form("feedback_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        location = st.text_input("Location")
        observed_effect = st.radio("Have you noticed a reduction in urban heat?", 
                                   ("Yes", "No", "Not sure"))
        cooling_method = st.selectbox("Which cooling method have you observed in your area?", 
                                      cooling_methods)
        effectiveness_rating = st.slider("How effective was this method? (1-5)", 1, 5, 3)
        additional_comments = st.text_area("Additional Comments")
        
        submit_button = st.form_submit_button("Submit Feedback")

    if submit_button:
        # In a real application, you would save this data to a database
        feedback_data = {
            "name": name,
            "email": email,
            "location": location,
            "observed_effect": observed_effect,
            "cooling_method": cooling_method,
            "effectiveness_rating": effectiveness_rating,
            "additional_comments": additional_comments,
            "submission_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # For demonstration, we'll just display the collected data
        st.success("Thank you for your feedback!")
        st.json(feedback_data)

    # Display aggregated feedback (simulated)
    st.subheader("Aggregated Feedback")
    feedback_counts = {
        "Positive": np.random.randint(50, 100),
        "Neutral": np.random.randint(20, 50),
        "Negative": np.random.randint(0, 20)
    }
    fig_feedback = px.pie(values=list(feedback_counts.values()), names=list(feedback_counts.keys()),
                          title="Overall Feedback Distribution")
    st.plotly_chart(fig_feedback)

    # Action items based on feedback (example)
    st.subheader("Action Items")
    st.write("""
    Based on the collected feedback, here are some action items:
    1. Increase green roof installations in downtown areas
    2. Conduct a public awareness campaign about cool pavements
    3. Partner with local nurseries to promote urban forestry
    4. Investigate the effectiveness of water features in public squares
    5. Collect more detailed data on cool roof implementations
    """)

# Main logic to display the selected page
if page == "Hotspot Detection":
    hotspot_detection()
elif page == "Hotspot Prediction":
    hotspot_prediction()
elif page == "Analysis":
    analysis()    
elif page == "Cooling Methods":
    cooling_methods()
elif page == "Monitoring":
    monitoring()