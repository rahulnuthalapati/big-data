from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle


app = Flask(__name__)

def process_visualize(files):
    combined_df = preprocess(files)
    graphs = plot_graphs(combined_df)
    results = process2(combined_df)

    return graphs, results


def preprocess(files):
    files_dict = {}
    for file in files:
        files_dict[file.filename] = file

    yield_df = pd.read_csv(files_dict["yield.csv"])
    yield_df = yield_df.rename(columns={'Area':'Country', 'Value': 'Yield Value'})
    yield_df = yield_df.drop(['Domain Code', 'Domain', 'Area Code', 'Element Code', 'Element', 'Item Code', 'Year Code', 'Unit'], axis=1)
    
    rain_df = pd.read_csv(files_dict['rainfall.csv'])
    rain_df = rain_df.rename(columns={' Area':'Country'})
    rain_df = rain_df.rename(columns={'average_rain_fall_mm_per_year':'rainfall_mm'})
    rain_df = rain_df.dropna(subset=['rainfall_mm'])

    temp_df = pd.read_csv(files_dict['temp.csv'])
    temp_df = temp_df.rename(columns={'country':'Country'})
    temp_df = temp_df.rename(columns={'year':'Year'})
    temp_df = temp_df.rename(columns={'avg_temp':'Temp'})
    temp_df = temp_df[temp_df["Temp"] > -5]
    temp_df['Temp'].dropna(inplace=True)

    pest_df = pd.read_csv(files_dict['pesticides.csv'])
    pest_df = pest_df.rename(columns={'Area': 'Country', 'Value': 'PesticideTonnes'})
    pest_df = pest_df.drop(['Domain', 'Element', 'Item','Unit'], axis=1)

    combined_df = yield_df.merge(rain_df, on=['Year','Country']).merge(temp_df, on=['Year','Country']).merge(pest_df, on=['Year','Country'])
    combined_df['rainfall_mm']  = pd.to_numeric(combined_df['rainfall_mm'], errors = 'coerce', downcast = 'float')

    return combined_df  

def process2(combined_df):
    yield_df_1hot = pd.get_dummies(combined_df, columns=['Country', 'Item'])
    yield_df_1hot = yield_df_1hot.dropna()
    label_df = yield_df_1hot['Yield Value']
    feature_df = yield_df_1hot.drop(['Yield Value'], axis=1)

    sc = MinMaxScaler()
    feature_df = pd.DataFrame(sc.fit_transform(feature_df), columns = feature_df.columns)

    X_train, X_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=42)

    rf_regrsr = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regrsr.fit(X_train, y_train)

    y_pred = rf_regrsr.predict(X_test)

    results = {}
    results['Mean Absolute Error'] = mean_absolute_error(y_test, y_pred)
    results['Mean Squared Error'] = mean_squared_error(y_test, y_pred)
    results['R-squared (R2) Score'] = r2_score(y_test, y_pred)

    with open("latest_model.pkl", "wb") as f:
        pickle.dump(rf_regrsr, f)

    return results

def plot_graphs(combined_df):
    graphs = {}
    numerical_col = combined_df.select_dtypes(include = [np.number]).columns
    corrMatrix = combined_df[numerical_col].corr()

    image_dir = os.path.join(os.getcwd(), "static/images/")
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Plotting correlation between columns
    sns.heatmap(corrMatrix, cmap = 'coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.savefig(image_dir + "image1.png")
    graphs["images/image1.png"] = "Correlation between columns"
    plt.close()

    # Plotting scatter graph between Yield Value and Temperature
    x = combined_df['Yield Value']
    y = combined_df['Temp']

    plt.scatter(x,y)
    plt.xlabel('Feature: Temp')
    plt.ylabel('Target: Yield Value')
    plt.title('Scatter Graph')
    plt.savefig(image_dir + "image2.png")
    graphs["images/image2.png"] = "Scatter graph between Yield Value and Temperature"
    plt.close()

    # Plotting scatter graph between Yield Value and Rainfall
    x = combined_df['Yield Value']
    y = combined_df['rainfall_mm']

    plt.scatter(x,y)

    plt.xlabel('Feature: Rain')
    plt.ylabel('Target: Yield Value')
    plt.title('Scatter Graph')
    plt.savefig(image_dir + "image3.png")
    graphs["images/image3.png"] = "Scatter graph between Yield Value and Rainfall"
    plt.close()

    # Plotting scatter graph between Yield Value and Pesticide

    x = combined_df['Yield Value']
    y = combined_df['PesticideTonnes']

    plt.scatter(x,y)

    plt.xlabel('Feature: Pesticide')
    plt.ylabel('Target: Yield Value')
    plt.title('Scatter Graph')
    plt.savefig(image_dir + "image4.png")
    graphs["images/image4.png"] = "Scatter graph between Yield Value and Pesticide"
    plt.close()

    # Plotting scatter graph between Rain fall and Temperature

    x = combined_df['Temp']
    y = combined_df['rainfall_mm']

    plt.scatter(x,y)

    plt.xlabel('Rainfall in mm')
    plt.ylabel('Temp')
    plt.title('Correlation')
    plt.savefig(image_dir + "image5.png")
    graphs["images/image5.png"] = "Scatter graph between Rain fall and Temperature"
    plt.close()

    #Temperature through years

    grouped_df = combined_df.groupby(['Year'])
    mean_temp = grouped_df['Temp'].mean()

    plt.plot(mean_temp.index, mean_temp)

    plt.title('Temperature by year')
    plt.xlabel('Year')
    plt.ylabel('Temperature (degree Celcius)')
    plt.savefig(image_dir + "image6.png")
    graphs["images/image6.png"] = "Temperature through years"
    plt.close()

    #Yield over Years
    grouped_df = combined_df.groupby(['Year'])

    mean_yield_by_year = grouped_df['Yield Value'].mean()

    plt.plot(mean_yield_by_year.index, mean_yield_by_year)

    plt.title('Mean Yield by Year')
    plt.xlabel('Year')
    plt.ylabel('Mean Yield hg/ha')
    plt.savefig(image_dir + "image7.png")
    graphs["images/image7.png"] = "Yield over Years"
    plt.close()

    return graphs  

def process_user_visualize(features, latest_model=False):
    df = create_df(features)
    model = None
    if latest_model:
        model_name = "latest_model.pkl"
    else:
        model_name = "base_model.pkl"
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    yield_value = model.predict(df)
    return yield_value


def create_df(features):
    columns=['Year', 'Yield Value', 'rainfall_mm', 'Temp', 'PesticideTonnes', 
    'Country_Albania', 'Country_Algeria', 'Country_Angola', 'Country_Argentina', 'Country_Armenia', 'Country_Australia', 'Country_Austria', 'Country_Azerbaijan', 'Country_Bahamas', 'Country_Bahrain', 'Country_Bangladesh', 'Country_Belarus', 'Country_Belgium', 'Country_Botswana', 'Country_Brazil', 'Country_Bulgaria', 'Country_Burkina Faso', 'Country_Burundi', 'Country_Cameroon', 'Country_Canada', 'Country_Central African Republic', 'Country_Chile', 'Country_Colombia', 'Country_Croatia', 'Country_Denmark', 'Country_Dominican Republic', 'Country_Ecuador', 'Country_Egypt', 'Country_El Salvador', 'Country_Eritrea', 'Country_Estonia', 'Country_Finland', 'Country_France', 'Country_Germany', 'Country_Ghana', 'Country_Greece', 'Country_Guatemala', 'Country_Guinea', 'Country_Guyana', 'Country_Haiti', 'Country_Honduras', 'Country_Hungary', 'Country_India', 'Country_Indonesia', 'Country_Iraq', 'Country_Ireland', 'Country_Italy', 'Country_Jamaica', 'Country_Japan', 'Country_Kazakhstan', 'Country_Kenya', 'Country_Latvia', 'Country_Lebanon', 'Country_Lesotho', 'Country_Libya', 'Country_Lithuania', 'Country_Madagascar', 'Country_Malawi', 'Country_Malaysia', 'Country_Mali', 'Country_Mauritania', 'Country_Mauritius', 'Country_Mexico', 'Country_Montenegro', 'Country_Morocco', 'Country_Mozambique', 'Country_Namibia', 'Country_Nepal', 'Country_Netherlands', 'Country_New Zealand', 'Country_Nicaragua', 'Country_Niger', 'Country_Norway', 'Country_Pakistan', 'Country_Papua New Guinea', 'Country_Peru', 'Country_Poland', 'Country_Portugal', 'Country_Qatar', 'Country_Romania', 'Country_Rwanda', 'Country_Saudi Arabia', 'Country_Senegal', 'Country_Slovenia', 'Country_South Africa', 'Country_Spain', 'Country_Sri Lanka', 'Country_Sudan', 'Country_Suriname', 'Country_Sweden', 'Country_Switzerland', 'Country_Tajikistan', 'Country_Thailand', 'Country_Tunisia', 'Country_Turkey', 'Country_Uganda', 'Country_Ukraine', 'Country_United Kingdom', 'Country_Uruguay', 'Country_Zambia', 'Country_Zimbabwe',
    'Item_Cassava', 'Item_Maize', 'Item_Plantains and others', 'Item_Potatoes', 'Item_Rice, paddy', 'Item_Sorghum', 'Item_Soybeans', 'Item_Sweet potatoes', 'Item_Wheat', 'Item_Yams']
    df = pd.DataFrame(columns=columns)
    df.loc[0] = [0] * len(columns)
    print(df.head())
    df = df.drop(['Yield Value'], axis=1)
    for feature, value in features.items():
        if feature == "country" or feature == "item":
            df[value] = 1
            continue
        df[feature] = float(value)
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/visualize", methods=["POST"])
def visualize():
    files = request.files.getlist("folder_name")
    graphs, results = process_visualize(files=files)
    return render_template("visualize.html", visualize_output=graphs, model_results = results)

@app.route("/user_visualize", methods=["POST"])
def user_visualize():
    feature_1 = request.form["year"]
    feature_2 = request.form["rainfall"]
    feature_3 = request.form["temp"]
    feature_4 = request.form["pesticides"]
    feature_5 = request.form["country"]
    feature_6 = request.form["item"]
    # latest_model = request.form.get("use_latest_model")
    features = {"Year": feature_1, "rainfall_mm": feature_2, "Temp": feature_3, "PesticideTonnes": feature_4, "country": feature_5, "item": feature_6}
    df = process_user_visualize(features)
    print()
    return render_template("user_visualize.html", features=df)