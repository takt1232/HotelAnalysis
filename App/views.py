import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from .forms import CSVUploadForm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings('ignore')

csv_path = None

def visualize_all_columns(df):
    # Create a visualization for each column
    visualizations = []
    for column in df.columns:
        # Check if the column is numerical
        if df[column].dtype in ['int64', 'float64']:
            plt.figure(figsize=(10, 6))
            if len(df[column].unique()) < 10:
                sns.countplot(x=column, data=df)
            else:
                sns.histplot(x=column, data=df, kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.tight_layout()
            # Save the plot to a temporary file
            plt_tmp = os.path.join('visualization', f'{column}_plot_tmp.png')
            plt.savefig(os.path.join(settings.STATICFILES_DIRS[0], plt_tmp))
            visualizations.append(plt_tmp)
            plt.close()
    return visualizations

def process_csv(request):
    plt_tmp = None  # Define plt_tmp with a default value
    global csv_path

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            print("CSV file received:", csv_file.name)

            # Save the CSV file to a specific directory
            csv_file_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_csv', csv_file.name)
            with open(csv_file_path, 'wb') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)

            csv_path = csv_file_path

            # Read the CSV file using pandas
            try:
                df = pd.read_csv(csv_file_path)

                # Description of the data
                description = {
                    'shape': df.shape,
                    'attributes': list(df.columns),
                    'data_types': df.dtypes,
                    'null_values': df.isnull().sum(),
                    'duplicates': df.duplicated().sum(),
                    'statistics': df.describe()
                }

                # Display only the first few rows
                df_display = df.head()

                # Generate visualizations for all columns
                visualizations = visualize_all_columns(df)

                # Render the home template with the form and processed data
                return render(request, 'home.html', {'form': form, 'csv_description': description, 'csv_data': df_display, 'visualizations': visualizations})

            except Exception as e:
                print("Error reading CSV:", str(e))
                description = "Error reading CSV file."
                df_display = None

                # Render the home template with the form and error message
                return render(request, 'home.html', {'form': form, 'csv_description': description, 'csv_data': df_display})
            
    elif request.method == 'GET':
        action = request.GET.get('action')
        column = request.GET.get('column')

        if action == "drop":
            # Placeholder logic to remove missing values
            try:
                # Load CSV file and remove missing values
                if csv_path:
                    df = pd.read_csv(csv_path)
                    
                    df.drop(column, axis =1, inplace=True)

                    # Overwrite the existing CSV file with the cleaned DataFrame
                    df.to_csv(csv_path, index=False)

                    description = {
                        'shape': df.shape,
                        'attributes': list(df.columns),
                        'data_types': df.dtypes,
                        'null_values': df.isnull().sum(),
                        'duplicates': df.duplicated().sum(),
                        'statistics': df.describe()
                    }

                    # Display only the first few rows
                    df_display = df.head()

                    # Generate a simple visualization using Matplotlib
                    plt.figure(figsize=(10, 6))
                    sns.countplot(x='is_canceled', data=df)  # Replace 'your_column' with the column you want to visualize
                    plt.title('is_canceled')
                    plt.xlabel('X-axis Label')
                    plt.ylabel('Y-axis Label')
                    plt.tight_layout()

                    # Save the plot to a temporary file
                    plt_tmp = os.path.join('visualization', 'plot_tmp.png')
                    plt.savefig(os.path.join(settings.STATICFILES_DIRS[0], plt_tmp))

                    # Render the cleaned data
                    return render(request, 'home.html', {'csv_description': description, 'csv_data': df_display, 'plot_tmp': plt_tmp})

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error processing CSV:", str(e))
                return HttpResponse("Error processing CSV file.")
            
        if action == "remove_missing_values":
            # Placeholder logic to remove missing values
            try:
                # Load CSV file and remove missing values
                if csv_path:
                    df = pd.read_csv(csv_path)
                    
                    df.dropna(inplace=True)

                    # Overwrite the existing CSV file with the cleaned DataFrame
                    df.to_csv(csv_path, index=False)

                    description = {
                        'shape': df.shape,
                        'attributes': list(df.columns),
                        'data_types': df.dtypes,
                        'null_values': df.isnull().sum(),
                        'duplicates': df.duplicated().sum(),
                        'statistics': df.describe()
                    }

                    # Display only the first few rows
                    df_display = df.head()

                    # Generate a simple visualization using Matplotlib
                    plt.figure(figsize=(10, 6))
                    sns.countplot(x='is_canceled', data=df)  # Replace 'your_column' with the column you want to visualize
                    plt.title('is_canceled')
                    plt.xlabel('X-axis Label')
                    plt.ylabel('Y-axis Label')
                    plt.tight_layout()

                    # Save the plot to a temporary file
                    plt_tmp = os.path.join('visualization', 'plot_tmp.png')
                    plt.savefig(os.path.join(settings.STATICFILES_DIRS[0], plt_tmp))

                    # Render the cleaned data
                    return render(request, 'home.html', {'csv_description': description, 'csv_data': df_display, 'plot_tmp': plt_tmp})

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error processing CSV:", str(e))
                return HttpResponse("Error processing CSV file.")
    

        elif action == "remove_duplicates":
            # Placeholder logic to remove duplicates
            try:
                # Load CSV file and remove duplicates
                if csv_path:
                    df = pd.read_csv(csv_path)
                    df.drop_duplicates(inplace=True)

                    # Overwrite the existing CSV file with the cleaned DataFrame
                    df.to_csv(csv_path, index=False)

                    description = {
                        'shape': df.shape,
                        'attributes': list(df.columns),
                        'data_types': df.dtypes,
                        'null_values': df.isnull().sum(),
                        'duplicates': df.duplicated().sum(),
                        'statistics': df.describe()
                    }

                    # Display only the first few rows
                    df_display = df.head()

                    # Generate a simple visualization using Matplotlib
                    plt.figure(figsize=(10, 6))
                    sns.countplot(x='is_canceled', data=df)  # Replace 'your_column' with the column you want to visualize
                    plt.title('Count of Something')
                    plt.xlabel('X-axis Label')
                    plt.ylabel('Y-axis Label')
                    plt.tight_layout()

                    # Save the plot to a temporary file
                    plt_tmp = os.path.join('visualization', 'plot_tmp.png')
                    plt.savefig(os.path.join(settings.STATICFILES_DIRS[0], plt_tmp))

                    # Render the cleaned data
                    return render(request, 'home.html', {'csv_description': description, 'csv_data': df_display, 'plot_tmp': plt_tmp})

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error processing CSV:", str(e))
                return HttpResponse("Error processing CSV file.")
            
        elif action == "visualize" and column:
            # Placeholder logic to remove duplicates
            try:
                # Load CSV file and remove duplicates
                if csv_path:
                    df = pd.read_csv(csv_path)

                    description = {
                        'shape': df.shape,
                        'attributes': list(df.columns),
                        'data_types': df.dtypes,
                        'null_values': df.isnull().sum(),
                        'duplicates': df.duplicated().sum(),
                        'statistics': df.describe()
                    }

                    # Display only the first few rows
                    df_display = df.head()

                    visualizations = visualize_all_columns(df[column])

                    # Render the cleaned data
                    return render(request, 'home.html', {'csv_description': description, 'csv_data': df_display, 'visualizations': visualizations})

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error processing CSV:", str(e))
                return HttpResponse("Error processing CSV file.")
        
        elif action == "train_model":
            # Train a decision tree classifier model
            try:
                if csv_path:
                    # Load the CSV file
                    df = pd.read_csv(csv_path)

                    # Data preprocessing
                    df = df.drop(['company'], axis=1)
                    df['children'] = df['children'].fillna(0)
                    df['hotel'] = df['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
                    df['arrival_date_month'] = df['arrival_date_month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                                                                            'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})
                    # Define function to determine if guest is a family
                    # Define function to determine if guest provided a deposit
                    def is_family(data):
                        if ((data['adults'] > 0) & (data['children'] > 0)):
                            val = 1
                        elif ((data['adults'] > 0) & (data['babies'] > 0)):
                            val = 1
                        else:
                            val = 0
                        return val

                    def did_deposit(data):
                        if ((data['deposit_type'] == 'No Deposit') | (data['deposit_type'] == 'Refundable')):
                            return 0
                        else:
                            return 1

                    df["is_family"] = df.apply(is_family, axis=1)
                    df["total_customer"] = df["adults"] + df["children"] + df["babies"]
                    df["deposit_given"] = df.apply(did_deposit, axis=1)
                    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

                    # Drop more unwanted/messy columns
                    df = df.drop(columns=['adults', 'babies', 'children', 'deposit_type', 'reservation_status_date'])
                    df = pd.get_dummies(data=df, columns=['meal', 'market_segment', 'distribution_channel',
                                                        'reserved_room_type', 'assigned_room_type', 'customer_type',
                                                        'reservation_status'])

                    le = LabelEncoder()
                    df['country'] = le.fit_transform(df['country'])
                    df = df.drop(columns=['reservation_status_Canceled', 'reservation_status_Check-Out', 'reservation_status_No-Show'],
                                axis=1)

                    y = df["is_canceled"]
                    X = df.drop(["is_canceled"], axis=1)

                    # Split data into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    X_test.to_csv(csv_path, index=False)

                    # Initialize and train the decision tree classifier
                    clf = DecisionTreeClassifier()
                    clf.fit(X_train, y_train)

                    # Make predictions on the test set
                    y_pred = clf.predict(X_test)

                    # Calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)

                    # Save the trained model to a file
                    model_file = os.path.join(settings.MEDIA_ROOT, 'decision-tree-model', 'decision_tree_model.joblib')
                    joblib.dump(clf, model_file)

                    # Render a response with the accuracy score and model path
                    return HttpResponse(f"Decision Tree Classifier Accuracy: {accuracy}. Model saved at: {model_file}")

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error training model:", str(e))
                return HttpResponse("Error training model.")

    else:
        form = CSVUploadForm()

    return render(request, 'home.html')

def test_model(request):
    if request.method == 'POST' and request.FILES['model-file'] and request.FILES['csv-file']:
        model_file = request.FILES['model-file']
        csv_file = request.FILES['csv-file']

        # Save uploaded files to a temporary location
        model_file_path = os.path.join(settings.MEDIA_ROOT, 'temp_model.joblib')
        with open(model_file_path, 'wb+') as destination:
            for chunk in model_file.chunks():
                destination.write(chunk)

        csv_file_path = os.path.join(settings.MEDIA_ROOT, 'temp_data.csv')
        with open(csv_file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        try:
            # Load the model
            clf = joblib.load(model_file_path)

            # Load CSV data
            df = pd.read_csv(csv_file_path)

            # Select first 20 entries
            df_first_20 = df.head(20)

            # Make predictions
            predictions = clf.predict(df_first_20)

            # Add predictions as a new column to DataFrame
            df_first_20['predictions'] = predictions

            # Convert CSV data to HTML table
            csv_data_html = df_first_20.to_html(index=False, classes='table table-striped')

            # Convert predictions to a list
            prediction_list = predictions.tolist()

            return render(request, 'test_model.html', {'csv_data': csv_data_html, 'predictions': prediction_list, 'columns': df.columns})

        except Exception as e:
            error_message = str(e)
            return render(request, 'test_model.html', {'error_message': error_message})

    else:
        error_message = 'Invalid request method or missing files'
        return render(request, 'test_model.html', {'error_message': error_message})
    
def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')
