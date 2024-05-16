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

def generate_pie_chart(data):
    plt.rcParams['figure.figsize'] = 8, 8
    labels = data['hotel'].value_counts().index.tolist()
    sizes = data['hotel'].value_counts().tolist()
    colors = ["darkorange", "lightskyblue"]

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
    
    plt_tmp = os.path.join('visualization', 'pie_chart_plot_tmp.png')
    plt.savefig(os.path.join(settings.STATICFILES_DIRS[0], plt_tmp))

    plt.close()
    
    return plt_tmp

def generate_countplot(data):
    plt.figure(figsize=(20, 5))
    sns.countplot(data=data, x="arrival_date_month", hue="hotel", order=["January", "February", "March", "April", "May", "June",
                                                                          "July", "August", "September", "October", "November", "December"]).set_title(
        'Illustration of Number of Visitors Each Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    
    plt_tmp = os.path.join('visualization', 'count_plot_tmp.png')
    plt.savefig(os.path.join(settings.STATICFILES_DIRS[0], plt_tmp))

    plt.close()
    
    return plt_tmp

def generate_cancelation_countplot(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='is_canceled').set_title("Cancellation Overview")
    plt.xlabel("Bookings Cancelled")
    plt.ylabel("Count")
    
    plt_tmp = os.path.join('visualization', 'cancellation_count_plot_tmp.png')
    plt.savefig(os.path.join(settings.STATICFILES_DIRS[0], plt_tmp))

    plt.close()
    
    return plt_tmp

def generate_repeated_guest_countplot(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x="is_repeated_guest", hue="hotel").set_title("Illustration of number of repeated guests")
    plt.xlabel("Repeated Guest")
    plt.ylabel("Count")
    
    plt_tmp = os.path.join('visualization', 'cancellation_count_plot_tmp.png')
    plt.savefig(os.path.join(settings.STATICFILES_DIRS[0], plt_tmp))

    plt.close()
    
    return plt_tmp

def generate_visualizations(data):
    visualization_paths = []

    # Generate and store pie chart visualization
    pie_chart_path = generate_pie_chart(data)
    visualization_paths.append(pie_chart_path)

    # Generate and store countplot visualization 1
    countplot1_path = generate_countplot(data)
    visualization_paths.append(countplot1_path)

    # Generate and store countplot visualization 2
    cancelation_countplot_path = generate_cancelation_countplot(data)
    visualization_paths.append(cancelation_countplot_path)

    # Generate and store countplot visualization 3
    repeated_guest_countplot_path = generate_repeated_guest_countplot(data)
    visualization_paths.append(repeated_guest_countplot_path)

    return visualization_paths

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

def generate_pivot_tables(df, description):
    # Create binary columns for previous cancellations and booking changes
    df['prev_cancel'] = df['previous_cancellations'].apply(lambda x: 'True' if x > 0 else 'False')
    df['book_changed'] = df['booking_changes'].apply(lambda x: 'True' if x > 0 else 'False')

    # Create bins for lead_time
    max_lead_time = df['lead_time'].max()
    min_lead_time = df['lead_time'].min()
    lead_time_bins = [min_lead_time, min_lead_time + (max_lead_time - min_lead_time) / 4,
                      min_lead_time + 2 * (max_lead_time - min_lead_time) / 4,
                      min_lead_time + 3 * (max_lead_time - min_lead_time) / 4,
                      max_lead_time]
    lead_time_labels = [f"{lead_time_bins[i]}-{lead_time_bins[i+1]}" for i in range(len(lead_time_bins)-1)]
    df['lead_time_binned'] = pd.cut(df['lead_time'], bins=lead_time_bins, labels=lead_time_labels, include_lowest=True)

    # Create bins for adr
    max_adr = df['adr'].max()
    min_adr = df['adr'].min()
    adr_bins = [min_adr, min_adr + (max_adr - min_adr) / 4,
                min_adr + 2 * (max_adr - min_adr) / 4,
                min_adr + 3 * (max_adr - min_adr) / 4,
                max_adr]
    adr_labels = [f"{adr_bins[i]}-{adr_bins[i+1]}" for i in range(len(adr_bins)-1)]
    df['adr_binned'] = pd.cut(df['adr'], bins=adr_bins, labels=adr_labels, include_lowest=True)

    # Specify the pivot table configurations
    pivot_specs = [
        ('feature_1', ['lead_time_binned', 'prev_cancel']),
        ('feature_2', ['lead_time_binned', 'book_changed']),
        ('feature_3', ['hotel']),
        ('feature_4', ['adr_binned', 'prev_cancel']),
        ('feature_5', ['adr_binned', 'book_changed']),
        ('demo_1', ['adults', 'children']),
        ('demo_2', ['adults', 'babies']),
        ('demo_3', ['adults', 'total_of_special_requests'])
    ]

    # Dictionary to store HTML representations of the pivot tables
    pivot_tables_html = {}
    for name, index in pivot_specs:
        if name == 'feature_3':
            pivot_table = pd.pivot_table(df, values='is_canceled', index=index, aggfunc='sum')
        else:
            pivot_table = pd.pivot_table(df, values='is_canceled', index=index, aggfunc='mean')

        pivot_tables_html[name] = pivot_table.to_html()

    # Context for rendering the HTML template
    context = {'csv_description': description}
    context.update(pivot_tables_html)

    return context

def process_csv(request):
    original_csv = os.path.join(settings.MEDIA_ROOT, 'uploaded_csv', 'data.csv')
    csv_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_csv', 'data.csv')

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            print("CSV file received:", csv_file.name)

            
            with open(original_csv, 'wb') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)

            with open(csv_path, 'wb') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)


            try:
                df = pd.read_csv(csv_path)

                is_cancelled_description = df['is_canceled'].describe()
                description = {
                    'shape': df.shape,
                    'attributes': list(df.columns),
                    'data_types': df.dtypes,
                    'null_values': df.isnull().sum(),
                    'duplicates': df.duplicated().sum(),
                    'statistics': df.describe(),
                    'col': is_cancelled_description
                }

                return render(request, 'analysis.html', {'form': form, 'csv_description': description})

            except Exception as e:
                print("Error reading CSV:", str(e))
                description = "Error reading CSV file."

                return render(request, 'analysis.html', {'form': form, 'csv_description': description})
            
    elif request.method == 'GET':
        action = request.GET.get('action')
        column = request.GET.get('column')
        page = request.GET.get('page')

        if action == "describe" and not column:
            try:
                if csv_path:
                    df = pd.read_csv(csv_path)

                    description = {
                        'shape': df.shape,
                        'attributes': list(df.columns),
                        'data_types': df.dtypes,
                        'null_values': df.isnull().sum(),
                        'duplicates': df.duplicated().sum(),
                        'statistics': df.describe(),
                    }

                    return render(request, 'analysis.html', {'csv_description': description })
                    
                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error processing CSV:", str(e))
                return HttpResponse("Error processing CSV file.")

        elif action == "describe" and page:
            try:
                if csv_path:
                    df = pd.read_csv(csv_path)

                    if column in df.columns:
                        column_description = df[column].describe()
                        description = {
                            'shape': df.shape,
                            'attributes': list(df.columns),
                            'data_types': df.dtypes,
                            'null_values': df.isnull().sum(),
                            'duplicates': df.duplicated().sum(),
                            'statistics': df.describe(),
                            'col': column_description
                        }

                        if page == "analysis":
                            context = generate_pivot_tables(df, description)

                            return render(request, 'analysis-pivot.html', context)
                        
                        elif page == "describe":
                            return render(request, 'analysis.html', {'csv_description': description })
                        
                        elif page == "visualize":
                            visualize = generate_visualizations()
                            return render(request, 'visualize-analysis.html', {'csv_description': description, 'visualization': visualize })
                    
                    else:
                        return HttpResponse("Error: Column not found in the CSV file.")

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error processing CSV:", str(e))
                return HttpResponse("Error processing CSV file.")

        elif action == "remove_duplicates":
            try:
                if csv_path:
                    df = pd.read_csv(csv_path)
                    df.drop_duplicates(inplace=True)

                    df.to_csv(csv_path, index=False)

                    description = {
                        'shape': df.shape,
                        'attributes': list(df.columns),
                        'data_types': df.dtypes,
                        'null_values': df.isnull().sum(),
                        'duplicates': df.duplicated().sum(),
                        'statistics': df.describe()
                    }

                    return render(request, 'analysis.html', {'csv_description': description})

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error processing CSV:", str(e))
                return HttpResponse("Error processing CSV file.")
            
        elif action == "remove_missing_values":
            try:
                if csv_path:
                    df = pd.read_csv(csv_path)
                    
                    columns_to_drop = ['company', 'agent']
                    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

                    if existing_columns_to_drop:
                        df.drop(existing_columns_to_drop, axis=1, inplace=True)

                    df.dropna(inplace=True)
                    df.to_csv(csv_path, index=False)

                    description = {
                        'shape': df.shape,
                        'attributes': list(df.columns),
                        'data_types': df.dtypes,
                        'null_values': df.isnull().sum(),
                        'duplicates': df.duplicated().sum(),
                        'statistics': df.describe()
                    }

                    return render(request, 'analysis.html', {'csv_description': description})

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error processing CSV:", str(e))
                return HttpResponse("Error processing CSV file.")
        
        elif action == "train_model":
            try:
                if csv_path:
                    df = pd.read_csv(csv_path)

                    df = df.drop(['company'], axis=1)
                    df['children'] = df['children'].fillna(0)
                    df['hotel'] = df['hotel'].map({'Resort Hotel': 0, 'City Hotel': 1})
                    df['arrival_date_month'] = df['arrival_date_month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                                                                            'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})
                   
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

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    X_test.to_csv(csv_path, index=False)

                    clf = DecisionTreeClassifier()
                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)

                    model_file = os.path.join(settings.MEDIA_ROOT, 'decision-tree-model', 'decision_tree_model.joblib')
                    joblib.dump(clf, model_file)

                    return HttpResponse(f"Decision Tree Classifier Accuracy: {accuracy}. Model saved at: {model_file}")

                else:
                    return HttpResponse("Error: CSV file not uploaded.")

            except Exception as e:
                print("Error training model:", str(e))
                return HttpResponse("Error training model.")
            
        elif action == "analysis":
            try:
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

                    context = generate_pivot_tables(df, description)

                    return render(request, 'analysis-pivot.html', context)

            except Exception as e:
                    print("Analysis Error:", str(e))
                    return HttpResponse("Analysis Error.")

        elif action == "visualize":
            try:
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

                    visualize = generate_visualizations(df)
                    print(visualize)
                    
                    return render(request, 'visualization-analysis.html', {'csv_description': description, 'visualization': visualize })
                
            except Exception as e:
                    print("Visualization Error:", str(e))
                    return HttpResponse("Visualization Error.")
    else:
        form = CSVUploadForm()

    return render(request, 'analysis.html')

def test_model(request):
    if request.method == 'POST' and request.FILES['model-file'] and request.FILES['csv-file']:
        model_file = request.FILES['model-file']
        csv_file = request.FILES['csv-file']

        model_file_path = os.path.join(settings.MEDIA_ROOT, 'temp_model.joblib')
        with open(model_file_path, 'wb+') as destination:
            for chunk in model_file.chunks():
                destination.write(chunk)

        csv_file_path = os.path.join(settings.MEDIA_ROOT, 'temp_data.csv')
        with open(csv_file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        try:
            clf = joblib.load(model_file_path)

            df = pd.read_csv(csv_file_path)

            df_first_20 = df.head(20)

            predictions = clf.predict(df_first_20)

            df_first_20['predictions'] = predictions

            csv_data_html = df_first_20.to_html(index=False, classes='table table-striped')

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

def test_view(request):
    return render(request, 'test.html')
