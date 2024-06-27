import argparse
import pickle
import pandas as pd
import numpy as np

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    # Load the model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Define categorical columns
    categorical = ['PULocationID', 'DOLocationID']

    # Read the data
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = read_data(filename, categorical)

    # Transform and predict
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Create ride_id
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Add predictions to DataFrame
    df['predictions'] = y_pred

    # Save results
    output_file = f'Yellow_taxi_{year}_{month:02d}_results.parquet'
    df_result = df[['ride_id', 'predictions']]
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # Calculate and print mean predicted duration
    mean_pred_duration = np.mean(y_pred)
    print(f'Mean predicted duration for {year}-{month:02d}: {mean_pred_duration:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to process')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to process')

    args = parser.parse_args()
    main(args.year, args.month)
