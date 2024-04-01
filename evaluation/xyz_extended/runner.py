import psycopg2
import os
from tqdm.auto import tqdm
import argparse
import itertools
import yaml

# Database configuration
DB_CONFIG = {
    'dbname': 'tpcds1',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',  # or your host
    'port': '5432'        # or your port
}

# Directory containing .sql files
SQL_DIR = '/home/ubuntu/tpc-ds-queries-1'

######
## Execute SQL commands from a file using a given connection.
#####
def execute_sql_file(filepath, conn): 
    with open(filepath, 'r') as file:
        sql_str = file.read()
        execute_sql(sql_str, conn)

######
## Execute SQL commands from a string using a given connection.
#####
def execute_sql(sql_str, conn):
    try:
        cur = conn.cursor()
        cur.execute(sql_str)
        cur.close()
        conn.commit()
    except Exception as e:
        print(f"Error executing {filepath}: {e}")
        conn.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter_file', type=str, default='./parameters.yml')
    args = parser.parse_args()

    # Load the YAML configuration file
    with open(args.parameter_file, 'r') as file:
        config = yaml.safe_load(file)

    param_combinations = []

    for param in config['parameters']:
        param_name = param['name']
        param_settings = param['settings']
        param_combinations.append(param_settings)

    # Iterate over all parameter combinations
    for combination in itertools.product(*param_combinations):
        # Create a dictionary to store the parameter values for this combination
        param_values = {}
        for i, param in enumerate(config['parameters']):
            param_name = param['name']
            param_values[param_name] = combination[i]

        print(param_values)

        for i in range(config['runs']):

            # Connect to the database
            try:
                conn = psycopg2.connect(**DB_CONFIG)
            except Exception as e:
                print(f"Unable to connect to the database: {e}")
                return
        
            # Set the parameters
            print(f"Repetition {i+1}/{config['runs']}")
            for param_name, param_value in param_values.items():
                execute_sql(f"SET {param_name} = {param_value};", conn)
            
            # Iterate through each .sql file in the directory
            for filename in tqdm(os.listdir(SQL_DIR)):
                if filename.endswith(".sql"):
                    filepath = os.path.join(SQL_DIR, filename)
                    execute_sql_file(filepath, conn)

            # Close the database connection
            conn.close()

if __name__ == "__main__":
    main()
